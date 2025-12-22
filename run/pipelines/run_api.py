import time
import uuid
import threading
import queue
import json
from typing import Any, Dict, List, Literal, Optional, Union

import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
	role: Literal["system", "user", "assistant", "tool", "developer"]
	content: Union[str, List[Any], Dict[str, Any], None] = None


class ChatCompletionsRequest(BaseModel):
	model: Optional[str] = None
	messages: List[ChatMessage]
	temperature: Optional[float] = None
	max_tokens: Optional[int] = Field(default=None, ge=1)
	stream: Optional[bool] = False
	stop: Optional[Union[str, List[str]]] = None


class CompletionsRequest(BaseModel):
	model: Optional[str] = None
	prompt: Union[str, List[str]]
	temperature: Optional[float] = None
	max_tokens: Optional[int] = Field(default=None, ge=1)
	stream: Optional[bool] = False
	stop: Optional[Union[str, List[str]]] = None


def _content_to_text(content: Any) -> str:
	if content is None:
		return ""
	if isinstance(content, str):
		return content
	if isinstance(content, dict):
		if "text" in content:
			return str(content.get("text") or "")
		for key in ("content", "value", "message"):
			if key in content and isinstance(content[key], str):
				return content[key]
		return str(content)
	if isinstance(content, (list, tuple)):
		parts: List[str] = []
		for x in content:
			t = _content_to_text(x)
			if t:
				parts.append(t)
		return "\n".join(parts)
	return str(content)


def _apply_stop(text: str, stop: Optional[Union[str, List[str]]]) -> str:
	if not stop:
		return text
	stops = [stop] if isinstance(stop, str) else list(stop)
	cut_idx = None
	for s in stops:
		if not s:
			continue
		idx = text.find(s)
		if idx != -1:
			cut_idx = idx if cut_idx is None else min(cut_idx, idx)
	return text if cut_idx is None else text[:cut_idx]


def _count_tokens_best_effort(tokenizer, text: str) -> int:
	try:
		return int(len(tokenizer.encode(text)))
	except Exception:
		return 0


def _build_input_ids(tokenizer, messages: List[Dict[str, str]], device: str) -> "torch.LongTensor":
	"""Build input_ids for chat-style messages.

	Prefers tokenizer.apply_chat_template when available; falls back to a simple
	string prompt for tokenizers that don't implement chat templates.
	"""
	if hasattr(tokenizer, "apply_chat_template") and callable(getattr(tokenizer, "apply_chat_template")):
		return tokenizer.apply_chat_template(
			messages,
			tokenize=True,
			add_generation_prompt=True,
			return_tensors="pt",
		).to(device)

	# Fallback: basic role-tagged transcript.
	parts: List[str] = []
	for m in messages:
		role = (m.get("role") or "user").strip()
		content = m.get("content") or ""
		parts.append(f"{role}: {content}")
	parts.append("assistant:")
	prompt = "\n".join(parts)

	try:
		enc = tokenizer(prompt, return_tensors="pt")
		input_ids = enc["input_ids"]
		return input_ids.to(device)
	except Exception as e:
		raise HTTPException(
			status_code=400,
			detail=f"Tokenizer does not support chat templates or basic tokenization: {e}",
		)


def create_app(builder) -> FastAPI:
	app = FastAPI(title="SubSpec OpenAI-Compatible API")

	# Build once at startup and reuse.
	generator, tokenizer, past_kv, draft_past_kv = builder.build()
	args = builder.args

	# GPU models + shared KV caches are not safe for concurrent requests.
	generation_lock = threading.Lock()

	model_id = getattr(args, "llm_path", None) or getattr(args, "method", "model")

	@app.get("/health")
	def health() -> Dict[str, Any]:
		return {"status": "ok", "method": getattr(args, "method", None), "model": model_id}

	@app.get("/v1/models")
	def list_models() -> Dict[str, Any]:
		now = int(time.time())
		return {
			"object": "list",
			"data": [
				{
					"id": model_id,
					"object": "model",
					"created": now,
					"owned_by": "subspec",
				}
			],
		}

	@app.post("/v1/chat/completions")
	def chat_completions(req: ChatCompletionsRequest):
		stop_strings = None
		if req.stop is not None:
			stop_strings = [req.stop] if isinstance(req.stop, str) else list(req.stop)

		messages: List[Dict[str, str]] = []
		for m in req.messages:
			content_text = _content_to_text(m.content)
			# Keep roles as-is; tokenizer chat template typically supports system/user/assistant.
			# If content is empty, skip.
			if content_text:
				messages.append({"role": m.role, "content": content_text})

		if not messages:
			raise HTTPException(status_code=400, detail="messages must not be empty")

		temperature = float(req.temperature) if req.temperature is not None else float(getattr(args, "temperature", 0.0) or 0.0)
		max_tokens = req.max_tokens

		created = int(time.time())
		completion_id = f"chatcmpl-{uuid.uuid4().hex}" 

		if req.stream:
			created = int(time.time())
			completion_id = f"chatcmpl-{uuid.uuid4().hex}"

			token_queue: "queue.Queue[Optional[str]]" = queue.Queue()
			error_queue: "queue.Queue[BaseException]" = queue.Queue()

			def stream_callback(token_ids):
				# token_ids: torch.LongTensor, possibly on GPU, shape (1, n)
				try:
					ids = token_ids.detach().to("cpu")
					if ids.dim() == 2:
						ids_list = ids[0].tolist()
					else:
						ids_list = ids.tolist()
					# Drop EOS tokens
					eos = getattr(tokenizer, "eos_token_id", None)
					if eos is not None:
						ids_list = [i for i in ids_list if i != eos]
					if not ids_list:
						return
					piece = tokenizer.decode(ids_list, skip_special_tokens=True)
					if piece:
						token_queue.put(piece)
				except BaseException as e:
					error_queue.put(e)

			def run_generation():
				try:
					with generation_lock:
						past_kv.reset()
						if draft_past_kv is not None:
							draft_past_kv.reset()

						input_ids = _build_input_ids(tokenizer, messages, args.device)

						if max_tokens is None:
							max_length_total = int(getattr(args, "max_length", 2048) or 2048)
						else:
							max_length_total = int(min(int(getattr(args, "max_length", 2048) or 2048), input_ids.shape[1] + int(max_tokens)))

						with torch.no_grad():
							generator.generate(
								input_ids,
								temperature=temperature,
								max_length=max_length_total,
								do_sample=getattr(args, "do_sample", False),
								stop_strings=stop_strings,
								stream_callback=stream_callback,
								past_key_values=past_kv,
								draft_past_key_values=draft_past_kv,
							)
				except BaseException as e:
					error_queue.put(e)
				finally:
					token_queue.put(None)

			t = threading.Thread(target=run_generation, daemon=True)
			t.start()

			def sse(data_obj: Union[Dict[str, Any], str]) -> str:
				if isinstance(data_obj, str):
					payload = data_obj
				else:
					payload = json.dumps(data_obj, ensure_ascii=False)
				return f"data: {payload}\n\n"

			def event_iter():
				# Initial role chunk
				yield sse(
					{
						"id": completion_id,
						"object": "chat.completion.chunk",
						"created": created,
						"model": req.model or model_id,
						"choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
					}
				)

				while True:
					try:
						err = error_queue.get_nowait()
						raise err
					except queue.Empty:
						pass

					piece = token_queue.get()
					if piece is None:
						break
					yield sse(
						{
							"id": completion_id,
							"object": "chat.completion.chunk",
							"created": created,
							"model": req.model or model_id,
							"choices": [{"index": 0, "delta": {"content": piece}, "finish_reason": None}],
						}
					)

				# Final chunk
				yield sse(
					{
						"id": completion_id,
						"object": "chat.completion.chunk",
						"created": created,
						"model": req.model or model_id,
						"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
					}
				)
				yield sse("[DONE]")

			return StreamingResponse(event_iter(), media_type="text/event-stream")

		with generation_lock:
			# Reset KV caches for fresh generation
			past_kv.reset()
			if draft_past_kv is not None:
				draft_past_kv.reset()

			input_ids = _build_input_ids(tokenizer, messages, args.device)

			# generator.generate expects max_length as total sequence length.
			if max_tokens is None:
				max_length_total = int(getattr(args, "max_length", 2048) or 2048)
			else:
				max_length_total = int(min(int(getattr(args, "max_length", 2048) or 2048), input_ids.shape[1] + int(max_tokens)))

			with torch.no_grad():
				output_ids = generator.generate(
					input_ids,
					temperature=temperature,
					max_length=max_length_total,
					do_sample=getattr(args, "do_sample", False),
					stop_strings=stop_strings,
					past_key_values=past_kv,
					draft_past_key_values=draft_past_kv,
				)

			text = tokenizer.decode(output_ids[0][input_ids.shape[1] :], skip_special_tokens=True)
			text = _apply_stop(text, req.stop)

		# Best-effort usage (OpenAI returns token counts)
		prompt_text = "".join([m["content"] for m in messages])
		prompt_tokens = _count_tokens_best_effort(tokenizer, prompt_text)
		completion_tokens = _count_tokens_best_effort(tokenizer, text)

		return {
			"id": completion_id,
			"object": "chat.completion",
			"created": created,
			"model": req.model or model_id,
			"choices": [
				{
					"index": 0,
					"message": {"role": "assistant", "content": text},
					"finish_reason": "stop",
				}
			],
			"usage": {
				"prompt_tokens": prompt_tokens,
				"completion_tokens": completion_tokens,
				"total_tokens": prompt_tokens + completion_tokens,
			},
		}

	@app.post("/v1/completions")
	def completions(req: CompletionsRequest):
		stop_strings = None
		if req.stop is not None:
			stop_strings = [req.stop] if isinstance(req.stop, str) else list(req.stop)

		if isinstance(req.prompt, list):
			if len(req.prompt) != 1:
				raise HTTPException(status_code=400, detail="Only a single prompt is supported")
			prompt = req.prompt[0]
		else:
			prompt = req.prompt

		temperature = float(req.temperature) if req.temperature is not None else float(getattr(args, "temperature", 0.0) or 0.0)
		max_tokens = req.max_tokens

		created = int(time.time())
		completion_id = f"cmpl-{uuid.uuid4().hex}" 

		# Treat plain prompt as a single user turn.
		messages = [{"role": "user", "content": prompt}]

		if req.stream:
			token_queue: "queue.Queue[Optional[str]]" = queue.Queue()
			error_queue: "queue.Queue[BaseException]" = queue.Queue()

			def stream_callback(token_ids):
				try:
					ids = token_ids.detach().to("cpu")
					if ids.dim() == 2:
						ids_list = ids[0].tolist()
					else:
						ids_list = ids.tolist()
					eos = getattr(tokenizer, "eos_token_id", None)
					if eos is not None:
						ids_list = [i for i in ids_list if i != eos]
					if not ids_list:
						return
					piece = tokenizer.decode(ids_list, skip_special_tokens=True)
					if piece:
						token_queue.put(piece)
				except BaseException as e:
					error_queue.put(e)

			def run_generation():
				try:
					with generation_lock:
						past_kv.reset()
						if draft_past_kv is not None:
							draft_past_kv.reset()

						input_ids = _build_input_ids(tokenizer, messages, args.device)

						if max_tokens is None:
							max_length_total = int(getattr(args, "max_length", 2048) or 2048)
						else:
							max_length_total = int(min(int(getattr(args, "max_length", 2048) or 2048), input_ids.shape[1] + int(max_tokens)))

						with torch.no_grad():
							generator.generate(
								input_ids,
								temperature=temperature,
								max_length=max_length_total,
								do_sample=getattr(args, "do_sample", False),
								stop_strings=stop_strings,
								stream_callback=stream_callback,
								past_key_values=past_kv,
								draft_past_key_values=draft_past_kv,
							)
				except BaseException as e:
					error_queue.put(e)
				finally:
					token_queue.put(None)

			t = threading.Thread(target=run_generation, daemon=True)
			t.start()

			def sse(data_obj: Union[Dict[str, Any], str]) -> str:
				if isinstance(data_obj, str):
					payload = data_obj
				else:
					payload = json.dumps(data_obj, ensure_ascii=False)
				return f"data: {payload}\n\n"

			def event_iter():
				while True:
					try:
						err = error_queue.get_nowait()
						raise err
					except queue.Empty:
						pass

					piece = token_queue.get()
					if piece is None:
						break
					yield sse(
						{
							"id": completion_id,
							"object": "text_completion",
							"created": created,
							"model": req.model or model_id,
							"choices": [{"index": 0, "text": piece, "finish_reason": None, "logprobs": None}],
						}
					)

				yield sse(
					{
						"id": completion_id,
						"object": "text_completion",
						"created": created,
						"model": req.model or model_id,
						"choices": [{"index": 0, "text": "", "finish_reason": "stop", "logprobs": None}],
					}
				)
				yield sse("[DONE]")

			return StreamingResponse(event_iter(), media_type="text/event-stream")

		with generation_lock:
			past_kv.reset()
			if draft_past_kv is not None:
				draft_past_kv.reset()

			input_ids = _build_input_ids(tokenizer, messages, args.device)

			if max_tokens is None:
				max_length_total = int(getattr(args, "max_length", 2048) or 2048)
			else:
				max_length_total = int(min(int(getattr(args, "max_length", 2048) or 2048), input_ids.shape[1] + int(max_tokens)))

			with torch.no_grad():
				output_ids = generator.generate(
					input_ids,
					temperature=temperature,
					max_length=max_length_total,
					do_sample=getattr(args, "do_sample", False),
					stop_strings=stop_strings,
					past_key_values=past_kv,
					draft_past_key_values=draft_past_kv,
				)

			text = tokenizer.decode(output_ids[0][input_ids.shape[1] :], skip_special_tokens=True)
			text = _apply_stop(text, req.stop)

		prompt_tokens = _count_tokens_best_effort(tokenizer, prompt)
		completion_tokens = _count_tokens_best_effort(tokenizer, text)

		return {
			"id": completion_id,
			"object": "text_completion",
			"created": created,
			"model": req.model or model_id,
			"choices": [
				{
					"index": 0,
					"text": text,
					"finish_reason": "stop",
					"logprobs": None,
				}
			],
			"usage": {
				"prompt_tokens": prompt_tokens,
				"completion_tokens": completion_tokens,
				"total_tokens": prompt_tokens + completion_tokens,
			},
		}

	@app.exception_handler(Exception)
	def unhandled_exception_handler(_, exc: Exception):
		# Return OpenAI-ish error shape.
		return JSONResponse(
			status_code=500,
			content={
				"error": {
					"message": str(exc),
					"type": exc.__class__.__name__,
					"param": None,
					"code": None,
				}
			},
		)

	return app


def main(builder, host: str = "0.0.0.0", port: int = 8000, workers: int = 1, log_level: str = "info"):
	import uvicorn

	app = create_app(builder)

	# Use a single worker by default for GPU safety.
	uvicorn.run(app, host=host, port=port, workers=workers, log_level=log_level)

