from typing import Any
from smolagents import Tool, MessageRole, TokenUsage, ChatMessage, Model

class SpecDecodesModel(Model):
    def __init__(
        self,
        generator,
        tokenizer,
        past_key_values=None,
        draft_past_key_values=None,
        device=None,
        **kwargs,
    ):
        self.model = generator
        self.tokenizer = tokenizer
        self.past_key_values = past_key_values
        self.draft_past_key_values = draft_past_key_values
        self.device = device

        model_id = getattr(generator, 'model_id', 'specdecodes-model')
        super().__init__(
            flatten_messages_as_text=True, model_id=model_id, **kwargs
        )

    def _prepare_completion_args(
        self,
        messages: list[ChatMessage | dict],
        stop_sequences: list[str] | None = None,
        tools_to_call_from: list[Tool] | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        completion_kwargs = self._prepare_completion_kwargs(
            messages=messages,
            stop_sequences=stop_sequences,
            tools_to_call_from=tools_to_call_from,
            tool_choice=None,
            **kwargs,
        )
        messages = completion_kwargs.pop("messages")
        stop_sequences = completion_kwargs.pop("stop", None)
        tools = completion_kwargs.pop("tools", None)
        
        prompt_tensor = (self.processor if hasattr(self, "processor") else self.tokenizer).apply_chat_template(
            messages,
            tools=tools,
            return_tensors="pt",
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
        )
        prompt_tensor = prompt_tensor.to(self.device)  # type: ignore
        if hasattr(prompt_tensor, "input_ids"):
            prompt_tensor = prompt_tensor["input_ids"]

        return dict(
            input_ids=prompt_tensor,
            past_key_values=self.past_key_values,
            draft_past_key_values=self.draft_past_key_values,
            **completion_kwargs,
        )

    def generate(
        self,
        messages: list[ChatMessage | dict],
        stop_sequences: list[str] | None = None,
        response_format: dict[str, str] | None = None,
        tools_to_call_from: list[Tool] | None = None,
        **kwargs,
    ) -> ChatMessage:
        if response_format is not None:
            raise ValueError("Transformers does not support structured outputs, use VLLMModel for this.")
        generation_kwargs = self._prepare_completion_args(
            messages=messages,
            stop_sequences=stop_sequences,
            tools_to_call_from=tools_to_call_from,
            **kwargs,
        )
        count_prompt_tokens = generation_kwargs["input_ids"].shape[1]  # type: ignore
        out = self.model.generate(
            **generation_kwargs,
        )
        generated_tokens = out[0, count_prompt_tokens:]
        output_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

        return ChatMessage(
            role=MessageRole.ASSISTANT,
            content=output_text,
            raw={
                "out": output_text,
                "completion_kwargs": {key: value for key, value in generation_kwargs.items() if key != "input_ids"},
            },
            token_usage=TokenUsage(
                input_tokens=count_prompt_tokens,
                output_tokens=len(generated_tokens),
            ),
        )