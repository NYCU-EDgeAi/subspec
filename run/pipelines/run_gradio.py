import gradio as gr
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
import logging
import os

from tqdm import trange
import nvtx

def main(builder, host: str = "127.0.0.1", port: int = 7860, share: bool = False):
    # set logging level by environment variable
    LOGLEVEL = os.environ.get("LOGLEVEL", "INFO").upper()
    logging.basicConfig(level=LOGLEVEL)
    
    # helper for device check
    args = builder.args
    
    print("Building model...")
    generator, tokenizer, past_kv, draft_past_kv = builder.build()
    print("Model built successfully.")

    def _content_to_text(content):
        """Best-effort conversion of Gradio 'content' payloads into a plain string.

        In Gradio 6.x, chat message content can sometimes arrive as a small dict
        like {'text': 'Hello', 'type': 'text'} or other structured payloads.
        Our tokenizer chat template expects a string.
        """
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        if isinstance(content, dict):
            # Common Gradio payload shape for text messages
            if "text" in content:
                return str(content.get("text") or "")
            # Fallback: try a few likely keys before stringifying everything
            for key in ("content", "value", "message"):
                if key in content and isinstance(content[key], str):
                    return content[key]
            return str(content)
        if isinstance(content, (list, tuple)):
            return "\n".join(_content_to_text(x) for x in content if _content_to_text(x))
        return str(content)

    def generate_response(history, temperature, max_length):
        # Gradio 6.x Chatbot history is messages. Content can sometimes be structured
        # (e.g. {'text': 'Hello', 'type': 'text'}), so normalize to plain strings.
        messages = []
        for item in history or []:
            if isinstance(item, dict) and "role" in item and "content" in item:
                content = _content_to_text(item.get("content"))
                if content:
                    messages.append({"role": item["role"], "content": content})
                continue

            # Support objects with role/content attributes (some Gradio message types)
            role = getattr(item, "role", None)
            content = getattr(item, "content", None)
            if role is not None and content is not None:
                content_text = _content_to_text(content)
                if content_text:
                    messages.append({"role": str(role), "content": content_text})
                continue

            # Support tuple/list pairs: (user, assistant) or [user, assistant, ...]
            if isinstance(item, (tuple, list)) and len(item) >= 2:
                u, b = item[0], item[1]
                u_text = _content_to_text(u)
                b_text = _content_to_text(b)
                if u_text:
                    messages.append({"role": "user", "content": u_text})
                if b_text:
                    messages.append({"role": "assistant", "content": b_text})
                continue
        
        # Reset KV cache for fresh generation
        past_kv.reset()
        if draft_past_kv is not None:
            draft_past_kv.reset()

        input_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt").to(args.device)
        
        with sdpa_kernel(backends=[SDPBackend.MATH]):
            output_ids = generator.generate(
                input_ids, 
                temperature=temperature, 
                max_length=max_length, 
                do_sample=args.do_sample, 
                past_key_values=past_kv, 
                draft_past_key_values=draft_past_kv
            )
            
        generated_text = tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True)
        return generated_text

    # Warmup
    if args.warmup_iter > 0:
        print("Warming up... It will take some time for the first few iterations to run.")
        with nvtx.annotate("Warming up"):
            is_profiling = generator.profiling
            generator.profiling = False
            for i in trange(args.warmup_iter, desc='Warming up'):
                input_message = "Write an essay about large language models."
                messages = [{"role": "user", "content": input_message}]
                tokenizer.use_default_system_prompt = True
                with nvtx.annotate("Warm up"):
                    input_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt").to(args.device)
                    with sdpa_kernel(backends=[SDPBackend.MATH]):
                        generator.generate(input_ids, temperature=args.temperature, max_length=args.max_length, do_sample=args.do_sample, past_key_values=past_kv, draft_past_key_values=draft_past_kv)
                
                past_kv.reset()
                if draft_past_kv is not None:
                    draft_past_kv.reset()
            generator.profiling = is_profiling

    # Gradio Interface
    with gr.Blocks() as demo:
        gr.Markdown(f"# SpecDecodes Playground: {args.method}")
        
        with gr.Row():
            with gr.Column():
                chatbot = gr.Chatbot(height=600)
                msg = gr.Textbox(label="Message")
                
                with gr.Accordion("Parameters", open=False):
                    temp_slider = gr.Slider(minimum=0.0, maximum=1.0, value=args.temperature, label="Temperature")
                    max_len_slider = gr.Slider(minimum=1, maximum=4096, value=args.max_length, step=8, label="Max Length")
                
                clear = gr.ClearButton([msg, chatbot])

        def user(user_message, history):
            history = history or []
            return "", history + [{"role": "user", "content": _content_to_text(user_message)}]

        def bot(history, temperature, max_length):
            history = history or []
            bot_response = generate_response(history, temperature, max_length)
            history.append({"role": "assistant", "content": bot_response})
            return history

        msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False).then(
            bot, [chatbot, temp_slider, max_len_slider], chatbot
        )
        
    print("Launching Gradio app...")
    demo.launch(server_name=host, server_port=int(port), share=bool(share))
