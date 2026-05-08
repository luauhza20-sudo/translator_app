import os
import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# =========================
# Modelo local
# =========================
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-0.5B-Instruct")

print(f"Cargando modelo: {MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float32,
    device_map="cpu",
    trust_remote_code=True
)

def translate_text(text, target_language):
    if not text or not text.strip():
        return "Escribe un texto para traducir."

    prompt = f"""Translate the following sentence into {target_language}.
Return only the translation and nothing else.

Sentence: {text}
"""

    messages = [
        {"role": "system", "content": "You are a translation assistant."},
        {"role": "user", "content": prompt}
    ]

    input_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(input_text, return_tensors="pt")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=60,
            do_sample=False,
            temperature=0.1,
            pad_token_id=tokenizer.eos_token_id
        )

    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return result.strip()

# =========================
# Interfaz
# =========================
with gr.Blocks(title="Qwen Translator") as demo:
    gr.Markdown("# Traductor con Qwen2-0.5B")
    gr.Markdown("App local para traducir texto de un idioma a otro usando un modelo pequeño Instruct.")

    with gr.Row():
        text_input = gr.Textbox(
            label="Texto de entrada",
            placeholder="Ejemplo: I like soccer",
            lines=3
        )

    with gr.Row():
        target_lang = gr.Dropdown(
            choices=["Spanish", "English", "French", "German", "Portuguese"],
            value="Spanish",
            label="Idioma de destino"
        )

    btn = gr.Button("Traducir")
    output = gr.Textbox(label="Traducción", lines=4)

    btn.click(
        fn=translate_text,
        inputs=[text_input, target_lang],
        outputs=output
    )

    gr.Markdown("## Ejemplos que debe soportar")
    gr.Markdown("""
- I like soccer
- How are you?
- What time is it?
    """)

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7861)
