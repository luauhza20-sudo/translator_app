import os
import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# =========================
# Modelo local
# =========================
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-0.5B-Instruct")

print(f"Cargando modelo: {MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    trust_remote_code=True
)

generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer
)

# =========================
# Traducción
# =========================
def translate_text(text, target_language):
    if not text or not text.strip():
        return "Escribe un texto para traducir."

    prompt = f"""
Translate the following sentence into {target_language}.
Return only the translation and nothing else.

Sentence: {text}
"""

    try:
        result = generator(
            prompt,
            max_new_tokens=60,
            do_sample=False,
            temperature=0.1,
            return_full_text=False
        )[0]["generated_text"].strip()

        return result if result else "No se pudo generar traducción."

    except Exception as e:
        return f"Error: {str(e)}"

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
    demo.launch(server_name="127.0.0.1", server_port=7860)
