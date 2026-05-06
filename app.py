"""
Traductor Español → Francés con Qwen2-0.5B
==========================================
Requisitos:
    pip install torch transformers flask sentencepiece accelerate

Uso:
    python app.py
    Abrir http://localhost:5000 en el navegador
"""

from flask import Flask, request, jsonify, Response
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import time
import os

app = Flask(__name__)

# ── Configuración del modelo ────────────────────────────────────────────────
MODEL_ID = "Qwen/Qwen2-0.5B-Instruct"
model     = None
tokenizer = None


def load_model():
    global model, tokenizer
    print(f"Cargando modelo {MODEL_ID} …")
    print("(La primera vez descarga ~1 GB desde Hugging Face, espera un momento)")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=torch.float32,        # compatible con CPU
        trust_remote_code=True,
    )
    model.eval()
    print("Modelo listo ✓")


def translate_es_fr(text: str) -> str:
    """Traduce texto de español a francés usando Qwen2-0.5B-Instruct."""
    messages = [
        {
            "role": "system",
            "content": (
                "Eres un traductor profesional de español a francés. "
                "Devuelve ÚNICAMENTE el texto traducido al francés, "
                "sin explicaciones, sin etiquetas, sin comillas."
            ),
        },
        {
            "role": "user",
            "content": f"Traduce al francés:\n\n{text}",
        },
    ]

    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer([prompt], return_tensors="pt").to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Solo los tokens generados (sin el prompt)
    new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
    result = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    return result


# ── Rutas ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    """Sirve el archivo index.html."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    html_path = os.path.join(base_dir, "index.html")
    with open(html_path, "r", encoding="utf-8") as f:
        html_content = f.read()
    return Response(html_content, mimetype="text/html")


@app.route("/translate", methods=["POST"])
def translate_endpoint():
    if model is None:
        return jsonify({"error": "Modelo no cargado aún."}), 503

    data = request.get_json(force=True)
    text = (data.get("text") or "").strip()

    if not text:
        return jsonify({"error": "No se recibió texto."}), 400

    try:
        t0 = time.time()
        translation = translate_es_fr(text)
        elapsed_ms  = round((time.time() - t0) * 1000)
        return jsonify({"translation": translation, "elapsed_ms": elapsed_ms})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── Inicio ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    load_model()
    print("\n🌐 Abre http://localhost:5000 en tu navegador\n")
    app.run(host="0.0.0.0", port=5000, debug=False)
