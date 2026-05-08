# 🌐 Traductor Neural — Español → Francés

Aplicación de traducción local usando **Qwen2-0.5B-Instruct**. Corre completamente en tu máquina, sin internet ni API keys una vez instalada.

## Screenshot

<img width="1916" height="1027" alt="Captura de pantalla 2026-05-01 210037" src="https://github.com/user-attachments/assets/26088b04-3ee7-4397-b9a8-88609969a4b1" />



## Ejemplos probados

| Español | Francés |
|---|---|
| Me gusta el fútbol | Je aime le football |
| ¿Cómo estás? | Comment vas-tu ? |
| ¿Qué hora es? | Quelle heure est-il ? |

## Requisitos

- Python 3.9+
- ~1 GB de espacio (descarga del modelo en el primer uso)
- No requiere GPU

## Instalación y uso

```bash
# 1. Instalar dependencias
pip install torch transformers flask sentencepiece accelerate

# 2. Correr el servidor
python app.py

# 3. Abrir en el navegador
http://localhost:5000
```

## Archivos
