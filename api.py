import os
import joblib
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from openai import OpenAI

from kfold import clean_text

# ----------------------
# Carga de modelo ML
# ----------------------

artifacts = joblib.load("kfold.joblib")

models = artifacts["models"]
threshold = artifacts["threshold"]

# Forzamos un modelo ligero para producción
embed_model_name = "sentence-transformers/all-MiniLM-L6-v2"
st = SentenceTransformer(embed_model_name)

# ----------------------
# OpenAI
# ----------------------

client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY")
)

# ----------------------
# FastAPI
# ----------------------

app = FastAPI(title="Investor Sentiment Chatbot API")

class TweetRequest(BaseModel):
    text: str


def ensemble_predict_proba(models, X):
    """
    Promedia las probabilidades del ensemble (StratifiedKFold)
    """
    return np.vstack([
        m.predict_proba(X)[:, 1] for m in models
    ]).mean(axis=0)


def investor_chat_response(tweet, sentiment, confidence):
    """
    Genera la respuesta del chatbot usando OpenAI,
    basándose en el resultado de TU modelo ML.
    """

    prompt = f"""
Eres un inversor profesional con experiencia en mercados financieros.

Un modelo de machine learning ha analizado el siguiente tweet:

Tweet:
"{tweet}"

Resultado del análisis:
- Sentimiento: {sentiment}
- Confianza: {confidence}

Explica al usuario, de forma clara, prudente y profesional,
si tendría sentido invertir o no según este tweet.

Incluye siempre un aviso de que NO es asesoramiento financiero.
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "Hablas como un inversor profesional, prudente y racional."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.6,
    )

    return response.choices[0].message.content.strip()


@app.post("/analyze")
def analyze_tweet(req: TweetRequest):
    # Limpieza y embedding
    text_clean = clean_text(req.text)
    emb = st.encode([text_clean], show_progress_bar=False)

    # Predicción ML
    prob = float(ensemble_predict_proba(models, emb)[0])
    sentiment = "positive" if prob >= threshold else "negative"

    # Respuesta del chatbot (OpenAI)
    chat_response = investor_chat_response(
        tweet=req.text,
        sentiment=sentiment,
        confidence=round(prob, 3)
    )

    return {
        "sentiment": sentiment,
        "confidence": round(prob, 3),
        "chatbot_response": chat_response
    }
