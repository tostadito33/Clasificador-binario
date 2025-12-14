from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
from sentence_transformers import SentenceTransformer
from kfold import clean_text

# Cargar artefactos del modelo
artifacts = joblib.load("kfold.joblib")
models = artifacts["models"]
threshold = artifacts["threshold"]
embed_model_name = "sentence-transformers/all-MiniLM-L6-v2"

st = SentenceTransformer(embed_model_name)

app = FastAPI(title="Investor Tweet Sentiment API")

class TweetRequest(BaseModel):
    text: str

def ensemble_predict_proba(models, X):
    return np.vstack([
        m.predict_proba(X)[:, 1] for m in models
    ]).mean(axis=0)

@app.post("/analyze")
def analyze_tweet(req: TweetRequest):
    text_clean = clean_text(req.text)
    emb = st.encode([text_clean], show_progress_bar=False)

    prob = float(ensemble_predict_proba(models, emb)[0])
    sentiment = "positive" if prob >= threshold else "negative"

    if sentiment == "positive":
        advice = (
            "Como inversor, interpreto este tweet como positivo. "
            "Puede reflejar confianza del mercado, aunque siempre "
            "conviene contrastarlo con otros indicadores."
        )
    else:
        advice = (
            "Desde un punto de vista inversor, el sentimiento es negativo. "
            "Yo sería prudente y evitaría entrar hasta que el contexto mejore."
        )

    return {
        "sentiment": sentiment,
        "confidence": round(prob, 3),
        "investor_response": advice
    }
