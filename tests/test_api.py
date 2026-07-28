from __future__ import annotations

from types import SimpleNamespace

import numpy as np
from fastapi.testclient import TestClient

from api import Runtime, create_app


class FakeEmbedder:
    def encode(self, texts, **kwargs):
        assert len(texts) == 1
        return np.array([[0.1, 0.2]], dtype=float)


class FakeModel:
    def __init__(self, positive_probability: float):
        self.positive_probability = positive_probability

    def predict_proba(self, features):
        count = len(features)
        return np.tile(
            [1.0 - self.positive_probability, self.positive_probability],
            (count, 1),
        )


class FakeResponses:
    def create(self, **kwargs):
        assert kwargs["model"] == "gpt-4o-mini"
        assert "tweet_no_confiable" in kwargs["input"]
        return SimpleNamespace(output_text="Explicación prudente. No es asesoramiento financiero.")


def make_runtime(
    probability: float = 0.8,
    with_openai: bool = False,
) -> Runtime:
    client = (
        SimpleNamespace(responses=FakeResponses())
        if with_openai
        else None
    )
    return Runtime(
        models=[FakeModel(probability), FakeModel(probability)],
        threshold=0.5,
        embed_model_name="fake-embedder",
        embedder=FakeEmbedder(),
        openai_client=client,
        openai_model="gpt-4o-mini",
    )


def test_health_reports_runtime_state():
    app = create_app(lambda: make_runtime())
    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "model_loaded": True,
        "embedding_model": "fake-embedder",
        "openai_configured": False,
    }


def test_analyze_positive_with_openai():
    app = create_app(lambda: make_runtime(0.8, with_openai=True))
    with TestClient(app) as client:
        response = client.post(
            "/analyze",
            json={"text": "Bitcoin adoption keeps growing."},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["sentiment"] == "positive"
    assert body["confidence"] == 0.8
    assert body["positive_probability"] == 0.8
    assert body["chatbot_source"] == "openai"
    assert "asesoramiento financiero" in body["disclaimer"]


def test_negative_confidence_is_probability_of_predicted_class():
    app = create_app(lambda: make_runtime(0.2))
    with TestClient(app) as client:
        response = client.post("/analyze", json={"text": "Bitcoin is crashing."})

    assert response.status_code == 200
    body = response.json()
    assert body["sentiment"] == "negative"
    assert body["confidence"] == 0.8
    assert body["positive_probability"] == 0.2
    assert body["chatbot_source"] == "fallback"


def test_rejects_blank_tweet():
    app = create_app(lambda: make_runtime())
    with TestClient(app) as client:
        response = client.post("/analyze", json={"text": "   "})

    assert response.status_code == 422


def test_optional_api_key_protects_analyze(monkeypatch):
    monkeypatch.setenv("APP_API_KEY", "secret")
    app = create_app(lambda: make_runtime())

    with TestClient(app) as client:
        unauthorized = client.post("/analyze", json={"text": "Bitcoin"})
        authorized = client.post(
            "/analyze",
            json={"text": "Bitcoin"},
            headers={"X-API-Key": "secret"},
        )

    assert unauthorized.status_code == 401
    assert authorized.status_code == 200
