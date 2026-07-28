from __future__ import annotations

import re


def clean_text(value: str) -> str:
    """Normaliza un tweet del mismo modo durante entrenamiento e inferencia."""
    if not isinstance(value, str):
        value = str(value)
    value = value.lower()
    value = re.sub(r"http\S+", "", value)
    value = re.sub(r"www\.[^\s]+", "", value)
    value = re.sub(r"@\w+", "", value)
    value = re.sub(r"\$\w+", "", value)
    value = re.sub(r"#", "", value)
    value = re.sub(r"[^\w\s\.,!?áéíóúüñ]", " ", value)
    value = re.sub(r"(.)\1{2,}", r"\1\1", value)
    value = re.sub(r"\d{2,}", " ", value)
    return re.sub(r"\s+", " ", value).strip()
