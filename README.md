# Bitcoin Sentiment Chatbot

API FastAPI que clasifica el sentimiento de un tweet con un ensemble de
LightGBM y genera una explicación prudente mediante OpenAI. El resultado es una
señal de tono textual, no una predicción del precio de Bitcoin ni asesoramiento
financiero.

## Cómo funciona

1. `text_utils.py` limpia el texto.
2. `all-mpnet-base-v2` genera un vector de 768 dimensiones.
3. Los cinco modelos del artefacto `kfold.joblib` promedian su probabilidad.
4. El umbral guardado en el propio artefacto decide entre `positive` y
   `negative`.
5. OpenAI explica el resultado. Si la API no está configurada o falla, se
   devuelve una explicación local segura.

El modelo incluido declara un umbral de `0.4377209325491541`. Las métricas
históricas anotadas en el proyecto no sustituyen una evaluación reproducible y
actual sobre datos no vistos. No debe utilizarse este servicio para automatizar
operaciones financieras.

## Ejecución local

Requisitos: Python 3.11.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.7.1
python -m pip install -r requirements-dev.txt
Copy-Item .env.example .env
```

Exporta al menos `OPENAI_API_KEY` en la terminal si quieres una respuesta
generada por OpenAI. La aplicación no lee `.env` automáticamente para evitar
configuraciones implícitas en producción.

```powershell
$env:OPENAI_API_KEY = "..."
python -m uvicorn api:app --host 127.0.0.1 --port 8080
```

Documentación interactiva: `http://127.0.0.1:8080/docs`.

Ejemplo:

```bash
curl -X POST http://127.0.0.1:8080/analyze \
  -H "Content-Type: application/json" \
  -H "X-API-Key: TU_APP_API_KEY" \
  -d '{"text":"Bitcoin adoption keeps growing."}'
```

`X-API-Key` solo es obligatorio cuando se define `APP_API_KEY`.

## Pruebas

```powershell
python -m pytest -q
python -m compileall -q api.py text_utils.py kfold.py
```

Para probar el modelo real sin llamar a OpenAI:

```powershell
Remove-Item Env:OPENAI_API_KEY -ErrorAction SilentlyContinue
python -m uvicorn api:app --host 127.0.0.1 --port 8080
```

El campo `chatbot_source` será `fallback`; la clasificación seguirá usando el
modelo real.

## Docker

La imagen descarga `all-mpnet-base-v2` durante la construcción y activa el modo
offline durante la ejecución. Así, un reinicio no depende de Hugging Face.

```bash
docker build -t bitcoin-sentiment-api .
docker run --rm -p 8080:8080 \
  -e OPENAI_API_KEY="..." \
  -e APP_API_KEY="una-clave-larga-y-aleatoria" \
  bitcoin-sentiment-api
```

Se usa un solo worker porque cada proceso cargaría otra copia de PyTorch, el
transformer y el ensemble en memoria.

## Despliegue en DigitalOcean App Platform

El archivo `.do/app.yaml` está preparado para:

- región `fra` (Frankfurt);
- construcción mediante el `Dockerfile`;
- puerto HTTP `8080` en `0.0.0.0`;
- comprobaciones de salud en `/health`;
- una instancia compartida de 1 vCPU y 2 GiB;
- despliegue automático desde `deploy/digitalocean-app-platform`.

Pasos:

1. En DigitalOcean, crea una aplicación desde GitHub y autoriza el repositorio
   `tostadito33/Clasificador-binario`.
2. Selecciona la rama `deploy/digitalocean-app-platform`. App Platform detectará
   el `Dockerfile`; también puedes importar `.do/app.yaml`.
3. En **Settings → App-Level Environment Variables**, añade
   `OPENAI_API_KEY` como **Encrypted/Secret**.
4. Añade `APP_API_KEY` como secreto para evitar que terceros consuman tu cuota
   de OpenAI. Los clientes deberán enviarlo en `X-API-Key`.
5. Si existe un frontend, define `CORS_ORIGINS` con sus orígenes exactos
   separados por comas, por ejemplo `https://app.example.com`.
6. Despliega y comprueba `GET /health`, `GET /docs` y `POST /analyze`.

No guardes claves reales en `.do/app.yaml`, `.env` ni GitHub. La configuración
de 2 GiB evita el plan predeterminado de 512 MiB, insuficiente para este modelo.

## Variables de entorno

| Variable | Obligatoria | Uso |
| --- | --- | --- |
| `OPENAI_API_KEY` | Para respuesta OpenAI | Secreto del proyecto OpenAI |
| `OPENAI_MODEL` | No | Modelo; por defecto `gpt-4o-mini` |
| `OPENAI_TIMEOUT_SECONDS` | No | Timeout; por defecto `20` |
| `APP_API_KEY` | Recomendada | Protege `POST /analyze` |
| `CORS_ORIGINS` | Solo con frontend | Orígenes exactos separados por comas |
| `MODEL_PATH` | No | Artefacto; por defecto `kfold.joblib` |

## Entrenamiento

El dataset no se copia a la imagen de producción. Para reentrenar:

```powershell
python kfold.py --train --data Tweets.parquet --save kfold.joblib --folds 5
python kfold.py --predict-file tweets.txt --load kfold.joblib
```

Todo artefacto nuevo debe conservar `embed_model_name`; la API verifica al
arrancar que su dimensión coincida con la esperada por el ensemble.
