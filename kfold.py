"""
Entrenamiento y utilidad interactiva para clasificación de sentimiento.
- Usa `Datasets/Tweets.parquet` por defecto (puedes pasar --data)
- Divide en train/val/test (estratificado)
- Genera embeddings con `sentence-transformers/all-MiniLM-L6-v2`
- Entrena LightGBM con early stopping
- Hace sweep de umbrales y selecciona uno equilibrado (minimiza FP+FN con restricción de recall)
- Guarda artefactos en `best_model.joblib` (o la ruta indicada)
- Soporta `--interactive`, `--predict-file` y `--eval-only` modos

Se añadió StratifiedKFold para entrenar un ensemble (votación por promedio de probabilidades)

Requisitos:
 pip install sentence-transformers lightgbm scikit-learn pandas joblib

Uso rápido:
 python kfold.py --train --data Tweets.parquet --save kfold.joblib
 python kfold.py --predict-file tweets.txt --load kfold.joblib
 python kfold.py --interactive --load kfold.joblib
"""
from __future__ import annotations
import argparse
import os
import re
import sys
from typing import List
import joblib
import numpy as np
import pandas as pd

try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None

try:
    import lightgbm as lgb
except Exception:
    lgb = None

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import confusion_matrix, precision_recall_curve, precision_recall_fscore_support


# ----------------------
# Utils
# ----------------------

def clean_text(s: str) -> str:
    if not isinstance(s, str):
        s = str(s)
    s = s.lower()
    s = re.sub(r"http\S+", "", s)
    s = re.sub(r"www\.[^\s]+", "", s)
    s = re.sub(r"@\w+", "", s)
    s = re.sub(r"\$\w+", "", s)
    s = re.sub(r"#", "", s)
    s = re.sub(r"[^\w\s\.,!?áéíóúüñ]", " ", s)
    s = re.sub(r"(.)\1{2,}", r"\1\1", s)
    s = re.sub(r"\d{2,}", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def load_dataset(path: str, text_col_candidates=None, label_col_candidates=None, drop_neutral=True):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    if path.lower().endswith('.parquet'):
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)

    if text_col_candidates is None:
        text_col_candidates = ['tweet_text', 'text', 'sentence', 'tweet', 'Sentence']
    if label_col_candidates is None:
        label_col_candidates = ['tweet_sentiment', 'sentiment', 'label', 'Sentiment']

    text_col = next((c for c in df.columns if c in text_col_candidates), None)
    label_col = next((c for c in df.columns if c in label_col_candidates), None)

    if text_col is None or label_col is None:
        raise ValueError(f"No se encontraron columnas válidas en {path}. Columnas: {list(df.columns)}")

    df = df[[text_col, label_col]].dropna()
    df.columns = ['text', 'label']
    df['text'] = df['text'].astype(str).apply(clean_text)
    df = df[df['text'].str.len() >= 2].reset_index(drop=True)

    # normalizar etiquetas
    df['label_norm'] = df['label'].astype(str).str.lower().str.strip()
    # map neutral -> drop or keep
    if drop_neutral:
        df = df[~df['label_norm'].isin(['neutral', 'neutro'])]
    # Keep only two classes if possible
    uniq = sorted(df['label_norm'].unique())
    if set(uniq) <= {'negative', 'positive'}:
        df['label_bin'] = df['label_norm'].map({'negative': 0, 'positive': 1})
    else:
        # try to map spanish words
        mapping = {}
        for u in uniq:
            if 'neg' in u:
                mapping[u] = 0
            elif 'pos' in u:
                mapping[u] = 1
            elif 'negativo' in u or 'malo' in u:
                mapping[u] = 0
            elif 'positivo' in u or 'bueno' in u:
                mapping[u] = 1
        if mapping:
            df['label_bin'] = df['label_norm'].map(mapping)
            df = df.dropna(subset=['label_bin'])
            df['label_bin'] = df['label_bin'].astype(int)
        else:
            raise ValueError(f"Etiquetas no binarias detectadas: {uniq}")

    return df['text'].tolist(), df['label_bin'].astype(int).tolist(), df


# ----------------------
# Embedding
# ----------------------

def embed_texts(sentences: List[str], model_name: str = 'all-mpnet-base-v2', batch_size: int = 256):
    if SentenceTransformer is None:
        raise ImportError('Instala sentence-transformers: pip install sentence-transformers')
    model = SentenceTransformer(model_name)
    embeddings = []
    for i in range(0, len(sentences), batch_size):
        batch = sentences[i:i+batch_size]
        emb = model.encode(batch, show_progress_bar=False)
        embeddings.append(emb)
    return np.vstack(embeddings), model


# ----------------------
# Train & eval
# ----------------------

def train_lightgbm(X_train, y_train, X_val, y_val, params=None):
    if lgb is None:
        raise ImportError('Instala lightgbm: pip install lightgbm')
    unique, counts = np.unique(y_train, return_counts=True)
    if len(unique) == 2:
        idx_pos = list(unique).index(1) if 1 in unique else None
        idx_neg = list(unique).index(0) if 0 in unique else None
        n_pos = int(counts[idx_pos]) if idx_pos is not None else 1
        n_neg = int(counts[idx_neg]) if idx_neg is not None else 1
        scale_pos_weight = max(1.0, n_neg / max(1, n_pos))
    else:
        scale_pos_weight = 1.0

    default_params = dict(
        objective='binary',
        learning_rate=0.05,
        n_estimators=2000,
        num_leaves=127,
        max_depth=12,
        min_child_samples=20,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        scale_pos_weight=scale_pos_weight,
        n_jobs=-1,
        verbosity=-1,
        random_state=42,
    )
    if params:
        default_params.update(params)

    clf = lgb.LGBMClassifier(**default_params)
    eval_sets = [(X_train, y_train), (X_val, y_val)]
    try:
        clf.fit(X_train, y_train, eval_set=eval_sets, eval_metric='binary_logloss', callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)])
    except Exception:
        clf.fit(X_train, y_train)
    return clf


def sweep_thresholds(y_true, y_prob, step=0.01):
    rows = []
    thresholds = np.arange(0.0, 1.0 + 1e-9, step)
    for t in thresholds:
        preds = (y_prob >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, preds, labels=[0, 1]).ravel()
        prec, rec, f1, _ = precision_recall_fscore_support(y_true, preds, average='binary', zero_division=0)
        rows.append({'threshold': float(t), 'tp': int(tp), 'fp': int(fp), 'fn': int(fn), 'tn': int(tn), 'precision': float(prec), 'recall': float(rec), 'f1': float(f1)})
    return rows


def choose_threshold_by_balance(y_true, y_prob, min_recall=0.25, fallback_min_recall=0.15):
    prec, rec, thr = precision_recall_curve(y_true, y_prob)
    if len(thr) == 0:
        return 0.5
    candidates = []
    for i, t in enumerate(thr):
        p = float(prec[i+1])
        r = float(rec[i+1])
        preds = (y_prob >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, preds, labels=[0,1]).ravel()
        fp_rate = fp / (fp + tn) if (fp + tn) > 0 else 1.0
        fn_rate = fn / (fn + tp) if (fn + tp) > 0 else 1.0
        combined = fp_rate + fn_rate
        candidates.append((t, combined, fp_rate, fn_rate, p, r))
    filtered = [c for c in candidates if c[5] >= min_recall]
    if not filtered:
        filtered = [c for c in candidates if c[5] >= fallback_min_recall]
    if not filtered:
        filtered = [c for c in candidates]
    filtered.sort(key=lambda x: (x[1], x[2]))
    return float(filtered[0][0])


# ----------------------
# CLI
# ----------------------

def train_and_save(data_path, save_path='best_model.joblib', embed_model_name='all-mpnet-base-v2', n_splits=5):
    print('Loading dataset:', data_path)
    texts, labels, df = load_dataset(data_path)
    print('Counts:', df['label_bin'].value_counts().to_dict())

    # split
    X_train_texts, X_temp_texts, y_train, y_temp = train_test_split(texts, labels, test_size=0.30, random_state=42, stratify=labels)
    X_val_texts, X_test_texts, y_val, y_test = train_test_split(X_temp_texts, y_temp, test_size=0.5, random_state=42, stratify=y_temp)
    print(f'Train/Val/Test sizes: {len(X_train_texts)}/{len(X_val_texts)}/{len(X_test_texts)}')

    # embed
    if SentenceTransformer is None:
        raise ImportError('Instala sentence-transformers')
    print('Generating embeddings with', embed_model_name)
    st = SentenceTransformer(embed_model_name)
    X_train_emb = st.encode(X_train_texts, show_progress_bar=True)
    X_val_emb = st.encode(X_val_texts, show_progress_bar=True)
    X_test_emb = st.encode(X_test_texts, show_progress_bar=True)

    # train ensemble using StratifiedKFold on TRAIN set
    print(f'Training ensemble with StratifiedKFold (n_splits={n_splits})...')
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    models = []
    y_train_arr = np.array(y_train)
    for fold, (tr_idx, va_idx) in enumerate(skf.split(X_train_emb, y_train_arr)):
        print(f' Fold {fold+1}/{n_splits}')
        X_tr, X_va = X_train_emb[tr_idx], X_train_emb[va_idx]
        y_tr, y_va = y_train_arr[tr_idx], y_train_arr[va_idx]
        m = train_lightgbm(X_tr, y_tr, X_va, y_va)
        models.append(m)

    # ensemble predictions (average probabilities)
    def ensemble_proba(models, X):
        probs = np.vstack([m.predict_proba(X)[:,1] if hasattr(m, 'predict_proba') else m.predict(X) for m in models])
        return probs.mean(axis=0)

    # choose threshold using val (ensemble)
    y_val_prob = ensemble_proba(models, X_val_emb)
    best_thresh = choose_threshold_by_balance(np.array(y_val), y_val_prob, min_recall=0.25, fallback_min_recall=0.15)
    print('Chosen threshold:', best_thresh)

    # evaluate on test (ensemble)
    y_test_prob = ensemble_proba(models, X_test_emb)
    y_test_pred = (y_test_prob >= best_thresh).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, y_test_pred).ravel()
    prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_test_pred, average='binary', zero_division=0)
    acc = float((np.array(y_test) == y_test_pred).mean())
    print('\nTest results:')
    print('Accuracy:', acc)
    print('Precision:', prec, 'Recall:', rec, 'F1:', f1)
    print('Confusion matrix:')
    print(np.array([[tn, fp],[fn, tp]]))

    # save artifacts
    artifacts = {
        'models': models,
        'embed_model_name': embed_model_name,
        'threshold': best_thresh,
        'n_splits': n_splits,
    }
    joblib.dump(artifacts, save_path)
    print('Saved artifacts to', save_path)


def _load_artifacts(model_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError(model_path)
    artifacts = joblib.load(model_path)
    models = artifacts.get('models')
    embed_model_name = artifacts.get('embed_model_name')
    thresh = artifacts.get('threshold', 0.5)
    return models, embed_model_name, thresh


def predict_file(file_path, model_path='best_model.joblib'):
    models, embed_model_name, thresh = _load_artifacts(model_path)
    if SentenceTransformer is None:
        raise ImportError('Instala sentence-transformers')
    st = SentenceTransformer(embed_model_name)

    texts = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            t = line.strip()
            if t:
                texts.append(clean_text(t))
    if not texts:
        print('No texts found in', file_path)
        return
    X_emb = st.encode(texts, show_progress_bar=False)

    if isinstance(models, list):
        probs = np.vstack([m.predict_proba(X_emb)[:,1] if hasattr(m, 'predict_proba') else m.predict(X_emb) for m in models]).mean(axis=0)
        preds = (probs >= thresh).astype(int)
    else:
        model = models
        probs = model.predict_proba(X_emb)[:,1] if hasattr(model, 'predict_proba') else None
        preds = (probs >= thresh).astype(int) if probs is not None else model.predict(X_emb)
        
    pos_count = 0
    neg_count = 0

    for t, p, pr in zip(texts, preds, probs if probs is not None else [None]*len(texts)):
        label = 'positive' if int(p) == 1 else 'negative'

        if label == 'positive':
            pos_count += 1
        else:
            neg_count += 1

        if pr is not None:
            print(f"{label}\t{pr:.4f}\t{t}")
        else:
            print(f"{label}\t{t}")

    print("\n--- RESUMEN ---")
    print(f"Positivos: {pos_count}")
    print(f"Negativos: {neg_count}")
    print(f"Total: {pos_count + neg_count}")



def interactive(model_path='best_model.joblib'):
    models, embed_model_name, thresh = _load_artifacts(model_path)
    if SentenceTransformer is None:
        raise ImportError('Instala sentence-transformers')
    st = SentenceTransformer(embed_model_name)
    print('Interactive mode. Type a tweet (or quit/exit).')
    try:
        while True:
            txt = input('> ').strip()
            if not txt:
                continue
            if txt.lower() in ('quit', 'exit'):
                break
            t = clean_text(txt)
            emb = st.encode([t], show_progress_bar=False)
            if isinstance(models, list):
                probs = np.vstack([m.predict_proba(emb)[:,1] if hasattr(m, 'predict_proba') else m.predict(emb) for m in models]).mean(axis=0)
                pr = float(probs[0])
                lab = 'positive' if pr >= thresh else 'negative'
                print(lab, f"{pr:.4f}")
            else:
                model = models
                if hasattr(model, 'predict_proba'):
                    pr = float(model.predict_proba(emb)[:,1][0])
                    lab = 'positive' if pr >= thresh else 'negative'
                    print(lab, f"{pr:.4f}")
                else:
                    p = model.predict(emb)[0]
                    lab = 'positive' if int(p) == 1 else 'negative'
                    print(lab)
    except KeyboardInterrupt:
        print('\nExit')


def main():
    parser = argparse.ArgumentParser(description='Train/eval/predict sentiment model (con StratifiedKFold ensemble)')
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--data', default='Tweets.parquet.parquet')
    parser.add_argument('--save', default='best_model.joblib')
    parser.add_argument('--predict-file', help='File with one tweet per line')
    parser.add_argument('--interactive', action='store_true')
    parser.add_argument('--load', help='Load model for predict/interactive')
    parser.add_argument('--folds', type=int, default=5, help='Número de folds para StratifiedKFold (ensemble)')
    args = parser.parse_args()

    if args.train:
        train_and_save(args.data, save_path=args.save, n_splits=args.folds)
        return
    if args.predict_file:
        predict_file(args.predict_file, model_path=args.load or args.save)
        return
    if args.interactive:
        interactive(model_path=args.load or args.save)
        return
    parser.print_help()


if __name__ == '__main__':
    main()
