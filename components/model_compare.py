import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np

# ✅ 공통 데이터 전처리 함수
def prepare_data(df):
    base_cols = ["steps", "calories", "very_active_minutes", "moderately_active_minutes", "distance"]
    df = df.dropna(subset=base_cols + ["id"])
    df_user = df.groupby("id")[base_cols].mean().reset_index()

    # CHURN 기준 라벨 생성
    score = 0
    score += (df_user["steps"] < 6400).astype(int)
    score += (df_user["calories"] < 1800).astype(int)
    score += (df_user["very_active_minutes"] < 6).astype(int) * 2
    score += (df_user["moderately_active_minutes"] < 8).astype(int)
    score += (df_user["distance"] < 4600).astype(int)
    df_user["CHURNED"] = (score >= 4).astype(int)

    X = df_user[base_cols]
    y = df_user["CHURNED"]
    return X, y

# ✅ 기본 모델 학습 및 평가
def evaluate_basic_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.3, random_state=42)
    model = XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    return {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred), 4),
        "recall": round(recall_score(y_test, y_pred), 4),
        "f1_score": round(f1_score(y_test, y_pred), 4)
    }

# ✅ 교차검증 기반 모델 평가
def evaluate_cv_model(X, y, cv=5):
    model = XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42)
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)

    return {
        "accuracy": round(cross_val_score(model, X, y, scoring='accuracy', cv=skf).mean(), 4),
        "precision": round(cross_val_score(model, X, y, scoring='precision', cv=skf).mean(), 4),
        "recall": round(cross_val_score(model, X, y, scoring='recall', cv=skf).mean(), 4),
        "f1_score": round(cross_val_score(model, X, y, scoring='f1', cv=skf).mean(), 4)
    }

# ✅ 실행 함수: 두 모델 비교
def compare_models(df):
    X, y = prepare_data(df)
    basic = evaluate_basic_model(X, y)
    cv = evaluate_cv_model(X, y)

    return {
        "기본 모델": basic,
        "교차검증 모델": cv
    }
