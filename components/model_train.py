# components/model_train.py
import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from utils.data_processor import load_fitbit_data

def train_rf_model(X_train, X_test, y_train, y_test):
    model = RandomForestClassifier(class_weight='balanced', random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    report = classification_report(y_test, y_pred, output_dict=True)
    matrix = confusion_matrix(y_test, y_pred)
    return model, report, matrix

def train_xgb_model_with_smote(df):
    # user_id, participant_id, id 순서로 컬럼 탐색
    if "participant_id" in df.columns:
        id_col = "participant_id"
    elif "user_id" in df.columns:
        id_col = "user_id"
    elif "id" in df.columns:
        id_col = "id"
    else:
        raise KeyError("user_id, participant_id 또는 id 컬럼이 없습니다.")

    # 이탈자 정의: 마지막 활동일이 기준일보다 7일 이상 전이면 CHURNED=1
    last_active = df.groupby(id_col)["date"].max().reset_index()
    cutoff = df["date"].max() - pd.Timedelta(days=7)
    last_active["CHURNED"] = last_active["date"] < cutoff
    df = df.merge(last_active[[id_col, "CHURNED"]], on=id_col)

    # 존재하는 컬럼 기준으로 동적 피처 추출
    candidate_features = ["steps", "calories", "sleep_minutes", "heartrate"]
    features = [col for col in candidate_features if col in df.columns]
    if not features:
        raise KeyError("분석에 사용할 수 있는 피처 컬럼이 없습니다: steps, calories, sleep_minutes, heartrate 중 최소 하나가 있어야 합니다.")

    df_model = df[features + ["CHURNED"]].dropna()

    print(f"총 데이터 수: {len(df_model)}")
    print(df_model.head())

    X = df_model[features]
    y = df_model["CHURNED"].astype(int)

    # train/test split + SMOTE
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.3, random_state=42)
    X_train_res, y_train_res = SMOTE(random_state=42).fit_resample(X_train, y_train)

    model = XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42)
    model.fit(X_train_res, y_train_res)
    y_pred = model.predict(X_test)

    report = classification_report(y_test, y_pred, output_dict=True)
    matrix = confusion_matrix(y_test, y_pred)

    return model, report, matrix

def plot_confusion_matrix(matrix, labels=["Not Churned", "Churned"], title="Confusion Matrix"):
    plt.figure(figsize=(5, 4))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels)
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    st = plt.gcf()
    return st
