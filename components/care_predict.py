import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier


# ✅ 데이터 전처리 함수 (이탈 조건 정의)
def prepare_data(df):
    base_cols = ["steps", "calories", "very_active_minutes", "moderately_active_minutes", "distance"]
    df = df.dropna(subset=base_cols + ["id"])
    df_user = df.groupby("id")[base_cols].mean().reset_index()

    score = 0
    score += (df_user["steps"] < 6400).astype(int)
    score += (df_user["calories"] < 1800).astype(int)
    score += (df_user["very_active_minutes"] < 6).astype(int) * 2
    score += (df_user["moderately_active_minutes"] < 8).astype(int)
    score += (df_user["distance"] < 4600).astype(int)
    df_user["CHURNED"] = (score >= 4).astype(int)

    return df_user, df_user[base_cols], df_user["CHURNED"]


# ✅ 예측 및 시각화 메인 함수
def show_prediction_summary(df):
    st.header("📊 이용자 이탈 예측 결과")

    df_user, X, y = prepare_data(df)

    # 모델 학습
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)
    model = XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42)
    model.fit(X_train, y_train)

    # 이탈 확률 예측
    churn_probs = model.predict_proba(X)[:, 1]
    df_user["churn_prob"] = churn_probs

    # 위험군 분류
    df_user["risk"] = pd.cut(df_user["churn_prob"],
                             bins=[-0.01, 0.3, 0.7, 1.01],
                             labels=["저위험", "중위험", "고위험"])

    # 시각화용 색상 및 라벨
    color_map = {"고위험": "red", "중위험": "orange", "저위험": "green"}
    label_map = {
        "고위험": "🔴 고위험 이용자\n이탈율 70% 이상",
        "중위험": "🟠 중위험 이용자\n이탈율 30~70%",
        "저위험": "🟢 저위험 이용자\n이탈율 30% 미만"
    }

    # 🧾 위험군 개수 요약 박스
    st.subheader("📌 위험군 분포")
    cols = st.columns(3)
    for i, level in enumerate(["고위험", "중위험", "저위험"]):
        group = df_user[df_user["risk"] == level]
        count = len(group)
        percent = round((count / len(df_user)) * 100, 2)
        with cols[i]:
            st.markdown(f"""
                <div style='border:2px solid {color_map[level]}; padding: 15px; border-radius: 8px; text-align:center'>
                    <strong style='font-size:18px'>{label_map[level]}</strong><br><br>
                    <span style='font-size:16px'>{count:,}명 ({percent}%)</span>
                </div>
            """, unsafe_allow_html=True)

    # ℹ️ 설명 영역
    st.markdown("### ℹ️ 이탈 확률 계산에 사용된 기준")
    st.info("- 활동량, 칼로리, 거리, 활동 시간 등을 기반으로 XGBoost 모델을 학습\n"
            "- 예측된 이탈 확률을 바탕으로 3단계 위험군으로 분류")

    # 📋 평균 테이블
    st.markdown("### 📋 위험군별 평균 활동량")
    mean_table = df_user.groupby("risk")[["steps", "calories", "very_active_minutes",
                                          "moderately_active_minutes", "distance"]].mean().round(2)
    st.dataframe(mean_table)

    # 📈 확률 분포
    st.markdown("### 📈 전체 이탈 확률 분포")
    fig, ax = plt.subplots()
    sns.histplot(df_user["churn_prob"], bins=20, kde=True, color="skyblue")
    ax.set_title("예측된 이탈 확률 분포")
    st.pyplot(fig)
