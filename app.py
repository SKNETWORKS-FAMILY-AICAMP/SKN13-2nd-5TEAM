# app.py 파일
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from utils.data_processor import load_and_process_data
from components.model_selector import get_model
from components.model_evaluator import evaluate_model  # 수정된 evaluate_model 가져오기
from components.model_compare import compare_models

# 페이지 설정
st.set_page_config(layout="wide")
st.title("📊 고객 이탈 예측 시스템")

# 사이드바 메뉴
menu_selection = st.sidebar.selectbox(
    "📌 메뉴를 선택하세요",
    ["모델 비교", "📈 모델 성능 향상 비교"]
)

# 🔁 X_train.csv 없으면 전처리 먼저 수행 
if not os.path.exists('data/processed/X_train.csv'):
    load_and_process_data()

# 데이터 로드
X_train = pd.read_csv('data/processed/X_train.csv')
X_test = pd.read_csv('data/processed/X_test.csv')
y_train = pd.read_csv('data/processed/y_train.csv')
y_test = pd.read_csv('data/processed/y_test.csv')

# 모델 목록
model_names = [
    'Logistic Regression', 'Random Forest', 'Gradient Boosting', 'SVM', 'XGBoost'
]

# 1. 모델 성능 향상 비교 메뉴
if menu_selection == "📈 모델 성능 향상 비교":
    st.title("📈 기본 모델 vs 튜닝 모델 성능 향상 비교")

    with st.spinner("모델 훈련 중..."):
        comparison_df = compare_models(X_train, y_train.values.ravel(), X_test, y_test)

    st.subheader("모델 비교 결과")
    st.dataframe(comparison_df)

    st.subheader("📊 향상률 시각화")
    fig, ax = plt.subplots(figsize=(8, 5))
    comparison_df.plot(x='Metric', y='Improvement (%)', kind='bar', legend=False, ax=ax, color='green')
    # 한글깨짐 ㄹㅇ 
    ax.set_title("Improvement (%)")
    ax.set_ylabel("Improvement (%)")
    st.pyplot(fig)

# 2. 모델별 성능 평가 메뉴
elif menu_selection == "모델 비교":
    st.sidebar.title("모델 선택")
    tabs = st.sidebar.radio("모델을 선택하세요:", model_names)

    model = get_model(tabs)
    model.fit(X_train, y_train.values.ravel())
    
    # 수정된 evaluate_model 함수 호출
    metrics = evaluate_model(model, X_test, y_test)

    left_col, right_col = st.columns(2)

    with left_col:
        st.subheader(f"{tabs} model")
        st.metric(label="Accuracy", value=f"{metrics['accuracy']:.4f}")
        st.metric(label="Precision", value=f"{metrics['precision']:.4f}")
        st.metric(label="Recall", value=f"{metrics['recall']:.4f}")
        st.metric(label="F1 Score", value=f"{metrics['f1']:.4f}")

    with right_col:
        st.subheader(f"{tabs} Confusion Matrix")
        cm = metrics['confusion_matrix']
        fig, ax = plt.subplots(figsize=(5, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Predicted No", "Predicted Yes"], yticklabels=["Actual No", "Actual Yes"])
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        st.pyplot(fig)

    st.subheader(f"{tabs} Classification Report")
    st.text(pd.DataFrame(metrics['report']).transpose().round(2).to_string())
