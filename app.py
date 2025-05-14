# ✅ app.py 개선 - 튜닝 모델 시각화 강화
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from utils.data_processor import load_and_process_data
from components.model_selector import get_model
from components.model_evaluator import evaluate_model
from components.model_compare import compare_models

st.set_page_config(layout="wide")
st.title("📊 고객 이탈률 예측 모델")

menu_selection = st.sidebar.selectbox(
    "📌 메뉴를 선택하세요",
    ["모델 비교", "📈 모델 성능 향상 비교"]
)

if not os.path.exists('data/processed/X_train.csv'):
    load_and_process_data()

X_train = pd.read_csv('data/processed/X_train.csv')
X_test = pd.read_csv('data/processed/X_test.csv')
y_train = pd.read_csv('data/processed/y_train.csv')
y_test = pd.read_csv('data/processed/y_test.csv')

model_names = [
    'Logistic Regression', 'Random Forest', 'Gradient Boosting', 'SVM', 'XGBoost'
]

if menu_selection == "📈 모델 성능 향상 비교":
    st.title("📈 기본 모델 vs 튜닝 모델 성능 향상 비교")

    st.sidebar.title("모델 선택")
    tabs = st.sidebar.radio("비교할 모델을 선택하세요:", model_names)

    with st.spinner("모델 훈련 및 튜닝 중..."):
        comparison_df = compare_models(X_train, y_train.values.ravel(), X_test, y_test, model_name=tabs)

    st.subheader("모델 성능 비교표")
    st.dataframe(comparison_df)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("compare chart")
        fig1, ax1 = plt.subplots(figsize=(6, 4))
        comparison_df.set_index("Metric")[['Base Model', 'Tuned Model']].plot(kind='bar', ax=ax1)
        ax1.set_ylabel("Score")
        ax1.set_title(f"{tabs} compare")
        ax1.grid(True, axis='y', linestyle='--', alpha=0.7)
        st.pyplot(fig1)

    with col2:
        st.subheader("Improvement (%)")
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        sns.barplot(x='Metric', y='Improvement (%)', data=comparison_df, ax=ax2, palette='crest')
        ax2.set_title("Improvement (%)")
        ax2.set_ylabel("Improvement (%)")
        ax2.axhline(0, color='gray', linestyle='--')
        st.pyplot(fig2)

elif menu_selection == "모델 비교":
    st.sidebar.title("모델 선택")
    tabs = st.sidebar.radio("모델을 선택하세요:", model_names)

    model = get_model(tabs)
    model.fit(X_train, y_train.values.ravel())
    metrics = evaluate_model(model, X_test, y_test)

    left_col, right_col = st.columns(2)
    with left_col:
        st.subheader(f"{tabs} 모델 성능")
        st.metric("Accuracy", f"{metrics['accuracy']:.4f}")
        st.metric("Precision", f"{metrics['precision']:.4f}")
        st.metric("Recall", f"{metrics['recall']:.4f}")
        st.metric("F1 Score", f"{metrics['f1']:.4f}")

    with right_col:
        st.subheader("Confusion Matrix")
        cm = metrics['confusion_matrix']
        fig, ax = plt.subplots(figsize=(5, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Predicted No", "Predicted Yes"], yticklabels=["Actual No", "Actual Yes"])
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        st.pyplot(fig)

    st.subheader(f"{tabs} Classification Report")
    st.text(pd.DataFrame(metrics['report']).transpose().round(2).to_string())
