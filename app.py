import streamlit as st
from components.model_train import (
    train_xgb_model_with_smote,
    plot_confusion_matrix
)
from utils.data_processor import load_fitbit_data  # 수정된 함수명

st.title("📉 Fitbit 기반 고객 이탈 예측 모델")

# 데이터 로딩
with st.spinner("📂 Fitbit 데이터 로딩 중..."):
    df = load_fitbit_data()
    st.success("✅ 데이터 로딩 완료!")
    st.dataframe(df.head())

# 모델 학습
with st.spinner("⚙️ XGBoost + SMOTE 모델 학습 중..."):
    model, report, matrix = train_xgb_model_with_smote(df)

# 결과 출력
st.subheader("📊 예측 성능 보고서")
st.json(report)

st.subheader("🧩 혼동 행렬")
plot_confusion_matrix(matrix)
