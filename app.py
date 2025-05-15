import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import plotly.express as px
from sklearn.metrics import confusion_matrix

from components.model_train import (
    prepare_data,
    train_best_xgb_model,
    train_xgb_model_with_smote,
    plot_metrics_bar
)
from utils.data_processor import load_fitbit_data
from components.care_analytic import show_healthcare_result
from components.care_graph import show_healthcare_graph
from components.care_predict import show_prediction_summary
from components.care_predict_graph import show_prediction_graphs
from components.model_compare import compare_models
from components.care_userData import show_user_data
# ───────────────────────────────────────────────
# 사이드바 상태 초기화
# ───────────────────────────────────────────────
if 'show_healthcare_sub' not in st.session_state:
    st.session_state['show_healthcare_sub'] = False
if 'show_prediction_sub' not in st.session_state:
    st.session_state['show_prediction_sub'] = False
if 'main_menu' not in st.session_state:
    st.session_state['main_menu'] = "지표 확인"

# ───────────────────────────────────────────────
# 사이드바 메뉴
# ───────────────────────────────────────────────
with st.sidebar:
    st.title("📌 LNB 메뉴")

    if st.button("지표 확인"):
        st.session_state['main_menu'] = "지표 확인"
        st.session_state['show_healthcare_sub'] = False
        st.session_state['show_prediction_sub'] = False

    if st.button("헬스케어 분석 열기/닫기"):
        st.session_state['show_healthcare_sub'] = not st.session_state['show_healthcare_sub']
        st.session_state['main_menu'] = "헬스케어 분석"

    if st.session_state['show_healthcare_sub']:
        sub_menu = st.radio("헬스케어 분석", ["결과", "그래프"], key="hc_sub")
        st.session_state['main_menu'] = f"헬스케어 분석 - {sub_menu}"

    if st.button("예측 결과 열기/닫기"):
        st.session_state['show_prediction_sub'] = not st.session_state['show_prediction_sub']
        st.session_state['main_menu'] = "예측 결과"

    if st.session_state['show_prediction_sub']:
        sub_menu = st.radio("예측 결과", ["결과", "그래프"], key="pred_sub")
        st.session_state['main_menu'] = f"예측 결과 - {sub_menu}"

    if st.button("이용자 데이터"):
        st.session_state['main_menu'] = "이용자 데이터"
        st.session_state['show_healthcare_sub'] = False
        st.session_state['show_prediction_sub'] = False

# ───────────────────────────────────────────────
# 데이터 로딩
# ───────────────────────────────────────────────
with st.spinner("📂 Fitbit 데이터 로딩 중..."):
    df = load_fitbit_data()
    st.sidebar.success("✅ 데이터 로드 완료!")

# ───────────────────────────────────────────────
# 화면 함수 정의
# ───────────────────────────────────────────────
def show_overview():
    st.header("🔢 연령별 사용자 수치")
    try:
        col1, col2 = st.columns([2, 1])
        with col1:
            age_labels = ['20대', '30대', '40대', '50대', '60대']
            age_values = [10, 15, 30, 25, 20]
            fig = px.pie(
                names=age_labels,
                values=age_values,
                title='연령별 사용자 분포',
                width=600,
                height=600,
                color_discrete_sequence=px.colors.sequential.Blues
            )
            fig.update_traces(textinfo='percent+label')
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            st.write("추가로 필요한 설명 내용")

    except ValueError as e:
        st.error(f"데이터 준비 오류: {e}")

# [다른 함수는 생략 - 기존 상태 유지]

# ───────────────────────────────────────────────
# 본문 렌더링
# ───────────────────────────────────────────────
menu = st.session_state.get("main_menu", "지표 확인")

if menu == "지표 확인":
    show_overview()
elif menu == "헬스케어 분석 - 결과":
    show_healthcare_result(df)
elif menu == "헬스케어 분석 - 그래프":
    show_healthcare_graph(df)
elif menu == "예측 결과 - 결과":
    show_prediction_summary(df)
elif menu == "예측 결과 - 그래프":
    show_prediction_graphs(df)
elif menu == "이용자 데이터":
    show_user_data(df)