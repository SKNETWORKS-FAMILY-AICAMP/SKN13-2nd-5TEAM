import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

from components.model_train import (
    prepare_data,
    train_best_xgb_model,
    train_xgb_model_with_smote,
    plot_metrics_bar
)
from utils.data_processor import load_fitbit_data

# ───────────────────────────────────────────────
# 🔧 사이드바 상태 초기화
# ───────────────────────────────────────────────
if 'show_healthcare_sub' not in st.session_state:
    st.session_state['show_healthcare_sub'] = False
if 'show_prediction_sub' not in st.session_state:
    st.session_state['show_prediction_sub'] = False
if 'main_menu' not in st.session_state:
    st.session_state['main_menu'] = "지표 확인"

# ───────────────────────────────────────────────
# 📌 사이드바 메뉴 구성 (LNB)
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
# 📂 데이터 로딩
# ───────────────────────────────────────────────
with st.spinner("📂 Fitbit 데이터 로딩 중..."):
    df = load_fitbit_data()
    st.sidebar.success("✅ 데이터 로드 완료!")

# ───────────────────────────────────────────────
# 📈 렌더링 함수 정의
# ───────────────────────────────────────────────
def show_overview():
    st.header("🔢 지표 확인")
    try:
        _, _, _, y = prepare_data(df)
        st.markdown(f"- 전체 샘플 수: **{len(df):,}건**")
        st.markdown(f"- SMOTE 전 이탈자 비율: **{y.sum():,} / {len(y):,} ({100*y.sum()/len(y):.1f}%)**")
        st.markdown("- 주요 모델: XGBoost (정밀도 튜닝 + F1 최적화 threshold)")
    except ValueError as e:
        st.error(f"데이터 준비 오류: {e}")

def show_health_results():
    st.header("🏥 헬스케어 분석 – 결과")
    try:
        X_train, _, y_train, _ = prepare_data(df)
        model = train_best_xgb_model(X_train, y_train)

        importances = model.feature_importances_
        feature_names = model.get_booster().feature_names
        top_feats = [f for f, _ in sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)[:8] if f in df.columns]

        st.subheader("📌 중요 피처 간 상관관계")
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(df[top_feats].corr(), annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)

        st.subheader("📦 이탈 여부에 따른 박스플롯")
        if "CHURNED" not in df.columns:
            st.warning("CHURNED 컬럼이 누락되었습니다.")
            return
        df_melt = df.melt(id_vars="CHURNED", value_vars=top_feats)
        fig2, ax2 = plt.subplots(figsize=(12, 6))
        sns.boxplot(data=df_melt, x="CHURNED", y="value", hue="variable", ax=ax2)
        st.pyplot(fig2)
    except ValueError as e:
        st.error(f"분석 불가: {e}")

def show_health_graphs():
    st.header("📊 헬스케어 분석 – 그래프")
    try:
        model, report, threshold = train_xgb_model_with_smote(df)
        scores = {'XGBoost': report["f1_score"]}

        fig, ax = plt.subplots(figsize=(8, 5))
        sns.barplot(x=list(scores.values()), y=list(scores.keys()), ax=ax)
        ax.set_xlim(0, 1)
        st.pyplot(fig)
    except ValueError as e:
        st.error(f"그래프 생성 오류: {e}")

def show_prediction():
    st.header("🤖 예측 결과 – 결과")
    try:
        model, report, threshold = train_xgb_model_with_smote(df)
        st.subheader("📌 평가 지표 요약")
        st.json(report)
        st.markdown(f"**최적 threshold 값:** `{threshold:.3f}`")
    except ValueError as e:
        st.error(f"예측 실패: {e}")

def show_prediction_graphs():
    st.header("📈 예측 결과 – 그래프")
    try:
        model, report, threshold = train_xgb_model_with_smote(df)

        st.subheader("지표 시각화")
        plot_metrics_bar(report)

        st.subheader("혼동 행렬")
        _, X_test, _, y_test = prepare_data(df)
        y_pred = (model.predict_proba(X_test)[:, 1] >= threshold).astype(int)
        matrix = confusion_matrix(y_test, y_pred)

        fig, ax = plt.subplots()
        sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", 
                    xticklabels=["Not Churned", "Churned"], 
                    yticklabels=["Not Churned", "Churned"], ax=ax)
        st.pyplot(fig)
    except ValueError as e:
        st.error(f"그래프 생성 오류: {e}")

def show_user_data():
    st.header("👥 이용자 데이터")
    st.dataframe(df)

# ───────────────────────────────────────────────
# 📄 본문 렌더링
# ───────────────────────────────────────────────
menu = st.session_state.get("main_menu", "지표 확인")

if menu == "지표 확인":
    show_overview()
elif menu == "헬스케어 분석 - 결과":
    show_health_results()
elif menu == "헬스케어 분석 - 그래프":
    show_health_graphs()
elif menu == "예측 결과 - 결과":
    show_prediction()
elif menu == "예측 결과 - 그래프":
    show_prediction_graphs()
elif menu == "이용자 데이터":
    show_user_data()
