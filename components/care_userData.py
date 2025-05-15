import streamlit as st
import pandas as pd

def show_user_data(df):
    st.subheader("📋 이용자 데이터")

    # ───────────────────────────────
    # ✅ 연령대 선택 및 이름 검색 입력
    # ───────────────────────────────
    age_options = ["전체", "20대", "30대", "40대", "50대", "60대"]
    selected_age = st.selectbox("연령선택", age_options)
    search_name = st.text_input("검색할 이용자 이름 입력", "")

    if st.button("검색"):
        filtered = df.copy()

        # 연령대 필터
        if selected_age != "전체":
            # 연령대 라벨 추가 없으면 생성
            if "연령대" not in df.columns:
                # 🔽 나이 숫자로 변환 (문자 포함 시 처리)
                df["age"] = pd.to_numeric(df["age"], errors="coerce")  # 잘못된 값은 NaN 처리
                df = df.dropna(subset=["age"])  # NaN 제거
                df["age"] = df["age"].astype(int)

                df["연령대"] = pd.cut(df["age"], 
                                    bins=[0, 29, 39, 49, 59, 150],
                                    labels=["20대", "30대", "40대", "50대", "60대"])
        # 이름 필터 (예시: 'name' 컬럼이 있다고 가정)
        if search_name:
            filtered = filtered[filtered["name"].str.contains(search_name, case=False, na=False)]

        if filtered.empty:
            st.warning("🔍 조건에 해당하는 사용자가 없습니다.")
            return

        # ───────────────────────────────
        # ✅ 결과 테이블 생성
        # ───────────────────────────────
        def get_risk_level(prob):
            if prob >= 0.7:
                return "🔴 고위험", f"{prob*100:.1f}%"
            elif prob >= 0.3:
                return "🟠 중위험", f"{prob*100:.1f}%"
            else:
                return "🟢 저위험", f"{prob*100:.1f}%"

        display_data = []
        for idx, row in filtered.iterrows():
            churn_prob = row.get("churn_prob", 0.0)  # 예측 이탈확률 컬럼이 있다고 가정
            risk_label, prob_percent = get_risk_level(churn_prob)

            display_data.append({
                "ID": row.get("id", f"ID_{idx}"),
                "이용자명": row.get("name", "알 수 없음"),
                "조건1": f"{row.get('steps', 0):.0f}보",
                "조건2": f"{row.get('calories', 0):.0f}kcal",
                "이탈가능성": f"{risk_label} {prob_percent}"
            })

        result_df = pd.DataFrame(display_data)
        st.dataframe(result_df, use_container_width=True)

    else:
        st.info("👆 연령대와 이름을 선택하고 '검색' 버튼을 눌러주세요.")
