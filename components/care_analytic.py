import streamlit as st
import pandas as pd

# 🔧 문자열 age -> 숫자로 변환
def convert_age_to_numeric(age_str):
    try:
        if isinstance(age_str, str):
            if '<' in age_str:
                return int(age_str.replace('<', '')) - 1
            elif '>' in age_str:
                return int(age_str.replace('>', '')) + 1
            elif '-' in age_str:
                a, b = age_str.split('-')
                return (int(a) + int(b)) // 2
            else:
                return int(age_str)
        elif isinstance(age_str, (int, float)):
            return age_str
    except:
        return None


def show_healthcare_result(df: pd.DataFrame):
    # ✅ 나이 컬럼 생성
    if "나이" not in df.columns:
        if "age" in df.columns:
            df["나이"] = df["age"].apply(convert_age_to_numeric)
        else:
            st.error("❌ '나이' 또는 'age' 컬럼이 없습니다.")
            return

    df["나이"] = pd.to_numeric(df["나이"], errors="coerce")

    # ✅ 연령대 파생
    bins = [0, 29, 39, 49, 59, 120]
    labels = ["20대", "30대", "40대", "50대", "60대"]
    try:
        df["연령대"] = pd.cut(df["나이"], bins=bins, labels=labels, right=True)
    except Exception as e:
        st.error(f"❌ 연령대 생성 오류: {e}")
        return

    # ✅ 마름모 Healtics 로고 + 텍스트
    col_logo, col_text = st.columns([1, 3])
    with col_logo:
        st.markdown("""
        <div style='
            width: 120px; height: 120px;
            background: #d6c8ff;
            transform: rotate(45deg);
            display: flex;
            align-items: center;
            justify-content: center;
            border: 4px solid #805ad5;
            margin-bottom: 20px;
        '>
            <div style='transform: rotate(-45deg); font-weight: bold; font-size: 18px;'>
                Healtics
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_text:
        st.markdown(f"<h5><b>{len(df):,}명의 데이터를 학습하였습니다.</b></h5>", unsafe_allow_html=True)
        st.markdown("정보를 보고 싶은 나이대를 선택하세요.")

    st.markdown("---")

    # ✅ 조건 선택 & 검색 버튼
    col1, col2 = st.columns([5, 1])
    age_filter = col1.selectbox("조건 1", ["전체"] + labels, label_visibility="collapsed")
    search_clicked = col2.button("검색")

    # ✅ 검색 결과 처리
    if search_clicked:
        if age_filter == "전체":
            filtered = df
        else:
            filtered = df[df["연령대"] == age_filter]

        if filtered.empty:
            example_data = pd.DataFrame({
                "성별": ["남", "여", "남", "여", "남"],
                "시작 체중": [84, 60, 72, 55, 90],
                "종료 체중": [76, 50, 68, 52, 82],
                "BMI 지수": [27.42, 23.43, 25.1, 21.3, 28.9],
                "BMI 상태": ["과체중", "정상", "정상", "정상", "과체중"],
                "변화량": [7.89, 3.9, 4.0, 3.0, 8.0]
            })
            st.dataframe(example_data, use_container_width=True)
        else:
            columns_to_show = [col for col in ["성별", "나이", "시작 체중", "종료 체중", "BMI 지수", "BMI 상태", "변화량"] if col in filtered.columns]
            st.dataframe(filtered[columns_to_show], use_container_width=True)
