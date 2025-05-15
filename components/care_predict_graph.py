import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import re
import matplotlib.font_manager as fm
import platform
# 한글 폰트 설정
if platform.system() == "Darwin":  # macOS
    plt.rcParams['font.family'] = 'AppleGothic'
elif platform.system() == "Windows":
    plt.rcParams['font.family'] = 'Malgun Gothic'
else:  # Linux (예: Colab, Ubuntu 등)
    plt.rcParams['font.family'] = 'NanumGothic'

plt.rcParams['axes.unicode_minus'] = False  # 마이너스 깨짐 방지
# ✅ 문자열 나이를 평균 나이(int)로 변환하는 함수
def convert_age_to_int(age_str):
    try:
        if pd.isnull(age_str):
            return None
        if isinstance(age_str, (int, float)):
            return int(age_str)

        age_str = str(age_str).strip()

        if re.match(r"^\d{2}$", age_str):  # e.g., '29'
            return int(age_str)
        if re.match(r"^\d{2}s?$", age_str):  # e.g., '30s', '30대'
            return int(age_str[:2]) + 5
        if re.match(r"^<\d+$", age_str):  # e.g., '<30'
            return int(age_str[1:]) - 1
        if re.match(r"^>=\d+$", age_str):  # e.g., '>=30'
            return int(age_str[2:]) + 2
        if re.match(r"^<=\d+$", age_str):  # e.g., '<=60'
            return int(age_str[2:]) - 2
        if re.match(r"^\d{2}[-~]\d{2}$", age_str):  # e.g., '30-39', '40~49'
            parts = re.split("[-~]", age_str)
            return (int(parts[0]) + int(parts[1])) // 2

    except:
        return None

    return None

# ✅ 연령대를 범주형으로 분류
def assign_age_group(age):
    if pd.isnull(age): return None
    age = int(age)
    if age < 30: return '20대'
    elif age < 40: return '30대'
    elif age < 50: return '40대'
    elif age < 60: return '50대'
    else: return '60대 이상'

# ✅ 이탈률 그래프 출력 함수
def show_prediction_graphs(df):
    try:
        st.subheader("📊 연령별 이탈률")

        # 🔹 나이 전처리
        df = df.copy()
        df["age_int"] = df["age"].apply(convert_age_to_int)
        df["age_group"] = df["age_int"].apply(assign_age_group)

        # 🔹 성별 통일
        df["gender"] = df["gender"].str.upper().map({
            "MALE": "남성", "FEMALE": "여성", "M": "남성", "F": "여성"
        })

        # 🔹 이탈자 예측 기준: 조건 점수 >= 4
        base_cols = ["steps", "calories", "very_active_minutes", "moderately_active_minutes", "distance"]
        df = df.dropna(subset=base_cols + ["gender", "age_int"])

        score = 0
        score += (df["steps"] < 6400).astype(int)
        score += (df["calories"] < 1800).astype(int)
        score += (df["very_active_minutes"] < 6).astype(int) * 2
        score += (df["moderately_active_minutes"] < 8).astype(int)
        score += (df["distance"] < 4600).astype(int)
        df["churned"] = (score >= 4).astype(int)

        # 🔹 연령대별 이탈률 계산
        summary = df.groupby(["age_group", "gender"])["churned"].mean().reset_index()
        summary["이탈률(%)"] = (summary["churned"] * 100).round(2)

        # 🔹 그래프 출력
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.lineplot(data=summary, x="age_group", y="이탈률(%)", hue="gender", marker="o", ax=ax)
        ax.set_title("연령대별 이탈률")
        ax.set_ylabel("이탈률 (%)")
        ax.set_xlabel("연령대")
        ax.set_ylim(0, 100)
        ax.grid(True)
        st.pyplot(fig)

    except Exception as e:
        st.error(f"그래프 생성 중 오류 발생: {e}")
