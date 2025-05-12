import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
import os

def load_and_process_data():
    file_path = 'data/raw/E Commerce Dataset.xlsx'
    df = pd.read_excel(file_path, sheet_name='E Comm')

    # CustomerID 제거 (모델 학습에 필요 없음)
    df = df.drop(columns=['CustomerID'])

    # 결측치 처리 (수치형은 평균으로)
    df.fillna(df.mean(numeric_only=True), inplace=True)

    # 범주형 변수 인코딩
    cat_cols = df.select_dtypes(include='object').columns
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])

    # 특성과 타겟 분리
    X = df.drop(columns=['Churn'])
    y = df['Churn']

    # 스케일링
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 최종 데이터프레임 생성
    processed_df = pd.DataFrame(X_scaled, columns=X.columns)
    processed_df['Churn'] = y.values
    output_path = 'data/processed/processed_data.csv'
    # 디렉토리 생성 및 저장
    os.makedirs('data/processed', exist_ok=True)
    processed_df.to_csv('data/processed/processed_data.csv', index=False)
    print(f"파일 저장 중: {output_path}")
    print("✅ 전처리 완료! → data/processed/processed_data.csv 저장됨")
    return processed_df
if __name__ == "__main__":
    load_and_process_data()