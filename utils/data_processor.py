import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import os

def load_and_process_data():
    file_path = 'data/raw/diabetes_dataset.csv'
    df = pd.read_csv(file_path)

    # 결측치 처리
    df.fillna(df.mean(numeric_only=True), inplace=True)

    # 범주형 인코딩
    cat_cols = df.select_dtypes(include='object').columns
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))

    # 특성과 타겟 분리
    X = df.drop(columns=['diabetes'])   # 'diabetes' 컬럼을 타겟 변수로 사용 타겟변수가 다를시 여기를 수정해면 됨!
    y = df['diabetes']                  # 

    # 스케일링
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 분할
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    # 디렉토리 생성
    os.makedirs('data/processed', exist_ok=True)

    # 저장
    X_train_df = pd.DataFrame(X_train, columns=X.columns)  # X_train의 컬럼명 설정
    X_test_df = pd.DataFrame(X_test, columns=X.columns)  # X_test의 컬럼명 설정
    X_train_df.to_csv('data/processed/X_train.csv', index=False)
    X_test_df.to_csv('data/processed/X_test.csv', index=False)
    y_train.to_frame(name='diabetes').to_csv('data/processed/y_train.csv', index=False)
    y_test.to_frame(name='diabetes').to_csv('data/processed/y_test.csv', index=False)

    print("✅ 전처리 및 분할 완료! → data/processed/ 에 저장됨")
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    load_and_process_data()
