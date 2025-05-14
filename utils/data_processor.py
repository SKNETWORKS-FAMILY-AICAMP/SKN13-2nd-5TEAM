import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder


def load_and_process_data():
    file_path = 'data/raw/healtics_data.csv'
    df = pd.read_csv(file_path)

    # 타겟 변수 변환
    df['churn'] = df['Retention'].map({'Yes': 0, 'No': 1})  # Retention = No → churn(1)

    # 제거할 열 목록
    drop_cols = [
        'Date of enrollment', 'Patient ID', 'First Name', 'Last Name', 'Case Assignment Date', 'Week Number', 'COC ID',
        'Patient Address', 'Patient City', 'Patient State', 'Reason for missing', 'Clinic Location'
    ]
    df.drop(columns=[col for col in drop_cols if col in df.columns], inplace=True)

    # 결측치 처리
    df.fillna(df.mean(numeric_only=True), inplace=True)

    # 범주형 인코딩
    cat_cols = df.select_dtypes(include='object').columns
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))

    # 특성과 타겟 분리
    X = df.drop(columns=['churn'])
    y = df['churn']

    # 스케일링
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 데이터 분할
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    # 저장
    os.makedirs('data/processed', exist_ok=True)
    X_train_df = pd.DataFrame(X_train, columns=X.columns)
    X_test_df = pd.DataFrame(X_test, columns=X.columns)
    X_train_df.to_csv('data/processed/X_train.csv', index=False)
    X_test_df.to_csv('data/processed/X_test.csv', index=False)
    y_train.to_frame(name='churn').to_csv('data/processed/y_train.csv', index=False)
    y_test.to_frame(name='churn').to_csv('data/processed/y_test.csv', index=False)

    print("전처리 끝!")
    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    load_and_process_data()