import pandas as pd
from sklearn.model_selection import train_test_split
import os

def split_data():
    df = pd.read_csv('data/processed/processed_data.csv')
    X = df.drop(columns=['Churn'])
    y = df['Churn']

    # 학습 데이터와 테스트 데이터 분할
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # 디렉토리 생성 및 데이터 저장
    os.makedirs('data/processed', exist_ok=True)
    X_train.to_csv('data/processed/X_train.csv', index=False)
    X_test.to_csv('data/processed/X_test.csv', index=False)
    y_train.to_csv('data/processed/y_train.csv', index=False)
    y_test.to_csv('data/processed/y_test.csv', index=False)

    print("✅ 데이터 분할 완료! → data/processed/ 디렉토리에 저장됨")
