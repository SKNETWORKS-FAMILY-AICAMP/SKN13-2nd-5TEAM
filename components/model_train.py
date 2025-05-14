import pandas as pd
import os
import joblib
from xgboost import XGBClassifier
from sklearn.utils.class_weight import compute_sample_weight
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def train_model(method='smote', threshold=0.5, best_params=None):
    print(f"XGBoost 학습 시작 ({method.upper()} 방식)")

    # 데이터 로드
    X_train = pd.read_csv('data/processed/X_train.csv')
    y_train = pd.read_csv('data/processed/y_train.csv').values.ravel()
    X_test = pd.read_csv('data/processed/X_test.csv')
    y_test = pd.read_csv('data/processed/y_test.csv').values.ravel()

    # 모델 정의 (best_params 적용 가능)
    if best_params is None:
        model = XGBClassifier(eval_metric='logloss', random_state=42)
    else:
        model = XGBClassifier(**best_params, eval_metric='logloss', random_state=42)

    if method == 'smote':
        print("🔁 SMOTE 오버샘플링 적용")
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
        model.fit(X_resampled, y_resampled)

    elif method == 'weight':
        print("⚖️ 클래스 가중치 조정 적용")
        sample_weights = compute_sample_weight(class_weight='balanced', y=y_train)
        model.fit(X_train, y_train, sample_weight=sample_weights)

    else:
        print("⚙️ 기본 학습 진행")
        model.fit(X_train, y_train)

    # 모델 저장
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, 'models/churn_model.pkl')
    print("💾 모델 저장 완료 → models/churn_model.pkl")

    # 예측 확률과 threshold 적용
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)

    # 평가 지표 출력
    print(f"✅ Threshold   : {threshold}")
    print(f"✅ Accuracy    : {accuracy_score(y_test, y_pred):.4f}")
    print(f"✅ Precision   : {precision_score(y_test, y_pred):.4f}")
    print(f"✅ Recall      : {recall_score(y_test, y_pred):.4f}")
    print(f"✅ F1 Score    : {f1_score(y_test, y_pred):.4f}")

    return model
