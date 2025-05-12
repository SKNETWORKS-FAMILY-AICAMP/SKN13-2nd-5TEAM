import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.utils.class_weight import compute_sample_weight
from imblearn.over_sampling import SMOTE

def train_model(method='smote'):
    print(f"🚀 XGBoost 학습 시작 ({method.upper()} 방식)")

    # 데이터 로드
    X_train = pd.read_csv('data/processed/X_train.csv')
    y_train = pd.read_csv('data/processed/y_train.csv').values.ravel()

    if method == 'smote':
        print("🔁 SMOTE 오버샘플링 적용")
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
        model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
        model.fit(X_resampled, y_resampled)

    elif method == 'weight':
        print("⚖️ 클래스 가중치 조정 적용")
        sample_weights = compute_sample_weight(class_weight='balanced', y=y_train)
        model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
        model.fit(X_train, y_train, sample_weight=sample_weights)

    else:
        raise ValueError("지원하지 않는 방법입니다. 'smote' 또는 'weight' 중 하나를 선택하세요.")

    print("✅ 모델 학습 완료")
    return model
