import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import GridSearchCV
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from imblearn.pipeline import Pipeline  # 이걸 꼭 써야 함! sklearn 말고!


class SMOTETransformer(BaseEstimator, TransformerMixin):
    def __init__(self, random_state=42):
        self.random_state = random_state  # 반드시 속성으로 등록해야 함

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        if y is None:
            raise ValueError("y 값 없음 어디감")
        smote = SMOTE(random_state=self.random_state)
        X_res, y_res = smote.fit_resample(X, y)
        return X_res, y_res

# 기본 모델 훈련 함수 (XGBoost 모델)
def train_base_model(X_train, y_train):
    model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    model.fit(X_train, y_train)
    return model


# 모델 훈련 함수 (튜닝된 모델)
def train_tuned_model(X_train, y_train):
    pipeline = Pipeline([
        ('smote', SMOTE(random_state=42)),  # SMOTE를 파이프라인에 직접 넣음
        ('model', RandomForestClassifier(random_state=42))
    ])

    param_grid = {
        'model__n_estimators': [50, 100, 200],
        'model__max_depth': [5, 10, 20]
    }

    grid_search = GridSearchCV(pipeline, param_grid, cv=3, scoring='accuracy', verbose=1, n_jobs=-1)
    grid_search.fit(X_train, y_train)

    return grid_search.best_estimator_

# 모델 평가
def evaluate_model(model, X_test, y_test):
    """
    모델 평가 함수. 모델, X_test, y_test를 받아서 성능을 평가!
    """
    # 예측값 생성
    y_pred = model.predict(X_test)
    
    # 성능 지표 계산
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "confusion_matrix": cm,
        "report": report
    }
# 모델 비교 함수
def compare_models(X_train, y_train, X_test, y_test):
    base_model = train_base_model(X_train, y_train)
    tuned_model = train_tuned_model(X_train, y_train)

    base_metrics = evaluate_model(base_model, X_test, y_test)
    tuned_metrics = evaluate_model(tuned_model, X_test, y_test)

    comparison = pd.DataFrame({
        "Metric": ["Accuracy", "Precision", "Recall", "F1 Score"],
        "Base Model": [
            base_metrics["accuracy"],
            base_metrics["precision"],
            base_metrics["recall"],
            base_metrics["f1"]
        ],
        "Tuned Model": [
            tuned_metrics["accuracy"],
            tuned_metrics["precision"],
            tuned_metrics["recall"],
            tuned_metrics["f1"]
        ]
    })
    comparison["Improvement (%)"] = (
        (comparison["Tuned Model"] - comparison["Base Model"]) / comparison["Base Model"] * 100
    ).round(2)

    return comparison
