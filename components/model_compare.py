import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
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


def compare_models(X_train, y_train, X_test, y_test, model_name='XGBoost'):
    if model_name == 'XGBoost':
        base_model = xgb.XGBClassifier(eval_metric='logloss', random_state=42)
        base_model.fit(X_train, y_train)

        pipeline = Pipeline([
            ('smote', SMOTE(random_state=42)),
            ('model', xgb.XGBClassifier(eval_metric='logloss', random_state=42))
        ])

        param_dist = {
            'model__n_estimators': [100, 200, 300],
            'model__max_depth': [3, 5, 7],
            'model__learning_rate': [0.01, 0.05, 0.1],
            'model__subsample': [0.8, 1.0],
            'model__colsample_bytree': [0.8, 1.0],
            'model__gamma': [0, 1],
            'model__min_child_weight': [1, 3]
        }

    elif model_name == 'Random Forest':
        base_model = RandomForestClassifier(random_state=42)
        base_model.fit(X_train, y_train)

        pipeline = Pipeline([
            ('smote', SMOTE(random_state=42)),
            ('model', RandomForestClassifier(random_state=42))
        ])

        param_dist = {
            'model__n_estimators': [100, 200],
            'model__max_depth': [5, 10, 15]
        }

    elif model_name == 'SVM':
        base_model = SVC()
        base_model.fit(X_train, y_train)

        pipeline = Pipeline([
            ('smote', SMOTE(random_state=42)),
            ('model', SVC())
        ])

        param_dist = {
            'model__C': [0.1, 1, 10],
            'model__kernel': ['linear', 'rbf']
        }

    elif model_name == 'Logistic Regression':
        base_model = LogisticRegression(max_iter=1000)
        base_model.fit(X_train, y_train)

        pipeline = Pipeline([
            ('smote', SMOTE(random_state=42)),
            ('model', LogisticRegression(max_iter=1000))
        ])

        param_dist = {
            'model__C': [0.01, 0.1, 1, 10]
        }

    elif model_name == 'Gradient Boosting':
        base_model = GradientBoostingClassifier()
        base_model.fit(X_train, y_train)

        pipeline = Pipeline([
            ('smote', SMOTE(random_state=42)),
            ('model', GradientBoostingClassifier())
        ])

        param_dist = {
            'model__n_estimators': [100, 200],
            'model__learning_rate': [0.01, 0.1],
            'model__max_depth': [3, 5]
        }

    else:
        raise ValueError("지원하지 않는 모델입니다.")

    # RandomizedSearchCV 적용
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_dist,
        n_iter=30,  # 조합 수 조절
        scoring='f1',
        n_jobs=-1,
        cv=cv,
        verbose=1,
        random_state=42
    )
    search.fit(X_train, y_train)
    tuned_model = search.best_estimator_

    # 평가
    base_metrics = evaluate_model(base_model, X_test, y_test)
    tuned_metrics = evaluate_model(tuned_model, X_test, y_test)

    # 비교
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
    comparison["Tuned Model Type"] = model_name

    return comparison
