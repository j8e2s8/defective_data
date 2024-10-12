import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
import seaborn as sns
from xgboost import XGBClassifier, plot_importance
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, classification_report, confusion_matrix, make_scorer
from matplotlib import rc 
import lightgbm as lgb
from sklearn.preprocessing import PolynomialFeatures

# XGBoost

# 필요한 데이터 불러오기
df = pd.read_csv('1주_실습데이터.csv')
df['Y'] = df['Y'].astype('boolean')

# X4, X13, X18, X19, X20을 제거
df2 = df.drop(columns=(['X4', 'X13', 'X18', 'X19', 'X20']))

# X와 y 정의 (X는 독립 변수, y는 종속 변수)
X = df2.drop('Y', axis=1)
y = df2['Y']

# 데이터셋을 학습용과 테스트용으로 분리 (90% 학습, 10% 테스트)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42, stratify=y)

# XGBoost 모델의 하이퍼 파라미터 후보 설정
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 6, 9],
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'gamma': [0, 0.1, 0.5]
}

# 여러 지표를 사용하여 평가하도록 scoring 설정
scoring = {
    'accuracy': 'accuracy',
    'precision': make_scorer(precision_score),
    'recall': make_scorer(recall_score),
    'f1': 'f1',
    'roc_auc': 'roc_auc'
}

# GridSearchCV를 사용하여 최적의 하이퍼 파라미터 찾기, f1을 기준으로 모델을 최종 선택
xgb_model = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, scoring=scoring, refit='f1', cv=3, verbose=1, n_jobs=-1)
grid_search.fit(X_train, y_train)

# 최적의 하이퍼 파라미터 출력
print("Best Hyperparameters:", grid_search.best_params_)

#'colsample_bytree': 0.6, 'gamma': 0, 'learning_rate': 0.1, 'max_depth': 9, 'n_estimators': 300, 'subsample': 0.6

# 최적의 모델로 예측
best_xgb_model = grid_search.best_estimator_
y_pred = best_xgb_model.predict(X_test)

# 성능 평가
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, best_xgb_model.predict_proba(X_test)[:, 1])

print(f"XGBoost Accuracy: {accuracy:.4f}")
print(f"XGBoost Precision: {precision:.4f}")
print(f"XGBoost Recall: {recall:.4f}")
print(f"XGBoost F1 Score: {f1:.4f}")
print(f"XGBoost ROC AUC: {roc_auc:.4f}")

# 상세 성능 평가 보고서 출력
print("\nClassification Report:\n", classification_report(y_test, y_pred))



# Light GBM


# LightGBM 모델의 하이퍼 파라미터 후보 설정
# 겹치는 파라미터는 xgboost 베스트 파라미터로 했음. 
param_grid_lgb = {
    'n_estimators': [300],
    'max_depth': [9],
    'learning_rate': [0.1],
    'subsample': [0.6],
    'colsample_bytree': [0.6],
    'reg_alpha': [0, 0.1, 0.5],  # L1 regularization
    'reg_lambda': [0, 0.1, 0.5]  # L2 regularization
}

# 여러 지표를 사용하여 평가하도록 scoring 설정
scoring = {
    'accuracy': 'accuracy',
    'precision': make_scorer(precision_score),
    'recall': make_scorer(recall_score),
    'f1': 'f1',
    'roc_auc': 'roc_auc'
}

# GridSearchCV를 사용하여 최적의 하이퍼 파라미터 찾기, f1을 기준으로 모델을 최종 선택
lgb_model = lgb.LGBMClassifier(random_state=42)
grid_search_lgb = GridSearchCV(estimator=lgb_model, param_grid=param_grid_lgb, scoring=scoring, refit='f1', cv=3, verbose=1, n_jobs=-1)
grid_search_lgb.fit(X_train, y_train)

# 최적의 하이퍼 파라미터 출력
print("Best Hyperparameters for LightGBM:", grid_search_lgb.best_params_)
#Best Hyperparameters for LightGBM: {'colsample_bytree': 0.6, 'learning_rate': 0.1, 'max_depth': 9, 'n_estimators': 300, 'reg_alpha': 0, 'reg_lambda': 0.5, 'subsample': 0.6}

# 최적의 모델로 예측
best_lgb_model = grid_search_lgb.best_estimator_
y_pred_lgb = best_lgb_model.predict(X_test)

# 성능 평가
accuracy_lgb = accuracy_score(y_test, y_pred_lgb)
precision_lgb = precision_score(y_test, y_pred_lgb)
recall_lgb = recall_score(y_test, y_pred_lgb)
f1_lgb = f1_score(y_test, y_pred_lgb)
roc_auc_lgb = roc_auc_score(y_test, best_lgb_model.predict_proba(X_test)[:, 1])

print(f"LightGBM Accuracy: {accuracy_lgb:.4f}")
print(f"LightGBM Precision: {precision_lgb:.4f}")
print(f"LightGBM Recall: {recall_lgb:.4f}")
print(f"LightGBM F1 Score: {f1_lgb:.4f}")
print(f"LightGBM ROC AUC: {roc_auc_lgb:.4f}")

df.groupby('Y').agg(count_y = ('Y','count'))

# XGB boost 혼동 행렬 시각화

# 한글 폰트 설정
rc('font', family='Malgun Gothic')
plt.rcParams['axes.unicode_minus'] = False  # 마이너스 기호가 깨지는 문제 방지

# 혼동 행렬 생성
conf_matrix = confusion_matrix(y_test, y_pred_lgb)

# 시각화
plt.figure(figsize=(6, 5))
ax = sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
                 xticklabels=['양품', '불량'], yticklabels=['양품', '불량'], 
                 cbar_kws={'label': '샘플 수'})  # 색상 바에 레이블 추가

# 1종 오류와 2종 오류 표시
ax.text(1.5, 0.6, '1종 오류', ha='center', va='center', fontsize=12, color='blue', weight='bold')  # 오른쪽 위
ax.text(0.5, 1.6, '2종 오류', ha='center', va='center', fontsize=12, color='red', weight='bold')  # 왼쪽 아래

# 축과 타이틀 설정
plt.xlabel('예측값', fontsize=12)
plt.ylabel('실제값', fontsize=12)
plt.title('Confusion Matrix', fontsize=15)
plt.xticks(rotation=0, fontsize=10)
plt.yticks(rotation=0, fontsize=10)
plt.tight_layout()  # 레이아웃 조정
plt.show()

# Light GBM 혼동행렬 

# 혼동 행렬 생성
conf_matrix = confusion_matrix(y_test, y_pred)

# 시각화
plt.figure(figsize=(6, 5))
ax = sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
                 xticklabels=['양품', '불량'], yticklabels=['양품', '불량'], 
                 cbar_kws={'label': '샘플 수'})  # 색상 바에 레이블 추가

# 1종 오류와 2종 오류 표시
ax.text(1.5, 0.6, '1종 오류', ha='center', va='center', fontsize=12, color='blue', weight='bold')  # 오른쪽 위
ax.text(0.5, 1.6, '2종 오류', ha='center', va='center', fontsize=12, color='red', weight='bold')  # 왼쪽 아래

# 축과 타이틀 설정
plt.xlabel('예측값', fontsize=12)
plt.ylabel('실제값', fontsize=12)
plt.title('Confusion Matrix', fontsize=15)
plt.xticks(rotation=0, fontsize=10)
plt.yticks(rotation=0, fontsize=10)
plt.tight_layout()  # 레이아웃 조정
plt.show()



# lgb로 선택

# 6번 

# 1차항

# 변수 중요도 추출
importances = best_lgb_model.feature_importances_
indices = np.argsort(importances)[::-1]

# 상위 중요도를 100을 기준으로 스케일링
scaled_importances = importances / importances[indices[0]] * 100

# 변수 중요도 시각화 (상위 20개의 중요한 변수만 표시)
plt.figure(figsize=(12, 6))
sns.barplot(x=X_train.columns[indices][:20], y=scaled_importances[indices][:20])
plt.title('Top 20 Feature Importances from LightGBM')
plt.xlabel('Features')
plt.ylabel('Importance (Scaled to 100)')
plt.xticks(rotation=90)

# 상위 5개의 변수 중요도 값을 그래프에 표시
for i in range(5):
    plt.text(x=i, y=scaled_importances[indices[i]], s=f'{scaled_importances[indices[i]]:.2f}', ha='center', va='bottom', color='red', fontsize=12)

plt.show()


# 2차항

# Polynomial Features 적용 (2차 항까지)
poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=False)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

# 새로운 폴리노미얼 특징 이름 생성
poly_feature_names = poly.get_feature_names_out(X_train.columns)

# DataFrame으로 변환 (변수명을 포함하여 보기 좋게 정리)
X_train_poly_df = pd.DataFrame(X_train_poly, columns=poly_feature_names)
X_test_poly_df = pd.DataFrame(X_test_poly, columns=poly_feature_names)

# LightGBM 모델을 사용하여 학습
lgb_model_poly = lgb.LGBMClassifier(random_state=42, **grid_search.best_params_)
lgb_model_poly.fit(X_train_poly_df, y_train)

# 변수 중요도 추출
importances_poly = lgb_model_poly.feature_importances_
indices_poly = np.argsort(importances_poly)[::-1]

# 상위 중요도를 100을 기준으로 스케일링
scaled_importances_poly = importances_poly / importances_poly[indices_poly[0]] * 100

# 변수 중요도 시각화 (상위 20개의 중요한 변수만 표시)
plt.figure(figsize=(12, 6))
sns.barplot(x=X_train_poly_df.columns[indices_poly][:20], y=scaled_importances_poly[indices_poly][:20])
plt.title('Top 20 Feature Importances from LightGBM with Polynomial Features (2nd Degree)')
plt.xlabel('Features')
plt.ylabel('Importance (Scaled to 100)')
plt.xticks(rotation=90)

# 상위 5개의 변수 중요도 값을 그래프에 표시
for i in range(5):
    plt.text(x=i, y=scaled_importances_poly[indices_poly[i]], s=f'{scaled_importances_poly[indices_poly[i]]:.2f}', ha='center', va='bottom', color='red', fontsize=12)

plt.show()


# 3차항

# Polynomial Features 적용 (3차 항까지)
poly = PolynomialFeatures(degree=3, include_bias=False, interaction_only=False)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

# 새로운 폴리노미얼 특징 이름 생성
poly_feature_names = poly.get_feature_names_out(X_train.columns)

# DataFrame으로 변환 (변수명을 포함하여 보기 좋게 정리)
X_train_poly_df = pd.DataFrame(X_train_poly, columns=poly_feature_names)
X_test_poly_df = pd.DataFrame(X_test_poly, columns=poly_feature_names)

# LightGBM 모델을 사용하여 학습
lgb_model_poly = lgb.LGBMClassifier(random_state=42, **grid_search.best_params_)
lgb_model_poly.fit(X_train_poly_df, y_train)

# 변수 중요도 추출
importances_poly = lgb_model_poly.feature_importances_
indices_poly = np.argsort(importances_poly)[::-1]

# 상위 중요도를 100을 기준으로 스케일링
scaled_importances_poly = importances_poly / importances_poly[indices_poly[0]] * 100

# 변수 중요도 시각화 (상위 20개의 중요한 변수만 표시)
plt.figure(figsize=(12, 6))
sns.barplot(x=X_train_poly_df.columns[indices_poly][:20], y=scaled_importances_poly[indices_poly][:20])
plt.title('Top 20 Feature Importances from LightGBM with Polynomial Features (3rd Degree)')
plt.xlabel('Features')
plt.ylabel('Importance (Scaled to 100)')
plt.xticks(rotation=90)

# 상위 5개의 변수 중요도 값을 그래프에 표시
for i in range(5):
    plt.text(x=i, y=scaled_importances_poly[indices_poly[i]], s=f'{scaled_importances_poly[indices_poly[i]]:.2f}', ha='center', va='bottom', color='red', fontsize=12)

plt.show()

################################

# 성능비교 (only poly)

# 성능 측정을 위한 함수 정의
def evaluate_model(X_train, X_test, y_train, y_test, degree, best_params):
    # Polynomial 변환 적용
    poly = PolynomialFeatures(degree=degree, include_bias=False, interaction_only=False)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)
    
    # LightGBM 모델 학습
    lgb_model_poly = lgb.LGBMClassifier(random_state=42, **best_params)
    lgb_model_poly.fit(X_train_poly, y_train)
    
    # 예측 및 성능 평가
    y_pred = lgb_model_poly.predict(X_test_poly)
    y_pred_proba = lgb_model_poly.predict_proba(X_test_poly)[:, 1]
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    return accuracy, precision, recall, f1, roc_auc, y_pred


# 베스트 파라미터 가져오기 (이전에 찾은 최적의 파라미터를 사용함)
best_params = grid_search_lgb.best_params_
#Best Hyperparameters for LightGBM: {'colsample_bytree': 0.6, 'learning_rate': 0.1, 'max_depth': 9, 'n_estimators': 300, 'reg_alpha': 0, 'reg_lambda': 0.5, 'subsample': 0.6}
best_params = {'colsample_bytree': 0.6, 'learning_rate': 0.1, 'max_depth': 9, 'n_estimators': 300, 'reg_alpha': 0, 'reg_lambda': 0.5, 'subsample': 0.6}

# 각 차수에 대해 모델 평가
results_poly = {}
for degree in [1, 2, 3]:
    accuracy, precision, recall, f1, roc_auc, y_pred = evaluate_model(X_train, X_test, y_train, y_test, degree, best_params)
    results_poly[degree] = {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1,
        'ROC AUC': roc_auc,
        'Y_Pred' : y_pred
    }


# 결과 출력
results_df = pd.DataFrame(results_poly).T
results_df.index.name = 'Polynomial Degree'
results_df.columns = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC', 'Y_Pred']

print(results_df)


# 혼동 행렬 생성
conf_matrix = confusion_matrix(y_test, results_poly[1]['Y_Pred'])

# 시각화
plt.figure(figsize=(6, 5))
ax = sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
                 xticklabels=['양품', '불량'], yticklabels=['양품', '불량'], 
                 cbar_kws={'label': '샘플 수'})  # 색상 바에 레이블 추가

# 1종 오류와 2종 오류 표시
ax.text(1.5, 0.6, '1종 오류', ha='center', va='center', fontsize=12, color='blue', weight='bold')  # 오른쪽 위
ax.text(0.5, 1.6, '2종 오류', ha='center', va='center', fontsize=12, color='red', weight='bold')  # 왼쪽 아래

# 축과 타이틀 설정
plt.xlabel('예측값', fontsize=12)
plt.ylabel('실제값', fontsize=12)
plt.title('Confusion Matrix', fontsize=15)
plt.xticks(rotation=0, fontsize=10)
plt.yticks(rotation=0, fontsize=10)
plt.tight_layout()  # 레이아웃 조정
plt.show()



#1      0.999715        1.0  0.997368  0.998682  0.999957
#2      0.999677        1.0  0.997018  0.998507  0.999807
#3      0.999715        1.0  0.997368  0.998682  0.999782



####################################

# smote 이후 폴리 적용

from imblearn.over_sampling import SMOTE

# SMOTE 적용
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)


# 1차
accuracy, precision, recall, f1, roc_auc, y_pred = evaluate_model(X_train_resampled, X_test, y_train_resampled, y_test, 1, best_params)
accuracy, precision, recall, f1, roc_auc, y_pred


# 혼동 행렬 생성
conf_matrix = confusion_matrix(y_test, y_pred)

# 시각화
plt.figure(figsize=(6, 5))
ax = sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
                 xticklabels=['양품', '불량'], yticklabels=['양품', '불량'], 
                 cbar_kws={'label': '샘플 수'})  # 색상 바에 레이블 추가

# 1종 오류와 2종 오류 표시
ax.text(1.5, 0.6, '1종 오류', ha='center', va='center', fontsize=12, color='blue', weight='bold')  # 오른쪽 위
ax.text(0.5, 1.6, '2종 오류', ha='center', va='center', fontsize=12, color='red', weight='bold')  # 왼쪽 아래

# 축과 타이틀 설정
plt.xlabel('예측값', fontsize=12)
plt.ylabel('실제값', fontsize=12)
plt.title('Confusion Matrix', fontsize=15)
plt.xticks(rotation=0, fontsize=10)
plt.yticks(rotation=0, fontsize=10)
plt.tight_layout()  # 레이아웃 조정
plt.show()









# 각 차수에 대해 SMOTE 적용 후 모델 평가
results_smote = {}
for degree in [1, 2, 3]:
    accuracy, precision, recall, f1, roc_auc, y_pred = evaluate_model(X_train_resampled, X_test, y_train_resampled, y_test, degree, best_params)
    results_smote[degree] = {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1,
        'ROC AUC': roc_auc,
        'Y_Pred' : y_pred
    }

# 결과 출력
results_df_smote = pd.DataFrame(results_smote).T
results_df_smote.index.name = 'Polynomial Degree'
results_df_smote.columns = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC', 'Y_Pred']

print(results_df_smote)

######################################


# SMOTE만 적용 후 모델 평가
# LightGBM 모델 학습
lgb_model_smote = lgb.LGBMClassifier(random_state=42, **best_params)
lgb_model_smote.fit(X_train_resampled, y_train_resampled)

# 예측 및 성능 평가
y_pred_smote = lgb_model_smote.predict(X_test)
y_pred_proba_smote = lgb_model_smote.predict_proba(X_test)[:, 1]

# 성능 측정
accuracy_smote = accuracy_score(y_test, y_pred_smote)
precision_smote = precision_score(y_test, y_pred_smote)
recall_smote = recall_score(y_test, y_pred_smote)
f1_smote = f1_score(y_test, y_pred_smote)
roc_auc_smote = roc_auc_score(y_test, y_pred_proba_smote)

# 결과 출력
results_smote_only = pd.DataFrame({
    'Accuracy': [accuracy_smote],
    'Precision': [precision_smote],
    'Recall': [recall_smote],
    'F1 Score': [f1_smote],
    'ROC AUC': [roc_auc_smote]
}, index=['SMOTE Only'])

print(results_smote_only)



############################################


# 스모트를 하기로 결정했고
# 스모트 + Light GBM 
# 얘는 0.999772   0.999824  0.998070 0.998946  0.999985


smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# LightGBM 모델 학습
lgb_model_lgb2 = lgb.LGBMClassifier(random_state=42, **best_params)
lgb_model_lgb2.fit(X_train_resampled, y_train_resampled)

# 변수 중요도 추출
importances = lgb_model_lgb2.feature_importances_
indices = np.argsort(importances)[::-1]

# 상위 중요도를 100을 기준으로 스케일링
scaled_importances = importances / importances[indices[0]] * 100

# 변수 중요도 시각화 (상위 20개의 중요한 변수만 표시)
plt.figure(figsize=(12, 6))
sns.barplot(x=X_train.columns[indices][:20], y=scaled_importances[indices][:20])
plt.title('Top 20 Feature Importances from LightGBM')
plt.xlabel('Features')
plt.ylabel('Importance (Scaled to 100)')
plt.xticks(rotation=90)

# 상위 5개의 변수 중요도 값을 그래프에 표시
for i in range(5):
    plt.text(x=i, y=scaled_importances[indices[i]], s=f'{scaled_importances[indices[i]]:.2f}', ha='center', va='bottom', color='red', fontsize=12)

plt.show()




