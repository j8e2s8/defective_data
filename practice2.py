# 10% -> 1분 걸림
# 20% -> 
# 30% 58분
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from joblib import Parallel, delayed  # 병렬 처리를 위한 joblib
import time

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

# 샘플링: 학습 데이터에서 10%만 사용 (변경 가능)
X_train_sample, _, y_train_sample, _ = train_test_split(X_train, y_train, test_size=0.3, random_state=42, stratify=y_train)

# SMOTE 적용 (소수 클래스에 대한 데이터 생성)
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_sample, y_train_sample)

# 스케일링 (SVM과 KNN에 필요)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_sample)
X_test_scaled = scaler.transform(X_test)
X_train_resampled_scaled = scaler.fit_transform(X_train_resampled)
X_test_resampled_scaled = scaler.transform(X_test)

# 사용할 모델 리스트
models = {
    'Logistic Regression': LogisticRegression(max_iter=500),
    'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=5),  # max_depth 조정
    'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100, max_depth=5, n_jobs=-1),  # 병렬 처리 추가
    'Gradient Boosting': GradientBoostingClassifier(random_state=42, n_estimators=100, max_depth=3),
    'XGBoost': XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss', n_estimators=100, n_jobs=-1),  # 병렬 처리 추가
    'LightGBM': LGBMClassifier(random_state=42, n_estimators=100, n_jobs=-1),  # 병렬 처리 추가
    'SVM': SVC(random_state=42, probability=True),
    'KNN': KNeighborsClassifier(n_neighbors=5, n_jobs=-1)  # 병렬 처리 추가
}

# 성능 지표 계산 함수
def evaluate_model(model, X_train, y_train, X_test, y_test):
    start_time = time.time()  # 시작 시간 기록
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else np.zeros(len(y_pred))  # ROC-AUC용 확률값
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob) if np.any(y_prob) else 0
    training_time = time.time() - start_time  # 학습에 걸린 시간 계산
    
    return accuracy, precision, recall, f1, roc_auc, training_time

# 성능 비교를 위한 빈 DataFrame 생성
columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC-AUC', 'Training Time (s)']
results_before_smote = pd.DataFrame(columns=columns)
results_after_smote = pd.DataFrame(columns=columns)

# 병렬 처리를 위한 함수
def process_model(name, model, X_train, y_train, X_test, y_test, X_train_resampled, y_train_resampled, X_test_resampled):
    if name in ['SVM', 'KNN']:  # SVM과 KNN은 스케일링된 데이터 사용
        metrics_before = evaluate_model(model, X_train_scaled, y_train, X_test_scaled, y_test)
        metrics_after = evaluate_model(model, X_train_resampled_scaled, y_train_resampled, X_test_resampled_scaled, y_test)
    else:
        metrics_before = evaluate_model(model, X_train, y_train, X_test, y_test)
        metrics_after = evaluate_model(model, X_train_resampled, y_train_resampled, X_test, y_test)
    
    return (name, metrics_before, metrics_after)

# 모델 성능을 병렬로 계산
results = Parallel(n_jobs=-1)(delayed(process_model)(
    name, model, X_train_sample, y_train_sample, X_test, y_test, X_train_resampled, y_train_resampled, X_test_resampled_scaled
) for name, model in models.items())

# 결과 저장
for name, metrics_before, metrics_after in results:
    results_before_smote = pd.concat([results_before_smote, pd.DataFrame([[name] + list(metrics_before)], columns=columns)], ignore_index=True)
    results_after_smote = pd.concat([results_after_smote, pd.DataFrame([[name] + list(metrics_after)], columns=columns)], ignore_index=True)

# 결과 출력
print("SMOTE 적용 전 성능 지표 및 학습 시간")
print(results_before_smote)
print("\nSMOTE 적용 후 성능 지표 및 학습 시간")
print(results_after_smote)




# 10%
# SMOTE 적용 전 성능 지표 및 학습 시간
#                  Model  Accuracy  Precision    Recall  F1 Score   ROC-AUC     Training Time (s) 
# 0  Logistic Regression  0.933871   1.000000  0.388596  0.559697  0.990698          0.122308  
# 1        Decision Tree  0.998065   0.999465  0.982632  0.990977  0.992676          0.450837  
# 2        Random Forest  0.997362   1.000000  0.975614  0.987657  0.998646          0.720772  
# 3    Gradient Boosting  0.998975   0.997182  0.993333  0.995254  0.998924          15.916995  
# 4              XGBoost  0.999488   1.000000  0.995263  0.997626  0.999371          1.387811 
# 5             LightGBM  0.999431   1.000000  0.994737  0.997361  0.99923           2.146311  
# 6                  SVM  0.998880   1.000000  0.989649  0.994798  0.998739          24.278623 
# 7                  KNN  0.997894   0.999642  0.980877  0.990171  0.996136          4.001001


# SMOTE 적용 후 성능 지표 및 학습 시간
#                  Model  Accuracy  Precision    Recall  F1 Score   ROC-AUC     Training Time (s) 
# 0  Logistic Regression  0.973510   0.807165  0.992105  0.890131  0.997353          0.605683
# 1        Decision Tree  0.998482   0.996993  0.988947  0.992954  0.996364          1.302933  
# 2        Random Forest  0.999127   1.000000  0.991930  0.995949  0.999139          1.726347
# 3    Gradient Boosting  0.999374   0.999295  0.994912  0.997099  0.999038         23.230859
# 4              XGBoost  0.999545   0.999648  0.996140  0.997891  0.999430          1.826188  
# 5             LightGBM  0.999526   0.999824  0.995789  0.997803  0.999213          0.492509
# 6                  SVM  0.999355   0.998768  0.995263  0.997012  0.998932         70.990640  
# 7                  KNN  0.998444   0.994716  0.990877  0.992793  0.996440          3.514405  



 
 
# 20%
# SMOTE 적용 전 성능 지표 및 학습 시간
#                  Model  Accuracy  Precision    Recall  F1 Score   ROC-AUC     Training Time (s)
# 0  Logistic Regression  0.958292   0.999715  0.614561  0.761191  0.996310         0.735584  
# 1        Decision Tree  0.998178   0.999643  0.983509  0.991510  0.991736         0.996107  
# 2        Random Forest  0.997552   1.000000  0.977368  0.988555  0.998434         2.726080 
# 3    Gradient Boosting  0.999108   0.999117  0.992632  0.995864  0.998911        29.884568 
# 4              XGBoost  0.999526   1.000000  0.995614  0.997802  0.999617         2.172666 
# 5             LightGBM  0.999469   0.999648  0.995439  0.997539  0.999692         4.876973 
# 6                  SVM  0.999222   1.000000  0.992807  0.996391  0.998827        65.824214 
# 7                  KNN  0.998615   0.999645  0.987544  0.993557  0.997364         7.376495


# SMOTE 적용 후 성능 지표 및 학습 시간
#                  Model  Accuracy  Precision    Recall  F1 Score   ROC-AUC   Training Time (s) 
# 0  Logistic Regression  0.985825   0.890307  0.991053  0.937983  0.997515         1.333962  
# 1        Decision Tree  0.998672   0.998230  0.989474  0.993833  0.995908         1.536019  
# 2        Random Forest  0.999108   0.999646  0.992105  0.995862  0.999387         2.131800 
# 3    Gradient Boosting  0.999279   0.998591  0.994737  0.996660  0.999027        47.345985  
# 4              XGBoost  0.999545   0.999121  0.996667  0.997892  0.999730         3.111152 
# 5             LightGBM  0.999620   0.999472  0.997018  0.998243  0.999674         1.926407 
# 6                  SVM  0.999355   0.998943  0.995088  0.997012  0.999047       229.873770  
# 7                  KNN  0.999089   0.997535  0.994035  0.995782  0.997508         4.565485 

