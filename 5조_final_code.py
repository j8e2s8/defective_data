import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc 
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import PowerTransformer, MinMaxScaler, StandardScaler, PolynomialFeatures
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report, confusion_matrix, make_scorer
from imblearn.over_sampling import SMOTE
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier, plot_importance
import lightgbm as lgb


# 데이터 불러오기
df = pd.read_csv('1주_실습데이터.csv')

df.head()

# Y변수를 효율적인 계산을 위해 boolean으로 처리
df['Y'] = df['Y'].astype('boolean')


# ----------------------------------------------------


# [상관계수 확인] - 원본 데이터 상관계수 시각화
correlation_matrix = df.corr()
plt.figure(figsize=(14, 12))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True, linewidths=0.5, annot_kws={"size": 9})
plt.title('Correlation Matrix Heatmap')


# [변수 제거 과정]
# 고정값 변수 : X4 , X13
# 완전히 중복되는 변수 : X18 , X19 , X20
# df2 : X4 , X13, X18 , X19 , X20 제거
df2 = df.drop(columns=(['X4', 'X13', 'X18' ,'X19' , 'X20']))
df2.columns



# [이상치 확인] - X4 , X13, X18 , X19 , X20 제거한 df2의 boxplot 그리기
sns.boxplot(data=df2)



# [이상치 비율 확인] - 각 열별 이상치 비율을 계산하는 함수만들어서 이상치 비율 그리기
def calculate_outlier_ratio(df):
    outlier_ratios = {}
    for column in df.columns:
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # 이상치의 개수 계산
        outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
        outlier_ratio = len(outliers) / len(df) * 100  # 이상치 비율 계산 (퍼센트)

        outlier_ratios[column] = outlier_ratio

    return outlier_ratios


outlier_ratios = calculate_outlier_ratio(df2)

for column, ratio in outlier_ratios.items():
    print(f"{column}의 이상치 비율: {ratio:.2f}%")



# [분포 확인] - X4 , X13, X18 , X19 , X20 제거한 df2의 히스토그램 그리기
columns_to_plot = [col for col in df2.columns[:-1]]

plt.figure(figsize=(20, 12))

for i, column in enumerate(columns_to_plot, 1):
    plt.subplot(5, 3, i) 
    sns.histplot(df2[column], bins=100, kde=True)  # 히스토그램과 커널 밀도 추정
    plt.title(f'Distribution of {column}')

plt.tight_layout()



# [상관계수 확인] - X4 , X13, X18 , X19 , X20 제거한 df2 상관계수 시각화
correlation_matrix = df2.corr()
plt.figure(figsize=(14, 12))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True, linewidths=0.5, annot_kws={"size": 9})
plt.title('Correlation Matrix Heatmap')


# [boxcox] - X4 , X13, X18 , X19 , X20 제거한 df2의 box-cox
box_add_df = df2.copy()
boxcox_vars = []

for column in df2.columns[:-1]:
    transformed_variable, lambda_value = stats.boxcox(df2[column] + 1)
    box_add_df[f'boxcox_{column}'] = transformed_variable
    boxcox_vars.append((column, lambda_value))

for original, lambda_val in boxcox_vars:
    print(f"{original}의 최적의 Box-Cox 변환 λ 값: {lambda_val}")

box_add_df.columns
np.where(box_add_df.columns == 'Y')
box_df = box_add_df.iloc[:,15:]

# box_add_df 데이터셋 : 기존에 있는 변수 + boxcox 변환한 변수 (boxcox_어쩌구) 포함한 데이터셋
# box_df 데이터셋 : boxcox 변환한 변수 (boxcox_어쩌구)만 포함한 데이터셋



# [분포 확인] - boxcox 변환한 box_df의 히스토그램 그리기
columns_to_plot = [col for col in box_df.columns[1:]]

plt.figure(figsize=(20, 12))

for i, column in enumerate(columns_to_plot, 1):
    plt.subplot(5, 4, i)  
    sns.histplot(box_df[column], bins=100, kde=True)  # 히스토그램과 커널 밀도 추정
    plt.title(f'Distribution of {column}')

plt.tight_layout()




# [표준화_Z-score] - Y컬럼 제외하고 수치컬럼 표준화하기
scaler = StandardScaler()
box_std_df = pd.DataFrame(scaler.fit_transform(box_df.iloc[:,1:]), columns=box_df.columns[1:])

box_std_df = pd.concat([box_std_df, box_df['Y']], axis=1)
box_std_df['Y'].dtype



# [분포 확인] - boxcox 변환후 표준화 후 box_std_df의 히스토그램 그리기
columns_to_plot = [col for col in box_std_df.columns[1:]]

plt.figure(figsize=(20, 12))

for i, column in enumerate(columns_to_plot, 1):
    plt.subplot(5, 3, i)  # 5x3 그리드로 서브플롯 만들기
    sns.histplot(box_std_df[column], bins=100, kde=True)  # 히스토그램과 커널 밀도 추정
    plt.title(f'Distribution of {column}')

plt.tight_layout()



# [다중공선성 확인] - X4 , X13, X18 , X19 , X20 제거한 df2의 다중공선성 확인
vif_X = df2.drop(columns=['Y'])
vif_data = pd.DataFrame()
vif_data['변수'] = vif_X.columns
vif_data['VIF'] = [variance_inflation_factor(vif_X.values, i) for i in range(vif_X.shape[1])]

high_vif_features = vif_data[vif_data['VIF'] > 10]['변수']
print("VIF가 높은 변수들:", high_vif_features.values) 
# 다중공선성 높은 변수 : X7, X9, X10, X14, X16, X17 

# [다중공선성 확인] - boxcox 변환후 box_df의 다중공선성 확인
vif_X = box_df.drop(columns=['Y'])
vif_data = pd.DataFrame()
vif_data['변수'] = vif_X.columns
vif_data['VIF'] = [variance_inflation_factor(vif_X.values, i) for i in range(vif_X.shape[1])]

high_vif_features = vif_data[vif_data['VIF'] > 10]['변수']
print("VIF가 높은 변수들:", high_vif_features.values) 
# 다중공선성 높은 변수 : 'boxcox_X2', 'boxcox_X9', 'boxcox_X11', 'boxcox_X12', 'boxcox_X14'

# [다중공선성 확인] - boxcox 변환후 표준화 후 box_std_df의 다중공선성 확인
vif_X = box_std_df.drop(columns=['Y'])
vif_data = pd.DataFrame()
vif_data['변수'] = vif_X.columns
vif_data['VIF'] = [variance_inflation_factor(vif_X.values, i) for i in range(vif_X.shape[1])]

high_vif_features = vif_data[vif_data['VIF'] > 10]['변수']
print("VIF가 높은 변수들:", high_vif_features.values) 
# 다중공선성 높은 변수 :'boxcox_X2' 'boxcox_X6' 'boxcox_X7' 'boxcox_X9' 'boxcox_X10' 'boxcox_X11' ,'boxcox_X14' 'boxcox_X16' 'boxcox_X17'


# [다중공선성 해결] - boxcox 변환후 표준화 후 box_std_df의 다중공선성 높은 변수 제거
vif_df1 = box_std_df.drop(columns =['boxcox_X2', 'boxcox_X9', 'boxcox_X11', 'boxcox_X12', 'boxcox_X14'])
vif_df1.columns


# ----------------------------------------------------


# [전처리 후 데이터][train, test 분리] - 데이터셋의 90% 학습셋, 10% 테스트셋으로 분리
y = vif_df1['Y']
X = vif_df1.drop(columns = ('Y'))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42, stratify=y)



# [전처리 후 데이터][로지스틱 분석]
lg_model = LogisticRegression(max_iter=1000, random_state=42)

cv_scores = cross_val_score(lg_model, X_train, y_train, cv=5, scoring='accuracy')
print("Cross-Validation Accuracy Scores:", cv_scores)
print("Mean Cross-Validation Accuracy: {:.4f}".format(cv_scores.mean()))

lg_model.fit(X_train, y_train)
y_pred = lg_model.predict(X_test)



# [전처리 후 데이터][성능 평가] - 로지스틱 성능 평가
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred)

print(f'Accuracy: {accuracy:.4f}')
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1 Score: {f1:.4f}')
print(f'ROC AUC: {roc_auc:.4f}')

# Accuracy: 0.9980
# Precision: 0.9998
# Recall: 0.9821
# F1 Score: 0.9909
# ROC AUC: 0.9910



# [전처리 후 데이터][의사결정 나무 분석]
dt_model = DecisionTreeClassifier(random_state=42)

cv_scores = cross_val_score(dt_model, X_train, y_train, cv=5, scoring='accuracy')
print("Cross-Validation Accuracy Scores:", cv_scores)
print("Mean Cross-Validation Accuracy: {:.4f}".format(cv_scores.mean()))

dt_model.fit(X_train, y_train)
y_pred = dt_model.predict(X_test)



# [전처리 후 데이터][성능 평가] - 의사결정 나무 성능 평가
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred)

print(f'Accuracy: {accuracy:.4f}')
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1 Score: {f1:.4f}')
print(f'ROC AUC: {roc_auc:.4f}')

# Accuracy: 0.9992
# Precision: 0.9967
# Recall: 0.9963
# F1 Score: 0.9965
# ROC AUC: 0.9980


# ----------------------------------------------------


# [전처리 전 데이터][train, test 분리] - 데이터셋의 90% 학습셋, 10% 테스트셋으로 분리
y = df2['Y']
X = df2.drop(columns = ('Y'))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42, stratify=y)



# [전처리 전 데이터][로지스틱 분석]
lg_model = LogisticRegression(max_iter=1000, random_state=42)

cv_scores = cross_val_score(lg_model, X_train, y_train, cv=5, scoring='accuracy')
print("Cross-Validation Accuracy Scores:", cv_scores)
print("Mean Cross-Validation Accuracy: {:.4f}".format(cv_scores.mean()))

lg_model.fit(X_train, y_train)
y_pred = lg_model.predict(X_test)



# [전처리 전 데이터][성능 평가] - 로지스틱 성능 평가
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred)

print(f'Accuracy: {accuracy:.4f}')
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1 Score: {f1:.4f}')
print(f'ROC AUC: {roc_auc:.4f}')

# Accuracy: 0.9928
# Precision: 0.9978
# Recall: 0.9353
# F1 Score: 0.9655
# ROC AUC: 0.9675



# [전처리 전 데이터][의사결정 나무 분석]
dt_model = DecisionTreeClassifier(random_state=42)

cv_scores = cross_val_score(dt_model, X_train, y_train, cv=5, scoring='accuracy')
print("Cross-Validation Accuracy Scores:", cv_scores)
print("Mean Cross-Validation Accuracy: {:.4f}".format(cv_scores.mean()))

dt_model.fit(X_train, y_train)
y_pred = dt_model.predict(X_test)



# [전처리 전 데이터][성능 평가] - 의사결정 나무 성능 평가
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred)

print(f'Accuracy: {accuracy:.4f}')
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1 Score: {f1:.4f}')
print(f'ROC AUC: {roc_auc:.4f}')

#Accuracy: 0.9993
#Precision: 0.9967
#Recall: 0.9967
#F1 Score: 0.9967
#ROC AUC: 0.9981



# ----------------------------------------------------


# [최종 모델 의사결정 나무 시각화] - 나무 가지 분할 그림
dt_model_m = DecisionTreeClassifier(random_state=42, max_depth=3)
dt_model_m.fit(X_train, y_train)

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.figure(figsize=(16, 10))
plot_tree(dt_model_m, feature_names=X_train.columns, class_names=['양품', '불량'], filled=True, rounded=True, fontsize=10)
plt.title('Simplified Decision Tree for Defect Classification (max_depth=3)')
plt.show()


# [KDE(커널 밀도 추정) 시각화] - 불량품 (Y=1)과 양품 (Y=0) 각각에 대해 KDE(커널 밀도 추정) 그리기
plt.figure(figsize=(10, 6))
sns.kdeplot(data=df[df['Y'] == 1], x='X3', fill=True, color='red', label='불량품', alpha=0.3)
sns.kdeplot(data=df[df['Y'] == 0], x='X3', fill=True, color='blue', label='양품', alpha=0.3)
plt.title('KDE plot of X3 for 불량품 vs 양품')
plt.xlabel('X3')
plt.ylabel('Density')
plt.legend()
plt.show()



# [최종 모델 의사결정 나무 시각화] - 가지 분할 산점도 그림
plt.xlim([0,0.75])
plot = sns.scatterplot(data = df2 , x='X3' , y='X15', hue='Y', alpha=0.3)
sns.lineplot(x=[-1,0.406], y=[0.027,0.027], color='black')
plt.axvline(x=0.406, color='black')
handles, labels = plot.get_legend_handles_labels()
new_labels = ['양품', '불량']  
plt.legend(handles, new_labels, title='Y')



# [최종 모델 의사결정 나무 시각화] - 가지 분할 산점도 그림
plt.xlim([0,0.75])
plot = sns.scatterplot(data = df2 , x='X3' , y='X17', hue='Y', alpha=0.3)
sns.lineplot(x=[0.406,0.75], y=[0.564,0.564], color='black')
plt.axvline(x=0.406, color='black')
handles, labels = plot.get_legend_handles_labels()
new_labels = ['양품', '불량'] 
plt.legend(handles, new_labels, title='Y')



# [최종 모델 의사결정 나무 시각화] - 변수 중요도 막대그래프 그림
importances = dt_model.feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(10, 6))
sns.barplot(y=importances[indices], x=X.columns[indices])
plt.title('Feature Importance (Decision Tree)')
plt.xlabel('Importance')
plt.ylabel('Features')

for i in range(3):
    plt.text(x=i, y=importances[indices[i]], s=f'{importances[indices[i]]:.2f}', ha='center', va='bottom', color='red', fontsize=12)



# ----------------------------------------------------

# [SMOTE 알아보기][train, test 분리] - 데이터셋의 70% 학습셋, 30% 테스트셋으로 분리
X = df2.drop('Y', axis=1)
y = df2['Y']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# [SMOTE 알아보기][데이터에 SMOTE 적용] - 소수 클래스에 대한 데이터 생성
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# [SMOTE 알아보기][Y라벨 변경] - 라벨을 문자열로 변환 후 양품/불량품으로 변경
y_train = y_train.astype(str).replace({'True': '불량품', 'False': '양품'})
y_train_resampled = y_train_resampled.astype(str).replace({'True': '불량품', 'False': '양품'})



# SMOTE 적용 전후의 데이터 분포를 비교
# [SMOTE 알아보기][SMOTE 적용 전] - 클래스 별 막대그래프 + 갯수 및 비율 표시
def format_number(n):
    return f'{int(n):,}건'  # 숫자를 천 단위로 포맷팅하는 함수

plt.figure(figsize=(16, 12))
plt.subplot(2, 2, 1)
total_count = len(y_train)
ax = sns.countplot(x=y_train, palette={'양품': 'gray', '불량품': 'red'})
for p in ax.patches:
    height = p.get_height()
    percentage = f'{height / total_count * 100:.2f}%'
    ax.annotate(f'{format_number(height)}\n{percentage}', 
                xy=(p.get_x() + p.get_width() / 2., height / 2),  # 중간쯤으로 위치 조정
                ha='center', va='center', fontsize=14, color='white', fontweight='bold', xytext=(0, 0),
                textcoords='offset points')
plt.title('SMOTE 적용 전 양품 불량품 분포', fontsize=16)
plt.xlabel('양품 불량품 구분', fontsize=14)
plt.ylabel('데이터 수', fontsize=14)

# [SMOTE 알아보기][SMOTE 적용 후] - 클래스 별 막대그래프 + 갯수 및 비율 표시
plt.subplot(2, 2, 2)
total_count_resampled = len(y_train_resampled)
ax = sns.countplot(x=y_train_resampled, palette={'양품': 'gray', '불량품': 'red'})
for p in ax.patches:
    height = p.get_height()
    percentage = f'{height / total_count_resampled * 100:.2f}%'
    ax.annotate(f'{format_number(height)}\n{percentage}', 
                xy=(p.get_x() + p.get_width() / 2., height / 2),  # 중간쯤으로 위치 조정
                ha='center', va='center', fontsize=14, color='white', fontweight='bold', xytext=(0, 0),
                textcoords='offset points')
plt.title('SMOTE 적용 후 양품 불량품 분포', fontsize=16)
plt.xlabel('양품 불량품 구분', fontsize=14)
plt.ylabel('데이터 수', fontsize=14)

# [SMOTE 알아보기][SMOTE 적용 전] - 클래스 별 산점도 (X3, X15)로 클래스 별 분포 알아보기
plt.subplot(2, 2, 3)
sns.scatterplot(x=X_train['X3'], y=X_train['X15'], hue=y_train, style=y_train,
                palette={'불량품': 'red', '양품': 'gray'}, markers={'불량품': '^', '양품': 'o'},
                s=100, alpha=0.5)
plt.title('SMOTE 적용 전 데이터 분포', fontsize=16)
plt.xlabel('X3', fontsize=14)
plt.ylabel('X15', fontsize=14)

# [SMOTE 알아보기][SMOTE 적용 후] - 클래스 별 산점도 (X3, X15)로 클래스 별 분포 알아보기
plt.subplot(2, 2, 4)
sns.scatterplot(x=X_train_resampled['X3'], y=X_train_resampled['X15'], hue=y_train_resampled, style=y_train_resampled,
                palette={'불량품': 'red', '양품': 'gray'}, markers={'불량품': '^', '양품': 'o'},
                s=100, alpha=0.5)
plt.title('SMOTE 적용 후 데이터 분포', fontsize=16)
plt.xlabel('X3', fontsize=14)
plt.ylabel('X15', fontsize=14)

plt.tight_layout()
plt.show()


# ----------------------------------------------------

# [SMOTE 모델 분석][train, test 분리] - 데이터셋의 90% 학습셋, 10% 테스트셋으로 분리
X = df2.drop('Y', axis=1)
y = df2['Y']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42, stratify=y)

# [SMOTE 모델 분석][샘플링] - 너무 오래거려서 학습 데이터에서 20%만 사용
X_train_sample, _, y_train_sample, _ = train_test_split(X_train, y_train, test_size=0.8, random_state=42, stratify=y_train)

# [SMOTE 모델 분석][데이터에 SMOTE 적용] - 소수 클래스에 대한 데이터 생성
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_sample, y_train_sample)

# [SMOTE 모델 분석][표준화 Z-score] - SVM과 KNN에서 사용할 데이터 셋
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_sample)
X_test_scaled = scaler.transform(X_test)
X_train_resampled_scaled = scaler.fit_transform(X_train_resampled)
X_test_resampled_scaled = scaler.transform(X_test)

# [SMOTE 모델 분석][사용할 모델 리스트]
models = {
    'Logistic Regression': LogisticRegression(max_iter=500),
    'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=5),  
    'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100, max_depth=5, n_jobs=-1), 
    'Gradient Boosting': GradientBoostingClassifier(random_state=42, n_estimators=100, max_depth=3),
    'XGBoost': XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss', n_estimators=100, n_jobs=-1), 
    'LightGBM': LGBMClassifier(random_state=42, n_estimators=100, n_jobs=-1),  
    'SVM': SVC(random_state=42, probability=True),
    'KNN': KNeighborsClassifier(n_neighbors=5, n_jobs=-1) 
}

# [SMOTE 모델 분석][분석 + 성능 평가] 
# - 성능 지표 계산 함수 만들기
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

# - 병렬 처리를 위한 함수
def process_model(name, model, X_train, y_train, X_test, y_test, X_train_resampled, y_train_resampled, X_test_resampled):
    if name in ['SVM', 'KNN']:  # SVM과 KNN은 스케일링된 데이터 사용
        metrics_before = evaluate_model(model, X_train_scaled, y_train, X_test_scaled, y_test)
        metrics_after = evaluate_model(model, X_train_resampled_scaled, y_train_resampled, X_test_resampled_scaled, y_test)
    else:
        metrics_before = evaluate_model(model, X_train, y_train, X_test, y_test)
        metrics_after = evaluate_model(model, X_train_resampled, y_train_resampled, X_test, y_test)
    
    return (name, metrics_before, metrics_after)

# - 성능 비교를 위한 빈 DataFrame 생성
columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC-AUC', 'Training Time (s)']
results_before_smote = pd.DataFrame(columns=columns)
results_after_smote = pd.DataFrame(columns=columns)

# - 모델 성능을 병렬로 계산
results = Parallel(n_jobs=-1)(delayed(process_model)(
    name, model, X_train_sample, y_train_sample, X_test, y_test, X_train_resampled, y_train_resampled, X_test_resampled_scaled
) for name, model in models.items())

# - 결과 저장
for name, metrics_before, metrics_after in results:
    results_before_smote = pd.concat([results_before_smote, pd.DataFrame([[name] + list(metrics_before)], columns=columns)], ignore_index=True)
    results_after_smote = pd.concat([results_after_smote, pd.DataFrame([[name] + list(metrics_after)], columns=columns)], ignore_index=True)

# - 결과 출력
print("SMOTE 적용 전 성능 지표 및 학습 시간")
print(results_before_smote)
print("\nSMOTE 적용 후 성능 지표 및 학습 시간")
print(results_after_smote)

# - 결과 테이블을 만들어 엑셀표처럼 출력하는 코드
def display_results_table(before_smote, after_smote):
    combined_results = pd.DataFrame({
        'Model': before_smote['Model'],
        'Accuracy (Before)': before_smote['Accuracy'],
        'Accuracy (After)': after_smote['Accuracy'],
        'Precision (Before)': before_smote['Precision'],
        'Precision (After)': after_smote['Precision'],
        'Recall (Before)': before_smote['Recall'],
        'Recall (After)': after_smote['Recall'],
        'F1 Score (Before)': before_smote['F1 Score'],
        'F1 Score (After)': after_smote['F1 Score'],
        'ROC-AUC (Before)': before_smote['ROC-AUC'],
        'ROC-AUC (After)': after_smote['ROC-AUC'],
        'Training Time (Before)': before_smote['Training Time (s)'],
        'Training Time (After)': after_smote['Training Time (s)'],
    })

    # 스타일 지정: 엑셀 표처럼 보기 좋게
    styled_results = combined_results.style.format({
        'Accuracy (Before)': "{:.4f}",
        'Accuracy (After)': "{:.4f}",
        'Precision (Before)': "{:.4f}",
        'Precision (After)': "{:.4f}",
        'Recall (Before)': "{:.4f}",
        'Recall (After)': "{:.4f}",
        'F1 Score (Before)': "{:.4f}",
        'F1 Score (After)': "{:.4f}",
        'ROC-AUC (Before)': "{:.4f}",
        'ROC-AUC (After)': "{:.4f}",
        'Training Time (Before)': "{:.2f} s",
        'Training Time (After)': "{:.2f} s",
    }).set_caption("SMOTE 적용 전후 성능 비교")

    return styled_results

# - 엑셀표처럼 결과 테이블 출력
display_results_table(results_before_smote, results_after_smote)



# [SMOTE 모델 분석][시각화] - xgboost, lightgbm 평가 지표 막대 그래프 그림
xgboost_before = [0.9995, 1.0, 0.9956, 0.9978, 0.9996, 2.38]
xgboost_after = [0.9995, 0.9991, 0.9967, 0.9979, 0.9997, 3.73]
lightgbm_before = [0.9995, 0.9996, 0.9954, 0.9975, 0.9997, 4.11]
lightgbm_after = [0.9996, 0.9995, 0.9970, 0.9982, 0.9997, 2.67]
metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC-AUC', 'Training Time (s)']

fig, axs = plt.subplots(2, 3, figsize=(15, 8))  
axs = axs.ravel()  # 2D 배열을 1D로 펼침

for i in range(6):  # 5개의 성능 지표 + 1개의 학습 시간
    axs[i].bar(['XGBoost Before', 'XGBoost After', 'LightGBM Before', 'LightGBM After'], 
               [xgboost_before[i], xgboost_after[i], lightgbm_before[i], lightgbm_after[i]], 
               color=['lightblue', 'skyblue', 'lightcoral', 'salmon'])
    axs[i].set_title(metrics[i])
    
    # 성능 지표 y축 범위 조정 (학습 시간 제외)
    if i < 5:
        axs[i].set_ylim(0.995, 1.0)  # 0.995에서 1까지 범위
    else:
        axs[i].set_ylim(0, 4.5)  # 학습 시간은 0부터 시작

    # 막대 위에 값 표시
    for j, v in enumerate([xgboost_before[i], xgboost_after[i], lightgbm_before[i], lightgbm_after[i]]):
        axs[i].text(j, v + 0.00005 if i < 5 else v + 0.1, f'{v:.4f}' if i < 5 else f'{v:.2f}', 
                    ha='center', va='bottom', fontsize=10)
plt.tight_layout()
plt.show()


# ----------------------------------------------------

# [XGBoost vs LightGBM][train, test 분리] - 데이터셋의 90% 학습셋, 10% 테스트셋으로 분리
X = df2.drop('Y', axis=1)
y = df2['Y']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42, stratify=y)


# [XGBoost vs LightGBM][XGBoost 분석] - 하이퍼 파라미터 튜닝하기
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 6, 9],
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'gamma': [0, 0.1, 0.5]
}

scoring = {
    'accuracy': 'accuracy',
    'precision': make_scorer(precision_score),
    'recall': make_scorer(recall_score),
    'f1': 'f1',
    'roc_auc': 'roc_auc'
}

xgb_model = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, scoring=scoring, refit='f1', cv=3, verbose=1, n_jobs=-1)
grid_search.fit(X_train, y_train)

print("Best Hyperparameters:", grid_search.best_params_)
#'colsample_bytree': 0.6, 'gamma': 0, 'learning_rate': 0.1, 'max_depth': 9, 'n_estimators': 300, 'subsample': 0.6

best_xgb_model = grid_search.best_estimator_
y_pred = best_xgb_model.predict(X_test)


# [XGBoost vs LightGBM][XGBoost 성능 평가] 
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

print("\nClassification Report:\n", classification_report(y_test, y_pred))


# [XGBoost vs LightGBM][XGBoost 시각화] - 혼동 행렬 그리기
rc('font', family='Malgun Gothic')
plt.rcParams['axes.unicode_minus'] = False  # 마이너스 기호가 깨지는 문제 방지

conf_matrix = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 5))
ax = sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
                 xticklabels=['양품', '불량'], yticklabels=['양품', '불량'], 
                 cbar_kws={'label': '샘플 수'})  # 색상 바에 레이블 추가

ax.text(1.5, 0.6, '1종 오류', ha='center', va='center', fontsize=12, color='blue', weight='bold')  # 오른쪽 위
ax.text(0.5, 1.6, '2종 오류', ha='center', va='center', fontsize=12, color='red', weight='bold')  # 왼쪽 아래

plt.xlabel('예측값', fontsize=12)
plt.ylabel('실제값', fontsize=12)
plt.title('Confusion Matrix', fontsize=15)
plt.xticks(rotation=0, fontsize=10)
plt.yticks(rotation=0, fontsize=10)
plt.tight_layout()  # 레이아웃 조정
plt.show()



# [XGBoost vs LightGBM][Light GBM 분석] - 하이퍼 파라미터 튜닝하기
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

scoring = {
    'accuracy': 'accuracy',
    'precision': make_scorer(precision_score),
    'recall': make_scorer(recall_score),
    'f1': 'f1',
    'roc_auc': 'roc_auc'
}

lgb_model = lgb.LGBMClassifier(random_state=42)
grid_search_lgb = GridSearchCV(estimator=lgb_model, param_grid=param_grid_lgb, scoring=scoring, refit='f1', cv=3, verbose=1, n_jobs=-1)
grid_search_lgb.fit(X_train, y_train)

print("Best Hyperparameters for LightGBM:", grid_search_lgb.best_params_)
#Best Hyperparameters for LightGBM: {'colsample_bytree': 0.6, 'learning_rate': 0.1, 'max_depth': 9, 'n_estimators': 300, 'reg_alpha': 0, 'reg_lambda': 0.5, 'subsample': 0.6}

best_lgb_model = grid_search_lgb.best_estimator_
y_pred_lgb = best_lgb_model.predict(X_test)


# [XGBoost vs LightGBM][LightGBM 성능 평가] 
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



# [XGBoost vs LightGBM][LightGBM 시각화] - 혼동 행렬 그리기
conf_matrix = confusion_matrix(y_test, y_pred_lgb)

plt.figure(figsize=(6, 5))
ax = sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
                 xticklabels=['양품', '불량'], yticklabels=['양품', '불량'], 
                 cbar_kws={'label': '샘플 수'})  # 색상 바에 레이블 추가

ax.text(1.5, 0.6, '1종 오류', ha='center', va='center', fontsize=12, color='blue', weight='bold')  # 오른쪽 위
ax.text(0.5, 1.6, '2종 오류', ha='center', va='center', fontsize=12, color='red', weight='bold')  # 왼쪽 아래

plt.xlabel('예측값', fontsize=12)
plt.ylabel('실제값', fontsize=12)
plt.title('Confusion Matrix', fontsize=15)
plt.xticks(rotation=0, fontsize=10)
plt.yticks(rotation=0, fontsize=10)
plt.tight_layout()  # 레이아웃 조정
plt.show()




# ----------------------------------------------------

# LightGBM으로 최종 결정
# [LightGBM 세부 분석]


# [LightGBM + Poly 세부 분석]
# LIght GBM에서 PolyNomial 이용해 파생변수 만들기


# [LightGBM 세부 분석][전처리 전] - 변수 중요도 시각화
importances = best_lgb_model.feature_importances_
indices = np.argsort(importances)[::-1]

scaled_importances = importances / importances[indices[0]] * 100

plt.figure(figsize=(12, 6))
sns.barplot(x=X_train.columns[indices][:20], y=scaled_importances[indices][:20])
plt.title('Top 20 Feature Importances from LightGBM')
plt.xlabel('Features')
plt.ylabel('Importance (Scaled to 100)')
plt.xticks(rotation=90)

for i in range(5):
    plt.text(x=i, y=scaled_importances[indices[i]], s=f'{scaled_importances[indices[i]]:.2f}', ha='center', va='bottom', color='red', fontsize=12)

plt.show()


# [LightGBM 세부 분석][poly 2차 전처리 후] - 변수 중요도 시각화
poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=False)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

poly_feature_names = poly.get_feature_names_out(X_train.columns)

X_train_poly_df = pd.DataFrame(X_train_poly, columns=poly_feature_names)
X_test_poly_df = pd.DataFrame(X_test_poly, columns=poly_feature_names)

lgb_model_poly = lgb.LGBMClassifier(random_state=42, **grid_search.best_params_)
lgb_model_poly.fit(X_train_poly_df, y_train)


importances_poly = lgb_model_poly.feature_importances_
indices_poly = np.argsort(importances_poly)[::-1]

scaled_importances_poly = importances_poly / importances_poly[indices_poly[0]] * 100

plt.figure(figsize=(12, 6))
sns.barplot(x=X_train_poly_df.columns[indices_poly][:20], y=scaled_importances_poly[indices_poly][:20])
plt.title('Top 20 Feature Importances from LightGBM with Polynomial Features (2nd Degree)')
plt.xlabel('Features')
plt.ylabel('Importance (Scaled to 100)')
plt.xticks(rotation=90)

for i in range(5):
    plt.text(x=i, y=scaled_importances_poly[indices_poly[i]], s=f'{scaled_importances_poly[indices_poly[i]]:.2f}', ha='center', va='bottom', color='red', fontsize=12)

plt.show()


# [LightGBM 세부 분석][poly 3차 전처리 후] - 변수 중요도 시각화
poly = PolynomialFeatures(degree=3, include_bias=False, interaction_only=False)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

poly_feature_names = poly.get_feature_names_out(X_train.columns)

X_train_poly_df = pd.DataFrame(X_train_poly, columns=poly_feature_names)
X_test_poly_df = pd.DataFrame(X_test_poly, columns=poly_feature_names)

lgb_model_poly = lgb.LGBMClassifier(random_state=42, **grid_search.best_params_)
lgb_model_poly.fit(X_train_poly_df, y_train)


importances_poly = lgb_model_poly.feature_importances_
indices_poly = np.argsort(importances_poly)[::-1]

scaled_importances_poly = importances_poly / importances_poly[indices_poly[0]] * 100

plt.figure(figsize=(12, 6))
sns.barplot(x=X_train_poly_df.columns[indices_poly][:20], y=scaled_importances_poly[indices_poly][:20])
plt.title('Top 20 Feature Importances from LightGBM with Polynomial Features (3rd Degree)')
plt.xlabel('Features')
plt.ylabel('Importance (Scaled to 100)')
plt.xticks(rotation=90)

for i in range(5):
    plt.text(x=i, y=scaled_importances_poly[indices_poly[i]], s=f'{scaled_importances_poly[indices_poly[i]]:.2f}', ha='center', va='bottom', color='red', fontsize=12)

plt.show()



# [LightGBM 세부 분석][Poly + LightGBM 성능비교] - Poly 1차 , 2차, 3차 동시에 성능 비교
# 1,2,3차를 한번에 비교하기 위해 def를 사용함
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


best_params = grid_search_lgb.best_params_  # 베스트 파라미터 가져오기 (이전에 찾은 최적의 파라미터를 사용함)

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

results_df = pd.DataFrame(results_poly).T
results_df.index.name = 'Polynomial Degree'
results_df.columns = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC', 'Y_Pred']

print(results_df)
#1      0.999715        1.0  0.997368  0.998682  0.999957 
#2      0.999677        1.0  0.997018  0.998507  0.999807
#3      0.999715        1.0  0.997368  0.998682  0.999782

# poly 1차 + light gbm 선택 => 1차가 가장 결과가 좋게 나와 파생변수를 안 쓰기로 결정



# [LightGBM 세부 분석][Poly 1차 + LightGBM 혼동 행렬]
conf_matrix = confusion_matrix(y_test, results_poly[1]['Y_Pred'])

plt.figure(figsize=(6, 5))
ax = sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
                 xticklabels=['양품', '불량'], yticklabels=['양품', '불량'], 
                 cbar_kws={'label': '샘플 수'})  # 색상 바에 레이블 추가

ax.text(1.5, 0.6, '1종 오류', ha='center', va='center', fontsize=12, color='blue', weight='bold')  # 오른쪽 위
ax.text(0.5, 1.6, '2종 오류', ha='center', va='center', fontsize=12, color='red', weight='bold')  # 왼쪽 아래

plt.xlabel('예측값', fontsize=12)
plt.ylabel('실제값', fontsize=12)
plt.title('Confusion Matrix', fontsize=15)
plt.xticks(rotation=0, fontsize=10)
plt.yticks(rotation=0, fontsize=10)
plt.tight_layout()  # 레이아웃 조정
plt.show()


######################3
# [LightGBM 세부 분석][성능비교] (SMOTE+ LightGBM)

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



####################################
# [LightGBM 세부 분석][성능비교] (SMOTE + Poly + LightGBM)
# SMOTE 적용
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

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

         # Accuracy    Precision      Recall      F1 Score   ROC AUC

# 폴리          0.999715     1.0        0.997368  0.998682  0.999957 

# 스모트        0.999772   0.999824     0.998070  0.998946  0.999985 

# 스모트+ 폴리  0.999772   0.999824      0.998070  0.998946  0.999985

# 폴리가 1차로 선택되었기에 2,3번의 결과가 같은 것 

# 스모트 + LightGBM 확정



############################################

# 최종 모델인 스모트 + Light GBM 의 변수 중요도 출력

# 0.999772   0.999824  0.998070 0.998946  0.999985

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




# ----------------------------------------
# 스모트 + LightGBM 최종모델의 혼동 행렬 구하기


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





# ----------------------------

# [중요 변수에 대한 파악]

# X3, X7, X17 컬럼만 선택하여 상관관계 분석
correlation_matrix = df[['X3', 'X7', 'X17']].corr()

# 상관관계 행렬 출력
correlation_matrix

# 히트맵 시각화 (글씨 크기 더 크게, 굵게 조정)
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5, annot_kws={"size": 18, "weight": "bold"})  # 글씨 크기 18, 굵게 설정
plt.title('Correlation Heatmap for X3, X7, X17', fontsize=20, weight='bold')  # 제목 크기와 굵기 설정
plt.xticks(fontsize=15, weight='bold')  # x축 글씨 크기와 굵기 설정
plt.yticks(fontsize=15, weight='bold')  # y축 글씨 크기와 굵기 설정
plt.show()



# X3, X7, X17의 분포 및 Y와의 관계를 (3, 1) 레이아웃으로 시각화

# 히스토그램 및 커널 밀도 추정 플롯 그리기 (3x1 레이아웃)
plt.figure(figsize=(6, 12))

# X3 분포
plt.subplot(3, 1, 1)
sns.histplot(df['X3'], kde=True, color='blue')
plt.title('Distribution of X3')

# X7 분포
plt.subplot(3, 1, 2)
sns.histplot(df['X7'], kde=True, color='green')
plt.title('Distribution of X7')

# X17 분포
plt.subplot(3, 1, 3)
sns.histplot(df['X17'], kde=True, color='red')
plt.title('Distribution of X17')

plt.tight_layout()
plt.show()




# KDE plot의 Y 설명을 '양품', '불량품'으로 수정한 코드

# 'Y' 값을 '양품'과 '불량'으로 변환한 컬럼 생성
df['Y_label'] = df['Y'].replace({True: '불량', False: '양품'})

# ----- KDE plot을 (2, 3) 레이아웃으로 합친 코드 (Y축 설명 수정)

fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# 첫 번째 행: 산점도 시각화
# X3과 Y의 관계
sns.scatterplot(x='X3', y='Y', data=df, hue='Y_label', palette='coolwarm', s=50, ax=axes[0, 0])
axes[0, 0].set_title('X3 vs Y', fontsize=14, fontweight='bold')
axes[0, 0].set_yticks([0, 1])
axes[0, 0].set_yticklabels(['양품', '불량'])

# X7과 Y의 관계
sns.scatterplot(x='X7', y='Y', data=df, hue='Y_label', palette='coolwarm', s=50, ax=axes[0, 1])
axes[0, 1].set_title('X7 vs Y', fontsize=14, fontweight='bold')
axes[0, 1].set_yticks([])  # Y축 제거

# X17과 Y의 관계
sns.scatterplot(x='X17', y='Y', data=df, hue='Y_label', palette='coolwarm', s=50, ax=axes[0, 2])
axes[0, 2].set_title('X17 vs Y', fontsize=14, fontweight='bold')
axes[0, 2].set_yticks([])  # Y축 제거

# 두 번째 행: KDE plot
# X3 KDE plot
sns.kdeplot(data=df, x="X3", hue="Y_label", fill=True, palette="coolwarm", alpha=0.5, ax=axes[1, 0])
axes[1, 0].set_title("KDE plot of X3 for 불량품 vs 양품", fontsize=16)

# X7 KDE plot
sns.kdeplot(data=df, x="X7", hue="Y_label", fill=True, palette="coolwarm", alpha=0.5, ax=axes[1, 1])
axes[1, 1].set_title("KDE plot of X7 for 불량품 vs 양품", fontsize=16)

# X17 KDE plot
sns.kdeplot(data=df, x="X17", hue="Y_label", fill=True, palette="coolwarm", alpha=0.5, ax=axes[1, 2])
axes[1, 2].set_title("KDE plot of X17 for 불량품 vs 양품", fontsize=16)

# Y축 제거 (두 번째 행에서)
axes[1, 1].set_ylabel('')
axes[1, 2].set_ylabel('')

# 첫 번째 행의 첫 번째 Y축만 유지하고 나머지 제거
axes[0, 1].set_ylabel('')
axes[0, 2].set_ylabel('')

plt.tight_layout()
plt.subplots_adjust(wspace=0.1, hspace=0.4)

plt.show()