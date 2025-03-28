# 피어슨 상관계수로 분석했을 때와 스피어만 상관계수로 비선형 관계를 해결했을 때의 성과 비교하기
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy import stats

from sklearn.preprocessing import PowerTransformer, MinMaxScaler, StandardScaler, PolynomialFeatures
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report, confusion_matrix, make_scorer
from imblearn.over_sampling import SMOTE


# 범주형 변수 기준으로 색 나눠서 보기
sns.scatterplot(data=df, x='설명변수1', y='설명변수2', hue='설명변수3')
plt.show()

# 또는 범주별로 서브플롯 나눠서 보기
sns.lmplot(data=df, x='설명변수1', y='설명변수2', hue='설명변수3', col='설명변수수3', fit_reg=True)  # 선형 회귀선 포함
plt.show()



# 데이터 불러오기
file_dir = input("데이터 파일의 절대 경로를 입력하세요 :")
df = pd.read_csv(file_dir)
# C:\Users\USER\Documents\D\LS_bigdataschool_3\defective_data\1주_실습데이터.csv

df.head()

# X4, X13 고정 변수 제거
df = df.drop(columns=['X4','X13'])
# 완전히 중복되는 변수 : X18 , X19 , X20
# (X6, X20) , (X8, X18), (X12, X19)
df = df.drop(columns=['X18' , 'X19' , 'X20'])

df.describe()


# correlation
pearson_m = df.corr()
plt.figure(figsize=(12,10))
sns.heatmap(pearson_m , annot=True, fmt=".2f" , cmap="coolwarm", square=True, cbar_kws={'shrink': .8}, annot_kws={'size':8})
					# annot=True : 각 셀에 값을 표시해줌.  # fmt=".2f" : 숫자를 소수점 2자리까지 표시   # cmap : 팔레트 지
					# square=True : 각 셀의 비율을 정사각형으로 함  # cbar_kws : 색상바 옵션 {'shrink': .8} 이면 색상바를 80%로 줄
					# annot_kws : 셀에 표시될 텍스트 크기 조정 옵션. {'size': 8} 이면 8 크기로 설정
plt.title('pearson correlation marix', fontsize=14)
plt.rcParams['font.family'] = 'Malgun Gothic'  # 한글 깨짐 해결
plt.rcParams['axes.unicode_minus'] = False  # 음수 깨짐 해결
plt.xticks(rotation= 45, ha='right', fontsize=10)  # ha='right' : 오른쪽 정
plt.yticks(rotation=0, fontsize=10)
plt.tight_layout()
plt.show()

spearman_m = df.corr(method='spearman')
plt.figure(figsize=(12,10))
sns.heatmap(spearman_m , annot=True, fmt=".2f" , cmap="coolwarm", square=True, cbar_kws={'shrink': .8}, annot_kws={'size':8})
					# annot=True : 각 셀에 값을 표시해줌.  # fmt=".2f" : 숫자를 소수점 2자리까지 표시   # cmap : 팔레트 지
					# square=True : 각 셀의 비율을 정사각형으로 함  # cbar_kws : 색상바 옵션 {'shrink': .8} 이면 색상바를 80%로 줄
					# annot_kws : 셀에 표시될 텍스트 크기 조정 옵션. {'size': 8} 이면 8 크기로 설정
plt.title('spearman correlation marix', fontsize=14)
plt.rcParams['font.family'] = 'Malgun Gothic'  # 한글 깨짐 해결
plt.rcParams['axes.unicode_minus'] = False  # 음수 깨짐 해결
plt.xticks(rotation=45, ha='right', fontsize=10)  # ha='right' : 오른쪽 정
plt.yticks(rotation=0, fontsize=10)
plt.tight_layout()
plt.show()


# 산점도 확인
plt.rcParams['font.family'] = 'Malgun Gothic'  # 한글 깨짐 해결
plt.rcParams['axes.unicode_minus'] = False  # 음수 깨짐 해결
sns.pairplot(df)
plt.show()

# 피어슨 높음, 스피어만 높음 : (X7, X16) (X9, X14) => X14, X16 제거
# 피어슨 높음 : (X7, X10) , (X7,X16), (X7, X17) , (X9, X14) , (X10, X16) , (X10, X17), (X16,X17) ⇒ X10, X14, X16, X17 제거?


# VIF
vif_X = df.drop(columns=['Y'])
vif_data = pd.DataFrame()
vif_data['변수'] = vif_X.columns
vif_data['VIF'] = [variance_inflation_factor(vif_X.values, i) for i in range(vif_X.shape[1])]

high_vif_features = vif_data[vif_data['VIF'] > 10]['변수']
print("VIF가 높은 변수들:", high_vif_features.values) 

# ['X3' 'X5' 'X7' 'X8' 'X9' 'X10' 'X12' 'X14' 'X15' 'X16' 'X17'] 가 10이 넘음음
# 이중 진짜 선형 관계인 건 (X14, X16) 밖에 없음
# 이거 제거 안 하고 변수 변환 하고 나서의 성능을 봐봤으니까
# 지금은 이거 제거하고 나서 성능 보기기

df2 = df.drop(columns = ['X3', 'X5', 'X7', 'X8', 'X9', 'X10', 'X12', 'X14', 'X15', 'X16', 'X17'])

df2.head()



for i in df2.columns:
	print(f'{i}컬럼의 unique 개수 :',len(df2[i].unique()))


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



# [분포 확인] - df2의 히스토그램 그리기
columns_to_plot = [col for col in df2.columns[:-1]]

plt.figure(figsize=(20, 12))
for i, column in enumerate(columns_to_plot, 1):
    plt.subplot(5, 3, i) 
    sns.histplot(df2[column], bins=100, kde=True)  # 히스토그램과 커널 밀도 추정
    plt.title(f'Distribution of {column}')
plt.tight_layout()



# [상관계수 확인] - df2 상관계수 시각화
pearson_m2 = df2.corr()
plt.figure(figsize=(14, 12))
sns.heatmap(pearson_m2, annot=True, fmt=".2f", cmap='coolwarm', square=True, linewidths=0.5, annot_kws={"size": 9})
plt.title('pearson Correlation Matrix Heatmap')


spearman_m2 = df2.corr()
plt.figure(figsize=(14, 12))
sns.heatmap(spearman_m2, annot=True, fmt=".2f", cmap='coolwarm', square=True, linewidths=0.5, annot_kws={"size": 9})
plt.title('spearman Correlation Matrix Heatmap')
# 피어슨도 스피어만도 값이 안 높음


# [boxcox] - df2의 box-cox
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
box_df = box_add_df.iloc[:,4:]

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


# < - boxcox 먼저 하고 다중공선성 해결하기
# [다중공선성 확인] - X4 , X13, X18 , X19 , X20 제거한 df2의 다중공선성 확인
vif_X = box_std_df.drop(columns=['Y'])
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


