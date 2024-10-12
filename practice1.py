# 어느 변수가 y에 영향을 어느정도 줬는지 1:1 관계를 볼 수 있는 패키지임. info_plot ?
# 변수간의 관계도 볼 수 있음. 



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.tree import DecisionTreeClassifier, plot_tree
# !pip install imbalanced-learn
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import PowerTransformer, MinMaxScaler, StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
import lightgbm as lgb


import os
os.getcwd()


# 데이터 불러오기
df = pd.read_csv('1주_실습데이터.csv')
df['Y'] = df['Y'].astype('boolean')

df.info()

# X_train, X_test, y_train, y_test = train_test_split(X, y , test_size=0.2, stratify=y, random_state=42)







# 시각화
def hist(df, numeric_col):
	plt.clf()
	plt.rcParams['font.family'] = 'Malgun Gothic'
	plt.rcParams['axes.unicode_minus'] = False
	a = numeric_col + "의 분포"
	plt.title(a)
	sns.histplot(df[numeric_col], stat='density')
	plt.tight_layout()
	plt.show()
	
def rel_nx_ny(df, numeric_col, y):
	plt.clf()
	plt.rcParams['font.family'] = 'Malgun Gothic'
	plt.rcParams['axes.unicode_minus'] = False
	a = numeric_col + "과 " + y +"컬럼의 관계"
	plt.title(a)
	sns.scatterplot(data=df, x=numeric_col, y=y)
	plt.tight_layout()
	plt.show()


df.info()   
df.columns

for i in df.columns:
	print(i,'변수의 값 :',df[i].unique())
	
for i in df.columns[5:]:
	print(i,'변수의 값 :',df[i].unique())


# 고정값 변수 : X4 , X13
# 완전히 중복되는 변수 : X18 , X19 , X20
# 제거

df2 = df.drop(columns=(['X4', 'X13', 'X18' ,'X19' , 'X20']))
df2.columns

len(df2[df2['Y']==0]['X7'])
len(df2[df2['Y']==1]['X7'])



# 히스토그램 그리기
columns_to_plot = [col for col in df2.columns[:-1]]

plt.figure(figsize=(20, 12))

for i, column in enumerate(columns_to_plot, 1):
    plt.subplot(5, 3, i)  # 5x3 그리드로 서브플롯 만들기
    sns.histplot(df2[column], bins=100, kde=True)  # 히스토그램과 커널 밀도 추정
    plt.title(f'Distribution of {column}')

plt.tight_layout()






hist(df, 'X1')
hist(df, 'X2')
rel_nx_ny(df, 'X2', 'Y')
rel_nx_ny(df, 'X3', 'Y')

out_df2 = df2.copy()
out_df2 = out_df2[out_df2['X2'] < 0.3]
hist(out_df2, 'X2')
rel_nx_ny(out_df2, 'X2', 'Y')

hist(df, 'X3')


# 상관계수 시각화
correlation_matrix = df2.corr()
# plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True, linewidths=0.5)
plt.title('Correlation Matrix Heatmap')
# plt.show()

df2['Y'].dtype


# box-cox
# df2 : X4 , X13 ,X18 , X19 , X20 제거한 데이터셋
box_add_df = df2.copy()
boxcox_vars = []

for column in df2.columns[:-1]:
    # Box-Cox 변환 (0이 포함된 경우를 대비해 1을 더함)
    transformed_variable, lambda_value = stats.boxcox(df2[column] + 1)
    box_add_df[f'boxcox_{column}'] = transformed_variable
    boxcox_vars.append((column, lambda_value))

# 변환된 변수와 최적의 λ 값 출력
for original, lambda_val in boxcox_vars:
    print(f"{original}의 최적의 Box-Cox 변환 λ 값: {lambda_val}")

box_add_df.columns
np.where(box_add_df.columns == 'Y')
box_df = box_add_df.iloc[:,15:]

# box_add_df 데이터셋 : 기존에 있는 변수 + boxcox 변환한 변수 (boxcox_어쩌구) 포함한 데이터셋
# box_df 데이터셋 : boxcox 변환한 변수 (boxcox_어쩌구)만 포함한 데이터셋


# 로그 변환 <- 별로임
# for column in ['X5', 'X9' , 'X14', 'X15']:
#    box_add_df[f'log_{column}'] = np.log(df2[column])

# box_add_df.columns
# np.where(box_add_df.columns == 'Y')
# box_df = box_add_df.iloc[:,15:]
# box_df.columns



# 히스토그램 그리기
columns_to_plot = [col for col in box_df.columns[1:]]

plt.figure(figsize=(20, 12))

for i, column in enumerate(columns_to_plot, 1):
    plt.subplot(5, 4, i)  # 5x3 그리드로 서브플롯 만들기
    sns.histplot(box_df[column], bins=100, kde=True)  # 히스토그램과 커널 밀도 추정
    plt.title(f'Distribution of {column}')

plt.tight_layout()

# box_df2 = box_df.iloc[:,:-4]



# X7, X10, X16, X17





# 정규화
from sklearn.preprocessing import StandardScaler

# Z-스코어 표준화
scaler = StandardScaler()
box_std_df = pd.DataFrame(scaler.fit_transform(box_df.iloc[:,1:]), columns=box_df.columns[1:])

box_df['Y'].dtype
box_std_df = pd.concat([box_std_df, box_df['Y']], axis=1)
box_std_df['Y'].dtype


# 히스토그램 그리기
columns_to_plot = [col for col in box_std_df.columns[1:]]

plt.figure(figsize=(20, 12))

for i, column in enumerate(columns_to_plot, 1):
    plt.subplot(5, 3, i)  # 5x3 그리드로 서브플롯 만들기
    sns.histplot(box_std_df[column], bins=100, kde=True)  # 히스토그램과 커널 밀도 추정
    plt.title(f'Distribution of {column}')

plt.tight_layout()

min(box_std_df['boxcox_X7']), max(box_std_df['boxcox_X7'])


box_std_df.columns

# 여기서 X7, X10 , X16 , X17



# boxcox 변환 전에는 X7, X9, X10, X14, X16, X17 가 다중공선성이 높아 보임.
# boxcox 변환 후에는 'boxcox_X2', 'boxcox_X9', 'boxcox_X11', 'boxcox_X12', 'boxcox_X14' 가 다중공선성이 높아 보임.
# boxcox 변환 후 표준화 후에는 ['boxcox_X2' 'boxcox_X6' 'boxcox_X7' 'boxcox_X9' 'boxcox_X10' 'boxcox_X11' ,'boxcox_X14' 'boxcox_X16' 'boxcox_X17']

from statsmodels.stats.outliers_influence import variance_inflation_factor

# 다중공선성 확인을 위해 VIF(분산 팽창 요인) 계산
# 타겟 변수 'Y'는 제외하고 계산
vif_X = box_std_df.drop(columns=['Y'])

# VIF를 계산하여 데이터프레임으로 저장
vif_data = pd.DataFrame()
vif_data['변수'] = vif_X.columns
vif_data['VIF'] = [variance_inflation_factor(vif_X.values, i) for i in range(vif_X.shape[1])]

# VIF 값이 10 이상인 변수 확인
high_vif_features = vif_data[vif_data['VIF'] > 10]['변수']

print("VIF가 높은 변수들:", high_vif_features.values) 


box_std_df['Y'].dtype


vif_df = box_std_df.drop(columns =['boxcox_X2' ,'boxcox_X6', 'boxcox_X7', 'boxcox_X9' ,'boxcox_X10' ,'boxcox_X11' ,'boxcox_X14' ,'boxcox_X16', 'boxcox_X17'])
vif_df.columns

vif_df1 = box_std_df.drop(columns =['boxcox_X2', 'boxcox_X9', 'boxcox_X11', 'boxcox_X12', 'boxcox_X14'])
vif_df1.columns











# X y 나누기
y = box_std_df['Y']
X = box_std_df.drop(columns = ('Y'))

y = vif_df['Y']
X = vif_df.drop(columns = ('Y'))

y = vif_df1['Y']
X = vif_df1.drop(columns = ('Y'))

y = df2['Y']
X = df2.drop(columns = ('Y'))


# 전처리 끝

# 데이터셋을 학습용과 테스트용으로 분리 (90% 학습, 10% 테스트)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42, stratify=y)

# 로지스틱 모델 초기화
lg_model = LogisticRegression(max_iter=1000, random_state=42)

# 교차 검증을 통해 성능 평가 (5-fold)
cv_scores = cross_val_score(lg_model, X_train, y_train, cv=5, scoring='accuracy')
print("Cross-Validation Accuracy Scores:", cv_scores)
print("Mean Cross-Validation Accuracy: {:.4f}".format(cv_scores.mean()))

# 필요없는 변수만 제거한거
# Cross-Validation Accuracy Scores: [0.98900485 0.98896268 0.98918406 0.98834071 0.9884883 ]
# Mean Cross-Validation Accuracy: 0.9888

# 전처리 정규화까지 했는데 다중공선성은 제거 안 한거
# Cross-Validation Accuracy Scores: [0.99764917 0.99777567 0.99788109 0.99787055 0.99808138]
# Mean Cross-Validation Accuracy: 0.9979

# 전처리는 정규화까지 했는데 다중공선성은 정규화 후 기준으로 판단한 변수로 제거 한거
# Cross-Validation Accuracy Scores: [0.99263125 0.99305292 0.99300021 0.99263125 0.99301075]
# Mean Cross-Validation Accuracy: 0.9929

# 전처리는 정규화까지 했는데 다중공선성은 정규화 전 기준으로 판단한 변수로 제거한 거
# Cross-Validation Accuracy Scores: [0.99775459 0.99803922 0.99792326 0.99798651 0.99817626]
# Mean Cross-Validation Accuracy: 0.9980


# 최종 모델 학습
lg_model.fit(X_train, y_train)

# 테스트 세트에 대한 예측 수행
y_pred = lg_model.predict(X_test)
# 성능 평가
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

# 필요없는 변수만 제거한거
# Accuracy: 0.9928
# Precision: 0.9978
# Recall: 0.9353
# F1 Score: 0.9655
# ROC AUC: 0.9675


# 전처리 정규화까지 했는데 다중공선성은 제거 안 한거
# Accuracy: 0.9978
# Precision: 0.9935
# Recall: 0.9861
# F1 Score: 0.9898
# ROC AUC: 0.9927


# 전처리는 정규화까지 했는데 다중공선성은 정규화 후 기준으로 판단한 변수로 제거 한거
# Accuracy: 0.9930
# Precision: 1.0000
# Recall: 0.9351
# F1 Score: 0.9665
# ROC AUC: 0.9675


# 전처리는 정규화까지 했는데 다중공선성은 정규화 전 기준으로 판단한 변수로 제거한 거
# Accuracy: 0.9980
# Precision: 0.9998
# Recall: 0.9821
# F1 Score: 0.9909
# ROC AUC: 0.9910


# -------------------------------------
# 의사결정 나무


# X y 나누기
y = vif_df1['Y']
X = vif_df1.drop(columns = ('Y'))


y = vif_df['Y']
X = vif_df.drop(columns = ('Y'))

y = df2['Y']
X = df2.drop(columns = ('Y'))



# 데이터셋을 학습용과 테스트용으로 분리 (90% 학습, 10% 테스트)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42, stratify=y)

# Decision Tree 모델 초기화
dt_model = DecisionTreeClassifier(random_state=42)

# 교차 검증을 통해 성능 평가 (5-fold)
cv_scores = cross_val_score(dt_model, X_train, y_train, cv=5, scoring='accuracy')
print("Cross-Validation Accuracy Scores:", cv_scores)
print("Mean Cross-Validation Accuracy: {:.4f}".format(cv_scores.mean()))

# 전처리는 정규화까지 했는데 다중공선성은 정규화 전 기준으로 판단한 변수로 제거한 거
# Cross-Validation Accuracy Scores: [0.99917774 0.99917774 0.99946237 0.99918828 0.99925153]
# Mean Cross-Validation Accuracy: 0.9993

# 전처리는 정규화까지 했는데 다중공선성은 정규화 후 기준으로 판단한 변수로 제거 한거
# Cross-Validation Accuracy Scores: [0.99860848 0.99857685 0.99888256 0.99859793 0.99865064]
# Mean Cross-Validation Accuracy: 0.9987


# 최종 모델 학습
dt_model.fit(X_train, y_train)

# 테스트 세트에 대한 예측 수행
y_pred = dt_model.predict(X_test)
# 성능 평가
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

# 전처리는 정규화까지 했는데 다중공선성은 정규화 전 기준으로 판단한 변수로 제거한 거
# Accuracy: 0.9992
# Precision: 0.9967
# Recall: 0.9963
# F1 Score: 0.9965
# ROC AUC: 0.9980


# 전처리는 정규화까지 했는데 다중공선성은 정규화 후 기준으로 판단한 변수로 제거 한거
# Accuracy: 0.9988
# Precision: 0.9951
# Recall: 0.9942
# F1 Score: 0.9946
# ROC AUC: 0.9968


# 트리 시각화 (깊이를 제한하여 보기 쉽게)
plt.figure(figsize=(16, 10))
plot_tree(dt_model, feature_names=X_train.columns, class_names=['0', '1'], filled=True, rounded=True, fontsize=10)
plt.title('Simplified Decision Tree for Defect Classification (max_depth=3)')
plt.show()


# Decision Tree 모델 학습 (최대 깊이 3으로 제한)
dt_model_m = DecisionTreeClassifier(random_state=42, max_depth=3)
dt_model_m.fit(X_train, y_train)

# 트리 시각화 (깊이를 제한하여 보기 쉽게)
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.figure(figsize=(16, 10))
plot_tree(dt_model_m, feature_names=X_train.columns, class_names=['양품', '불량'], filled=True, rounded=True, fontsize=10)
plt.title('Simplified Decision Tree for Defect Classification (max_depth=3)')
plt.show()


# 변수 중요도 추출
importances = dt_model.feature_importances_
indices = np.argsort(importances)[::-1]

# 중요도 시각화
plt.figure(figsize=(10, 6))
sns.barplot(y=importances[indices], x=X.columns[indices])
plt.title('Feature Importance (Decision Tree)')
plt.xlabel('Importance')
plt.ylabel('Features')

for i in range(3):
    plt.text(x=i, y=importances[indices[i]], s=f'{importances[indices[i]]:.2f}', ha='center', va='bottom', color='red', fontsize=12)


plt.xlim([0,0.75])
plot = sns.scatterplot(data = df2 , x='X3' , y='X15', hue='Y', alpha=0.3)
sns.lineplot(x=[-1,0.406], y=[0.027,0.027], color='black')
plt.axvline(x=0.406, color='black')
handles, labels = plot.get_legend_handles_labels()
new_labels = ['양품', '불량']  # 원하는 레이블로 변경
plt.legend(handles, new_labels, title='Y')


plt.xlim([0,0.75])
plot = sns.scatterplot(data = df2 , x='X3' , y='X17', hue='Y', alpha=0.3)
sns.lineplot(x=[0.406,0.75], y=[0.564,0.564], color='black')
plt.axvline(x=0.406, color='black')
handles, labels = plot.get_legend_handles_labels()
new_labels = ['양품', '불량']  # 원하는 레이블로 변경
plt.legend(handles, new_labels, title='Y')



# ---------------------------
# xgboost

# 필요한 데이터 불러오기
df['Y'] = df['Y'].astype('boolean')

# X4, X13, X18, X19, X20을 제거
df2 = df.drop(columns=(['X4', 'X13', 'X18', 'X19', 'X20']))

# X와 y 정의 (X는 독립 변수, y는 종속 변수)
X = df2.drop('Y', axis=1)
y = df2['Y']

# 데이터셋을 학습용과 테스트용으로 분리 (90% 학습, 10% 테스트)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42, stratify=y)




# Decision Tree 모델 학습 (최대 깊이 3으로 제한)
dt_model = XGBClassifier(random_state=42)
dt_model.fit(X_train, y_train)







# 시각화
def hist(df, numeric_col):
	plt.clf()
	plt.rcParams['font.family'] = 'Malgun Gothic'
	plt.rcParams['axes.unicode_minus'] = False
	a = numeric_col + "의 분포"
	plt.title(a)
	sns.histplot(df[numeric_col], stat='density')
	plt.tight_layout()
	plt.show()
	
def rel_nx_ny(df, numeric_col, y):
	plt.clf()
	plt.rcParams['font.family'] = 'Malgun Gothic'
	plt.rcParams['axes.unicode_minus'] = False
	a = numeric_col + "과 " + y +"컬럼의 관계"
	plt.title(a)
	sns.scatterplot(data=df, x=numeric_col, y=y)
	plt.tight_layout()
	plt.show()
      
# X3 알아보기
hist(df, 'X3')