# 피어슨 상관계수로 분석했을 때와 스피어만 상관계수로 비선형 관계를 해결했을 때의 성과 비교하기
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report, confusion_matrix, make_scorer


from sklearn.preprocessing import PowerTransformer, MinMaxScaler, StandardScaler, PolynomialFeatures
from statsmodels.stats.outliers_influence import variance_inflation_factor
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

df2 = df.drop(columns = ['X10', 'X14', 'X16', 'X17'])
df2.head()


# [전처리 후 데이터][train, test 분리] - 데이터셋의 90% 학습셋, 10% 테스트셋으로 분리
y = df2['Y']
X = df2.drop(columns = ('Y'))

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

# Accuracy: 0.9913
# Precision: 0.9972
# Recall: 0.9221
# F1 Score: 0.9582
# ROC AUC: 0.9609







# -----------------------------------------
# 원본 데이터로 분석
# [전처리 후 데이터][train, test 분리] - 데이터셋의 90% 학습셋, 10% 테스트셋으로 분리
y = df['Y']
X = df.drop(columns = ('Y'))

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

# Accuracy: 0.9929
# Precision: 0.9978
# Recall: 0.9365
# F1 Score: 0.9662
# ROC AUC: 0.9681


# ----------------------------
# 피어슨과 스피어만 모두 높은거만 제거
# 피어슨 높음, 스피어만 높음 : (X7, X16) (X9, X14) => X14, X16 제거
# 피어슨 높음 : (X7, X10) , (X7,X16), (X7, X17) , (X9, X14) , (X10, X16) , (X10, X17), (X16,X17) ⇒ X10, X14, X16, X17 제거?

df2 = df.drop(columns = ['X14', 'X16'])
df2.head()


# [전처리 후 데이터][train, test 분리] - 데이터셋의 90% 학습셋, 10% 테스트셋으로 분리
y = df2['Y']
X = df2.drop(columns = ('Y'))

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


# Accuracy: 0.9913
# Precision: 0.9972
# Recall: 0.9226
# F1 Score: 0.9584
# ROC AUC: 0.9612