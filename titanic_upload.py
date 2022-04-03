"""
kaggle
Titanic - Machine Learning from Disaster
ロジスティック回帰:0.756
ランダムフォレスト:0.739
ロジスティック回帰＋ランダムフォレストのアンサンブル：0.766
順位:10605位
"""
import os
import glob
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

pd.set_option('display.max_columns',13)

# ディレクトリ変更
print(os.getcwd())
path_desktop = r""
os.chdir(path_desktop)  # excelファイルがある上位ディレクトリに変更
print(os.getcwd())

# データのロード
path_train = "train.csv"
df_train = pd.read_csv(path_train)
path_test = "test.csv"
df_test = pd.read_csv(path_test)
target_column = "Survived"  # 目的変数
random_seed = 1234        # 乱数固定用

# データの情報
print(df_train.head(5))
print(df_train.shape)
print(df_train.info())
# 数値変数の分布
print(df_train.describe())

# 欠損値の確認
print(df_train.isnull().any())
print(df_test.isnull().any())
print(df_train.isnull().sum())
print(df_test.isnull().sum())

# 欠損値処理
# 年齢を中央値で置換する
def missing_value_age(df):
    # 欠損値フラグ
    df["Age_na"] = df["Age"].isnull().astype(np.int64)
    # 欠損値を中央値で埋める
    df["Age"].fillna(df["Age"].median(), inplace=True)
missing_value_age(df_train)  # trainデータ
missing_value_age(df_test)    # testデータ

# 欠損値の数が少ないので最頻のカテゴリー/中央値で置換する
df_train["Embarked"].fillna("S", inplace=True)
df_test["Fare"].fillna(df_test['Fare'].median(), inplace=True)

# cabin
#欠損値のフラグ
def missing_value_cabin(df):
    # 欠損値フラグ
    df["Cabin_na"] = df["Cabin"].isnull().astype(np.int64)
    # 欠損値を"Unkown"で埋める
    df["Cabin"].fillna("Unkown", inplace=True)
missing_value_cabin(df_train)
missing_value_cabin(df_test)

# 標準化
def normalization(df, name):
    df[name+"_norm"] = (df[name] - df[name].mean()) / df[name].std()
normalization(df_train, 'Age')
normalization(df_test, 'Age')
normalization(df_train, 'Fare')
normalization(df_test, 'Fare')

# 新しい特徴量生成
# 仮説:家族の数が多いほど亡くなる確率が高くなる
def featuregeneration(df):
    df['FamilySize'] = df['Parch'] + df['SibSp'] + 1
featuregeneration(df_train)
featuregeneration(df_test)
normalization(df_train, 'FamilySize')
normalization(df_test, 'FamilySize')
"""
図示してみる
df_train['FamilySize'] = df_train['FamilySize'][:len(df_train)]
sns.countplot(x='FamilySize', data=df_train, hue='Survived')
plt.show()
"""
#仮説2:一人であることも可能性が高くなりそう
def featuregeneration1(df):
    df['FamilySize'] = df['Parch'] + df['SibSp'] + 1
    df['IsAlone'] = 0
    df.loc[df['FamilySize'] == 1, 'IsAlone'] = 1
featuregeneration1(df_train)
featuregeneration1(df_test)

# ダミー化
def dummy(df):
    df = pd.get_dummies(df, columns=[
        "Pclass",
        "Sex",
        "Embarked",
    ])
    return df
df_train = dummy(df_train)
df_test = dummy(df_test)

# データの情報 再確認
print(df_train.head(5))

# -----------------------------------
# モデル作成 ロジスティック回帰
# -----------------------------------
# すでに意味がないコラムは使わない（名前など）
print(list(df_train.columns))
select_columns_clf = [
    #'PassengerId',
    # 'Survived',
    # 'Name',
    #'Age',
    'Age_na',
    'Age_norm',
    #'SibSp',
    # 'Parch',
    # 'Ticket',
    #'Fare',
    'Fare_norm',
    #'Cabin',
    'Cabin_na',
    #'FamilySize',
    'FamilySize_norm',
    'IsAlone',
    "Pclass_1",
    "Pclass_2",
    #"Pclass_3",  # dummy除外
    "Sex_male",
    #"Sex_female",  # dummy除外
    "Embarked_C",
    "Embarked_Q",
    #"Embarked_S",  # dummy除外
]

# 目的変数を落とす
X = df_train[select_columns_clf]
print(f"X shape:{X.shape}")
Y = df_train['Survived']
print(f"Y shape:{Y.shape}")
print(f"test shape:{df_test.shape}")
# 分割する
X_train, X_validation, y_train, y_validation = train_test_split(X, Y, test_size = 0.2,
                                                                random_state=1,
                                                                stratify=Y)

"""
# グリッドサーチ
seed = 0
params = {"penalty":["l2"],
          "C": [10 ** i for i in range(-3, 3)],
          "random_state":[seed]
          }
clf = LogisticRegression()
# 学習する
gscv = GridSearchCV(clf, param_grid=params, cv=3, scoring="roc_auc", verbose=1)
gscv.fit(X_train, y_train)
print(gscv.best_params_)
{'C': 1, 'penalty': 'l2', 'random_state': 0}
print(gscv.best_score_)
0.8631366691892272
"""
clf = LogisticRegression(penalty='l2', C=1, random_state=0,max_iter=300)
clf.fit(X_train,y_train)
# 予測する
y_model_clf = clf.predict(X_validation)
y_model_clf_proba = clf.predict_proba(X_validation)[:, 1]
# 評価
print(f"accuracy_score_rf:{accuracy_score(y_validation, y_model_clf)}")
# 交差検証
scores_clf = cross_val_score(clf, X, Y,cv=5)
print(f"score_rf:{scores_clf}")
print(f"score_rf mean:{scores_clf.mean()}")
print(metrics.classification_report(y_validation, y_model_clf))
print(confusion_matrix(y_validation, y_model_clf))
#予測（出力する、提出用の予測)
test_pred_clf = clf.predict(df_test[select_columns_clf])
sub_clf = pd.DataFrame(df_test['PassengerId'])
sub_clf['Survived'] = list(map(int, test_pred_clf))
print(sub_clf)
# 出力
sub_clf.to_csv('submission_clf.csv', index=False)

# -----------------------------------
# モデル作成 ランダムフォレスト
# -----------------------------------
select_columns_rf = [
    #'PassengerId',
    #'Survived',
    #'Name',
    'Age',
    'Age_na',
    #'Age_norm',
    'SibSp',
    'Parch',
    #'Ticket',
    'Fare',
    #'Fare_norm',
    #'Cabin',
    'Cabin_na',
    'FamilySize',
    #'FamilySize_norm',
    'IsAlone',
    "Pclass_1",
    "Pclass_2",
    #"Pclass_3",  # dummy除外
    "Sex_male",
    #"Sex_female",  # dummy除外
    "Embarked_C",
    "Embarked_Q",
    #"Embarked_S",  # dummy除外
]

# 目的変数を落とす
X = df_train[select_columns_rf]
print(f"X shape:{X.shape}")
Y = df_train['Survived']
print(f"Y shape:{Y.shape}")
print(f"test shape:{df_test.shape}")
# 分割する
X_train, X_validation, y_train, y_validation = train_test_split(X, Y, test_size = 0.2,
                                                                random_state=1,
                                                                stratify=Y)

"""
# グリッドサーチ
params = {"n_estimators": [750],
          "max_depth": [125,150],
          "max_features": ["sqrt"]
          }
model_rf = RandomForestClassifier()
gscv = GridSearchCV(model_rf, param_grid=params, cv=3, scoring="roc_auc", verbose=1)
gscv.fit(X_train, y_train)
print(gscv.best_params_)
print(gscv.best_score_)
Fitting 3 folds for each of 2 candidates, totalling 6 fits
{'max_depth': 150, 'max_features': 'sqrt', 'n_estimators': 750}
0.8635956113591003
"""
# 学習する
model_rf = RandomForestClassifier(n_estimators=750, max_depth=150,
                                  max_features='sqrt', random_state=0)
model_rf.fit(X_train,y_train)
# 予測する
y_model_rf = model_rf.predict(X_validation)
y_model_rf_proba = model_rf.predict_proba(X_validation)[:, 1]
# 評価
print(f"accuracy_score_rf:{accuracy_score(y_validation, y_model_rf)}")
# 交差検証
scores_rf = cross_val_score(model_rf, X, Y,cv=5)
print(f"score_rf:{scores_rf}")
print(f"score_rf mean:{scores_rf.mean()}")
print(metrics.classification_report(y_validation, y_model_rf))
print(confusion_matrix(y_validation, y_model_rf))
#予測（出力する、提出用の予測)
test_pred_rf = model_rf.predict(df_test[select_columns_rf])
sub_rf = pd.DataFrame(df_test['PassengerId'])
sub_rf['Survived'] = list(map(int, test_pred_rf))
print(sub_rf)
# 出力
sub_rf.to_csv('submission_rf.csv', index=False)

# -----------------------------------
# アンサンブル　Ensemble　ランダムフォレストとxgboost
# -----------------------------------
# 予測値の加重平均をとる
#重み
w_clf = 0.3
w_rf = 1 - w_clf
threshold = 0.5
#0.7
"""
# 以下のループを回して適切な重みを確認
for i in np.arange(0,1.0,0.1):
    w_rf  = i
    w_clf = 1 - i
    pred_proba_clf_rf = y_model_clf_proba * w_clf + y_model_rf_proba * w_rf
    ACS = []
    for j in np.arange(0,1.0,0.1):
        threshold = j
        pred_label_Ensemble = np.where(pred_proba_clf_rf > threshold , 1, 0)
        ACS.append(accuracy_score(y_validation, pred_label_Ensemble))
    print(ACS)
    for k in range(1,10,1):
        if ACS[k-1] < ACS[k]:
            num = k
            best_score = ACS[k]
        else:
            num = k-1
            best_score = ACS[k-1]
    print(f"w_rf:{w_rf},w_clf:{w_clf},num:{num},best_score:{best_score}")
    # ↑一部不具合。numのところがちょっとおかしいが直接確認して確認
"""
pred_proba_clf_rf = y_model_clf_proba * w_clf + y_model_rf_proba * w_rf
pred_label_Ensemble = np.where(pred_proba_clf_rf > threshold , 1, 0)
print(f"accuracy_score_clf_rf:{accuracy_score(y_validation, pred_label_Ensemble)}")
print(metrics.classification_report(y_validation, pred_label_Ensemble))
print(confusion_matrix(y_validation, pred_label_Ensemble))
#予測
test_pred_clf = clf.predict(df_test[select_columns_clf])
test_pred_rf = model_rf.predict(df_test[select_columns_rf])
test_pred_clf_rf = test_pred_clf * w_clf + test_pred_rf * w_rf
test_pred_label_ensemble_test = np.where(test_pred_clf_rf > threshold, 1, 0)
# 提出ファイルの作成
sub_clf_rf = pd.DataFrame(df_test['PassengerId'])
sub_clf_rf['Survived'] = list(map(int, test_pred_label_ensemble_test))
print(sub_clf_rf)
sub_clf_rf.to_csv('submission_clf_rf.csv', index=False)

