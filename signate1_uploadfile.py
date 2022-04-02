"""
社内研修の一環で行われたSIGNATEのデータサイエンティストコンテスト
採血データを用いて心不全かどうかを見分けるモデルを作成する
1：心不全を発症する
0：心不全を発症しない
方針：いくつか分類ベースのモデルを作成した後に結果がよいものについてアンサンブルする
順位:参加者に対して半分程度
反省:
・特徴量生成に時間をかけたかった。(特徴量のエンジニアリングはほぼしていない。時間がなかった)
・アンサンブルは性質が違うモデルで行った方が良いことは今回学べた。
"""

import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
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
df_train = pd.read_csv(path_train, index_col=1)
path_test = "test.csv"
df_test = pd.read_csv(path_test, index_col=1)

# データの情報
print(df_train.head(5))
print(df_train.shape)
print(df_train.info())
# 数値変数の分布
print(df_train.describe())

# 変数の箱ひげ図を表示
plt.figure(figsize=(100, 100))
for i in np.arange(df_data.shape[1]):
    plt.subplot(4, 4, i+1)
    plt.boxplot(df_train.dropna().iloc[:, i])
    plt.title(df_train.columns[i])
plt.show()

# 変数のヒストグラムを描く
plt.figure(figsize=(200, 200))
for i in np.arange(df_data.shape[1]):
    plt.subplot(2, 7, i+1)
    plt.hist(df_train.dropna().iloc[:, i])
    plt.title(df_train.columns[i])
plt.show()

# 目的変数を落とす
X = df_train.drop('target', axis=1)
print(f"X shape:{X.shape}")
Y = df_train['target']
print(f"Y shape:{Y.shape}")
print(f"test shape:{df_test.shape}")

# 分割する
X_train, X_validation, y_train, y_validation = train_test_split(X, Y, test_size = 0.2, random_state=1)

# -----------------------------------
# モデル作成 ベイズモデル
# -----------------------------------
# 学習する
model = GaussianNB()
model.fit(X_train,y_train)
y_model = model.predict(X_validation)
# 評価
print(f"accuracy_score:{accuracy_score(y_validation, y_model)}")
# 交差検証
scores = cross_val_score(model, X, Y, cv=5)
print(f"score:{scores}")
print(f"score mean:{scores.mean()}")
print(metrics.classification_report(y_model, y_validation))
#予測（出力する、提出用の予測)
test_pred = model.predict(df_test)
df_colum_test_id = pd.DataFrame(df_test['id'])
df_pred = pd.DataFrame(test_pred)
print(pd.concat([df_colum_test_id, df_pred], axis=1))
# 中身の確認
print(df_pred)
# 出力
df_pred.to_csv("pred_GaussianNB.csv",index=False,header=None)

# -----------------------------------
# モデル作成 ランダムフォレスト
# -----------------------------------
"""
# グリッドサーチ
params = {"n_estimators": [750,1000,1250,1500],
          "max_depth": [50],
          "max_features": ["sqrt"]
          }
gscv = GridSearchCV(model_rf, param_grid=params, cv=5, scoring="roc_auc", verbose=1)
gscv.fit(X_train, y_train)
print(gscv.best_params_)
print(gscv.best_score_)
{'max_depth': 50, 'max_features': 'sqrt', 'n_estimators': 500}
0.8285612025769507
{'max_depth': 50, 'max_features': 'sqrt', 'n_estimators': 1000}
0.8314722023383441
params = {"n_estimators": [750,1000,1250,1500],
          "max_depth": [50],
          "max_features": ["sqrt"]
          }
{'max_depth': 50, 'max_features': 'sqrt', 'n_estimators': 1250}
0.8331901694106417
"""
# 学習する
model_rf = RandomForestClassifier(n_estimators=1250, max_depth=50,
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
test_pred_rf = model_rf.predict(df_test)
df_colum_test_id = pd.DataFrame(df_test['id'])
df_pred_rf = pd.DataFrame(test_pred_rf)
print(pd.concat([df_colum_test_id, df_pred_rf], axis=1))
# 中身の確認
print(df_pred_rf)
print(df_test['id'])
print(df_pred_rf.shape)
print(df_test['id'].shape)
# 出力
df_pred_rf.to_csv("pred_RandomForestClassifier.csv",index=False,header=None)

# -----------------------------------
# モデル作成 xgboost
# -----------------------------------
"""
# グリッドサーチ
cv_params = {'metric': ['error'],
             'objective': ['binary:logistic'],
             'n_estimators': [50000],
             'random_state': [seed],
             'booster': ['gbtree'],
             'learning_rate': [0.1],
             'min_child_weight': [1,2,3],
             'max_depth': [10],
             'colsample_bytree': [0.5],
             'subsample': [1.0]
             }
cls = xgb.XGBClassifier()
cls_grid = GridSearchCV(cls, cv_params, cv=5, scoring = 'accuracy')
cls_grid.fit(X_train,
             y_train,
             early_stopping_rounds=50,
             eval_set=[(X_validation, y_validation)],
             eval_metric='error',
             verbose=0)
print(cls_grid.best_params_)
print(cls_grid.best_score_)
pred_2 = cls_grid.best_estimator_.predict(X_validation)
grid_score = accuracy_score(y_validation, pred_2)
print(grid_score)
print(confusion_matrix(y_validation, pred_2))
"""
# 学習する
# モデルの作成および学習データを与えての学習
seed =71
params = {'metric': 'error',
          'objective': 'binary:logistic',
          'n_estimators': 50000,
          'random_state': seed,
          'booster': 'gbtree',
          'learning_rate': 0.05,
          'min_child_weight': 1,
          'max_depth': 5,
          'colsample_bytree': 0.5,
          'subsample': 1
         }
model_xgb = xgb.XGBClassifier()
model_xgb.set_params(**params)
model_xgb.fit(X_train,
              y_train,
              early_stopping_rounds=50,
              eval_set=[(X_validation, y_validation)],
              eval_metric='error',
              verbose=0)
# テストデータの予測値を確率で出力する。0と1になる確率それぞれを返すからスライスで指定
pred_proba_xgb = model_xgb.predict_proba(X_validation)[:, 1]
# テストデータの予測値を二値に変換する
pred_label_xgb = np.where(pred_proba_xgb > 0.5, 1, 0)
# 評価
print(f"accuracy_score_xgb:{accuracy_score(y_validation, pred_label_xgb)}")
# 交差検証
scores_xgb = cross_val_score(model_xgb, X, Y,cv=5)
print(f"score_xgb:{scores_xgb}")
print(f"score_xgb mean:{scores_xgb.mean()}")
print(metrics.classification_report(y_validation, pred_label_xgb))
print(confusion_matrix(y_validation, pred_label_xgb))
#予測
test_pred_xgb = model_xgb.predict(df_test)
df_pred_xgb = pd.DataFrame(test_pred_xgb,columns=['id','target'])
df_pred_xgb['id'] = df_test['id']
# 出力
df_pred_xgb.to_csv("pred_xgb.csv",index=False,header=None)

# -----------------------------------
# アンサンブル　Ensemble　ランダムフォレストとxgboost
# -----------------------------------
# 予測値の加重平均をとる
#重み
w_xgb = 0.3
w_rf = 1 - w_xgb
threshold = 0.4
"""
以下のループを回して適切な重みを確認
for i in np.arange(0,1.0,0.1):
    w_rf  = i
    w_xgb = 1 - i
    pred_proba_xgb_rf = pred_proba_xgb * w_xgb + y_model_rf_proba * w_rf
    ACS = []
    for j in np.arange(0,1.0,0.1):
        threshold = j
        pred_label_Ensemble = np.where(pred_proba_xgb_rf > threshold , 1, 0)
        ACS.append(accuracy_score(y_validation, pred_label_Ensemble))
    print(ACS)
    for k in range(1,10,1):
        if ACS[k-1] < ACS[k]:
            num = k
            best_score = ACS[k]
        else:
            num = k-1
            best_score = ACS[k-1]
    print(f"w_rf:{w_rf},w_xgb:{w_xgb},num:{num},best_score:{best_score}")

"""
pred_proba_xgb_rf = pred_proba_xgb * w_xgb + y_model_rf_proba * w_rf
pred_label_Ensemble = np.where(pred_proba_xgb_rf > threshold , 1, 0)
print(f"accuracy_score_xgb_rf:{accuracy_score(y_validation, pred_label_Ensemble)}")
print(metrics.classification_report(y_validation, pred_label_Ensemble))
print(confusion_matrix(y_validation, pred_label_Ensemble))
#予測
test_pred_proba_xgb = model_xgb.predict_proba(df_test)[:, 1]
test_model_rf_proba = model_rf.predict_proba(df_test)[:, 1]
test_pred_proba_xgb_rf = test_pred_proba_xgb * w_xgb + test_model_rf_proba * w_rf
pred_label_Ensemble_test = np.where(test_pred_proba_xgb_rf > threshold, 1, 0)
df_pred_xgb_rf = pd.DataFrame(pred_label_Ensemble_test,columns=['id','target'])
df_pred_xgb_rf['id'] = df_test['id']
# 出力
df_pred_xgb_rf.to_csv("pred_xgb_rf.csv",index=False,header=None)