import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

from data_preprocessing import preprocess_data


SEED = 42


df = pd.read_csv(
    r'D:\ml\Predicting_F1_pits\playground-series-s6e5\train.csv'
)

x = df.drop('PitNextLap', axis=1)

y = df['PitNextLap']


x = preprocess_data(x)


cat_features = [
    'Driver',
    'Compound',
    'Race',
    'Driver_Race',
    'Compound_Stint',
    'RacePhase'
]


kf = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=SEED
)


cat_oof = np.zeros(len(x))
lgb_oof = np.zeros(len(x))
xgb_oof = np.zeros(len(x))


for fold, (train_idx, val_idx) in enumerate(kf.split(x, y)):

    print(f'Fold {fold + 1}')

    x_train = x.iloc[train_idx]
    x_val = x.iloc[val_idx]

    y_train = y.iloc[train_idx]
    y_val = y.iloc[val_idx]


    cat_model = CatBoostClassifier(

        iterations=3000,

        learning_rate=0.03,

        depth=8,

        l2_leaf_reg=5,

        bagging_temperature=1,

        random_strength=1,

        loss_function='Logloss',

        eval_metric='AUC',

        random_seed=SEED,

        early_stopping_rounds=300,

        verbose=200
    )


    cat_model.fit(
        x_train,
        y_train,

        cat_features=cat_features,

        eval_set=(x_val, y_val),

        use_best_model=True
    )


    cat_oof[val_idx] = (
        cat_model.predict_proba(x_val)[:, 1]
    )


    x_train_lgb = x_train.copy()
    x_val_lgb = x_val.copy()

    for col in cat_features:
        x_train_lgb[col] = (
            x_train_lgb[col].astype('category')
        )

        x_val_lgb[col] = (
            x_val_lgb[col].astype('category')
        )


    lgb_model = LGBMClassifier(

        n_estimators=3000,

        learning_rate=0.03,

        max_depth=8,

        subsample=0.8,

        colsample_bytree=0.8,

        objective='binary',

        random_state=SEED
    )


    lgb_model.fit(
        x_train_lgb,
        y_train
    )


    lgb_oof[val_idx] = (
        lgb_model.predict_proba(x_val_lgb)[:, 1]
    )


    x_train_xgb = x_train.copy()
    x_val_xgb = x_val.copy()


    for col in cat_features:

        x_train_xgb[col] = (
            x_train_xgb[col]
            .astype('category')
            .cat.codes
        )

        x_val_xgb[col] = (
            x_val_xgb[col]
            .astype('category')
            .cat.codes
        )


    xgb_model = XGBClassifier(

        n_estimators=3000,

        learning_rate=0.03,

        max_depth=8,

        subsample=0.8,

        colsample_bytree=0.8,

        eval_metric='auc',

        random_state=SEED
    )


    xgb_model.fit(
        x_train_xgb,
        y_train
    )


    xgb_oof[val_idx] = (
        xgb_model.predict_proba(x_val_xgb)[:, 1]
    )


final_oof = (
    0.5 * cat_oof +
    0.3 * lgb_oof +
    0.2 * xgb_oof
)


auc_score = roc_auc_score(
    y,
    final_oof
)


print(f'Final CV AUC : {auc_score:.6f}')


joblib.dump(cat_model, 'catboost_model.pkl')
joblib.dump(lgb_model, 'lightgbm_model.pkl')
joblib.dump(xgb_model, 'xgboost_model.pkl')

print('Models Saved Successfully')