# model_training.py
!pip install catboost

import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

from catboost import CatBoostClassifier

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

    'RacePhase',

    'Driver_Compound',
    'Race_Compound'
]


kf = StratifiedKFold(

    n_splits=5,

    shuffle=True,

    random_state=SEED
)


oof_preds = np.zeros(len(x))


for fold, (train_idx, val_idx) in enumerate(kf.split(x, y)):

    print(f'\nFold {fold + 1}')


    x_train = x.iloc[train_idx]

    x_val = x.iloc[val_idx]


    y_train = y.iloc[train_idx]

    y_val = y.iloc[val_idx]


    model = CatBoostClassifier(

        task_type='GPU',

        devices='0',

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


    model.fit(

        x_train,

        y_train,

        cat_features=cat_features,

        eval_set=(x_val, y_val),

        use_best_model=True
    )


    val_preds = model.predict_proba(x_val)[:, 1]

    oof_preds[val_idx] = val_preds


    joblib.dump(
        model,
        f'catboost_fold_{fold}.pkl'
    )


auc = roc_auc_score(y, oof_preds)

print(f'\nFinal CV AUC : {auc:.6f}')

print('\nAll Fold Models Saved Successfully')