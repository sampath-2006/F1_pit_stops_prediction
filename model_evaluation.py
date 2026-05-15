import pandas as pd
import numpy as np
import joblib

from data_preprocessing import preprocess_data


cat_model = joblib.load(
    r'D:\ml\Predicting_F1_pits\F1_pit_stops_prediction\catboost_model_2.pkl'
)

lgb_model = joblib.load(
    r'D:\ml\Predicting_F1_pits\F1_pit_stops_prediction\lightgbm_model_2.pkl'
)

xgb_model = joblib.load(
    r'D:\ml\Predicting_F1_pits\F1_pit_stops_prediction\xgboost_model_2.pkl'
)


cat_features = [
    'Driver',
    'Compound',
    'Race',
    'Driver_Race',
    'Compound_Stint',
    'RacePhase'
]


df_test = pd.read_csv(
    r'D:\ml\Predicting_F1_pits\playground-series-s6e5\test.csv'
)


test_ids = df_test['id']


x = preprocess_data(df_test)


cat_pred = (
    cat_model.predict_proba(x)[:, 1]
)


x_lgb = x.copy()

for col in cat_features:
    x_lgb[col] = (
        x_lgb[col].astype('category')
    )


lgb_pred = (
    lgb_model.predict_proba(x_lgb)[:, 1]
)


x_xgb = x.copy()

for col in cat_features:

    x_xgb[col] = (
        x_xgb[col]
        .astype('category')
        .cat.codes
    )


xgb_pred = (
    xgb_model.predict_proba(x_xgb)[:, 1]
)


final_pred = (
    0.9 * cat_pred +
    0.1 * lgb_pred 
    
)


submission = pd.DataFrame({

    'id': test_ids,

    'PitNextLap': final_pred

})


submission.to_csv(
    'submission_ensemble_cat_0.9_lgb_0.1.csv',
    index=False
)


print(submission.head())

print(submission.shape)

print('Submission File Created Successfully')


