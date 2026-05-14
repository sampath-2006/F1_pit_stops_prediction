import pandas as pd
import joblib

from data_preprocessing_2 import preprocess_data


model = joblib.load(
    'catboost_model.pkl'
)

df_test = pd.read_csv(
    r'D:\ml\Predicting_F1_pits\playground-series-s6e5\test.csv'
)

test_ids = df_test['id']

x = preprocess_data(
    df_test,
    is_training=False
)

predictions = model.predict_proba(x)[:, 1]

submission_2 = pd.DataFrame({
    'id': test_ids,
    'PitNextLap': predictions
})

submission_2.to_csv(
    'submission2.csv',
    index=False
)

print(submission_2.head())

print(submission_2.shape)