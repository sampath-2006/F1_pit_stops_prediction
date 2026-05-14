import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import joblib
import pandas as pd
import tensorflow as tf

from tensorflow.keras.models import load_model


model = load_model(
    r"D:\ml\Predicting_F1_pits\F1_pit_stops_prediction\f1_pitstop_ann_model.h5"
)

scaler = joblib.load("scaler.pkl")
label_encoders = joblib.load("Compound_label_encoder.pkl")

df_test = pd.read_csv(
    r'D:\ml\Predicting_F1_pits\playground-series-s6e5\test.csv'
)

print(df_test.shape)

test_ids = df_test['id']

x = df_test


x = x.drop(['Driver', 'Race'], axis=1)

categorical_cols = x.select_dtypes(include=['object']).columns


x['Compound'] = label_encoders.transform(x['Compound'].astype(str))



x = scaler.transform(x)

y_pred = model.predict(x)

y_pred = (y_pred > 0.5).astype(int).flatten()

test_ids = df_test['id']

submission = pd.DataFrame({
    'id': test_ids,
    'PitNextLap': y_pred
})

submission.to_csv("submission.csv", index=False)

print(submission.head())

print(submission['PitNextLap'].value_counts())