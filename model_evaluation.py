import pandas as pd
import joblib

from data_preprocessing import preprocess_data


model = joblib.load(
    'catboost_model.pkl'
)


df_test = pd.read_csv(
    r'D:\ml\Predicting_F1_pits\playground-series-s6e5\test.csv'
)


test_ids = df_test['id']


x = preprocess_data(df_test)


predictions = model.predict_proba(x)[:, 1]


submission = pd.DataFrame({

    'id': test_ids,

    'PitNextLap': predictions

})


submission.to_csv(
    'submission_final.csv',
    index=False
)


print(submission.head())

print(submission.shape)

print("Submission File Created Successfully")