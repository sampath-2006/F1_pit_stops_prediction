# model_evaluation.py

import pandas as pd
import numpy as np
import joblib

from data_preprocessing import preprocess_data


df_test = pd.read_csv(
    r'D:\ml\Predicting_F1_pits\playground-series-s6e5\test.csv'
)


test_ids = df_test['id']


x = preprocess_data(df_test)


final_preds = np.zeros(len(x))


for fold in range(5):

    print(f'Loading Fold {fold + 1} Model')


    model = joblib.load(
        f'catboost_fold_{fold}.pkl'
    )


    fold_preds = model.predict_proba(x)[:, 1]


    final_preds += fold_preds / 5


submission = pd.DataFrame({

    'id': test_ids,

    'PitNextLap': final_preds
})


submission.to_csv(

    'submission_fold_avg.csv',

    index=False
)


print(submission.head())

print(submission.shape)

print('\nFold Averaged Submission Created Successfully')