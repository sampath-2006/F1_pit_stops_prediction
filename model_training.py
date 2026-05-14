import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from catboost import CatBoostClassifier

from data_preprocessing_2 import preprocess_data


df = pd.read_csv(
    r'D:\ml\Predicting_F1_pits\playground-series-s6e5\train.csv'
)

x = df.drop('PitNextLap', axis=1)
y = df['PitNextLap']


x = preprocess_data(
    x,
    is_training=True
)


x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


model = CatBoostClassifier(

    iterations=3000,
    learning_rate=0.03,
    depth=8,

    loss_function='Logloss',
    eval_metric='AUC',

    random_seed=42,

    verbose=200
)


model.fit(
    x_train,
    y_train,
    eval_set=(x_test, y_test),
    use_best_model=True
)


y_pred = model.predict(x_test)

accuracy = accuracy_score(
    y_test,
    y_pred
)

print(f"Accuracy : {accuracy:.4f}")


joblib.dump(
    model,
    'catboost_model.pkl'
)

print("Model Saved Successfully")