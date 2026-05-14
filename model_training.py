import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from catboost import CatBoostClassifier

from data_preprocessing import preprocess_data


df = pd.read_csv(
    r'D:\ml\Predicting_F1_pits\playground-series-s6e5\train.csv'
)

x = df.drop('PitNextLap', axis=1)

y = df['PitNextLap']


x = preprocess_data(x)


x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


cat_features = [
    'Driver',
    'Compound',
    'Race'
]


model = CatBoostClassifier(

    iterations=5000,

    learning_rate=0.02,

    depth=10,

    l2_leaf_reg=5,

    bagging_temperature=1,

    random_strength=1,

    loss_function='Logloss',

    eval_metric='AUC',

    random_seed=42,

    early_stopping_rounds=300,

    verbose=200
)


model.fit(
    x_train,
    y_train,

    cat_features=cat_features,

    eval_set=(x_test, y_test),

    use_best_model=True
)


y_pred_proba = model.predict_proba(x_test)[:, 1]

auc_score = roc_auc_score(
    y_test,
    y_pred_proba
)

print(f"AUC Score : {auc_score:.6f}")


joblib.dump(
    model,
    'catboost_model.pkl'
)

print("Model Saved Successfully")