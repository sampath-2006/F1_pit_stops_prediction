import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib


def preprocess_data(x, is_training=True):

    x = x.copy()

    drop_cols = ['id', 'Driver', 'Race']

    existing_drop_cols = [
        col for col in drop_cols
        if col in x.columns
    ]

    x.drop(existing_drop_cols, axis=1, inplace=True)

    categorical_cols = ['Compound']

    numerical_cols = [
        col for col in x.columns
        if col not in categorical_cols
    ]

    x['TyreWearPerLap'] = (
        x['TyreLife'] / (x['LapNumber'] + 1)
    )

    x['Deg_Tyre'] = (
        x['Cumulative_Degradation'] * x['TyreLife']
    )

    x['PositionPressure'] = (
        x['Position_Change'] / (x['LapNumber'] + 1)
    )

    x['LateRace'] = (
        x['RaceProgress'] > 0.7
    ).astype(int)

    x['LapTime_Per_TyreLife'] = (
        x['LapTime_Delta'] / (x['TyreLife'] + 1)
    )

    new_feature_cols = [
        'TyreWearPerLap',
        'Deg_Tyre',
        'PositionPressure',
        'LateRace',
        'LapTime_Per_TyreLife'
    ]

    numerical_cols.extend(new_feature_cols)

    if is_training:

        label_encoders = {}

        for col in categorical_cols:

            le = LabelEncoder()

            x[col] = le.fit_transform(
                x[col].astype(str)
            )

            label_encoders[col] = le

        scaler = StandardScaler()

        x[numerical_cols] = scaler.fit_transform(
            x[numerical_cols]
        )

        joblib.dump(
            label_encoders,
            'label_encoders.pkl'
        )

        joblib.dump(
            scaler,
            'scaler.pkl'
        )

    else:

        label_encoders = joblib.load(
            'label_encoders.pkl'
        )

        scaler = joblib.load(
            'scaler.pkl'
        )

        for col in categorical_cols:

            le = label_encoders[col]

            x[col] = le.transform(
                x[col].astype(str)
            )

        x[numerical_cols] = scaler.transform(
            x[numerical_cols]
        )

    return x