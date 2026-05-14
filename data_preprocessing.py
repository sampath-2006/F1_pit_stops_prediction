import pandas as pd
import numpy as np


def preprocess_data(x):

    x = x.copy()

    if 'id' in x.columns:
        x.drop('id', axis=1, inplace=True)

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

    x['StintProgress'] = (
        x['LapNumber'] / (x['Stint'] + 1)
    )

    x['TyreLifeSquared'] = (
        x['TyreLife'] ** 2
    )

    x['LapNumberSquared'] = (
        x['LapNumber'] ** 2
    )

    x['DegradationPerLap'] = (
        x['Cumulative_Degradation'] /
        (x['LapNumber'] + 1)
    )

    x['PositionTimesTyre'] = (
        x['Position'] * x['TyreLife']
    )

    x.replace([np.inf, -np.inf], np.nan, inplace=True)

    x.fillna(0, inplace=True)

    return x