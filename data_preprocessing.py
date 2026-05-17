# data_preprocessing.py

import pandas as pd
import numpy as np


def preprocess_data(x):

    x = x.copy()


    if 'id' in x.columns:

        x.drop(
            'id',
            axis=1,
            inplace=True
        )


    x.columns = [

        col.replace(' ', '_')

        for col in x.columns
    ]


    x['TyreWearPerLap'] = (

        x['TyreLife'] /

        (x['LapNumber'] + 1)
    )


    x['Deg_Tyre'] = (

        x['Cumulative_Degradation'] *

        x['TyreLife']
    )


    x['PositionPressure'] = (

        x['Position_Change'] /

        (x['LapNumber'] + 1)
    )


    x['LateRace'] = (

        x['RaceProgress'] > 0.7

    ).astype(int)


    x['LapTime_Per_TyreLife'] = (

        x['LapTime_Delta'] /

        (x['TyreLife'] + 1)
    )


    x['StintProgress'] = (

        x['LapNumber'] /

        (x['Stint'] + 1)
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

        x['Position'] *

        x['TyreLife']
    )


    x['Driver_Race'] = (

        x['Driver'].astype(str)

        + '_'

        + x['Race'].astype(str)
    )


    x['Compound_Stint'] = (

        x['Compound'].astype(str)

        + '_'

        + x['Stint'].astype(str)
    )


    x['Driver_Compound'] = (

        x['Driver'].astype(str)

        + '_'

        + x['Compound'].astype(str)
    )


    x['Race_Compound'] = (

        x['Race'].astype(str)

        + '_'

        + x['Compound'].astype(str)
    )


    x['AggressiveTyreUsage'] = (

        x['TyreLife'] *

        x['LapTime_Delta']
    )


    x['RelativeDeg'] = (

        x['Cumulative_Degradation'] /

        (x['TyreLife'] + 1)
    )


    x['RacePhase'] = pd.cut(

        x['RaceProgress'],

        bins=[0, 0.3, 0.7, 1],

        labels=['Early', 'Mid', 'Late']
    )


    x.replace(
        [np.inf, -np.inf],
        np.nan,
        inplace=True
    )


    for col in x.columns:

        if str(x[col].dtype) == 'category':

            x[col] = (

                x[col]
                .cat
                .add_categories(['Unknown'])
            )

            x[col] = x[col].fillna('Unknown')

        else:

            x[col] = x[col].fillna(0)


    return x