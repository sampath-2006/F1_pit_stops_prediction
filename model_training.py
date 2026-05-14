import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'


import pandas as pd
import joblib
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

from data_preprocessing import preprocess_data


#loading the data 
df = pd.read_csv(r'D:\ml\Predicting_F1_pits\playground-series-s6e5\train.csv')
#splitting the data into features and target variable
x = df.drop('PitNextLap',axis=1)
y = df['PitNextLap']

#preprocessing the data 

x,le,scaler = preprocess_data(x)


#building the model
def build_model(x):

    model = Sequential([

        Dense(128, activation='relu', input_shape=(x.shape[1],)),
        Dropout(0.3),

        Dense(64, activation='relu'),
        Dropout(0.3),

        Dense(32, activation='relu'),

        Dense(1, activation='sigmoid')

    ])

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return model



model = build_model(x)


early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

history = model.fit(
    x,
    y,
    validation_split=0.2,
    epochs=10,
    batch_size=64,
    callbacks=[early_stopping],
    verbose=1
)

model.save("f1_pitstop_ann_model.h5")