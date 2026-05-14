import pandas as pd
from sklearn.preprocessing import StandardScaler,LabelEncoder
import joblib

df = pd.read_csv(r'D:\ml\Predicting_F1_pits\playground-series-s6e5\train.csv')

print(df.shape)

#splitting the data into features and target variable

x = df.drop('PitNextLap',axis=1)
y = df['PitNextLap']




def preprocess_data(x):
    
    #dropping the columns which are not required for training the model
    x = x.drop(['Driver','Race'],axis = 1)
    
    #label encoding categorical features 
    
    le = LabelEncoder()
    x['Compound'] = le.fit_transform(x['Compound'])
    joblib.dump(le, 'Compound_label_encoder.pkl')

    #scaling the features using standard scaler
    scaler = StandardScaler()
    x = scaler.fit_transform(x)
    joblib.dump(scaler, 'scaler.pkl')
    
    return x,le,scaler