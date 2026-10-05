import pandas as pd 
import numpy as np 
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import joblib
from model import FEATURE_COLUMNS, WINDOW_SIZE

df = pd.read_csv(
    "AirQuality.csv",
    sep=";",
    decimal=",",
    encoding="latin1"
)

df.dropna(axis=1, how="all", inplace=True)

df["datetime"] = pd.to_datetime(
    df["Date"].astype(str) + " " + df["Time"].astype(str),
    format="%d/%m/%Y %H.%M.%S")
df.dropna(subset=["datetime"], inplace=True)
df.drop(columns=["Date", "Time"], inplace=True)
print(df.head())

df.set_index("datetime", inplace=True)
df.replace(-200, np.nan, inplace=True)
df.interpolate(method="time", inplace=True)
print(df.head())

df["hour"]=df.index.hour
df["month"]=df.index.month

df = df[FEATURE_COLUMNS]
df.dropna(inplace=True)

print(df.head())


scaler=MinMaxScaler()
scaled=scaler.fit_transform(df)
joblib.dump(scaler, "scaler.pkl")

def create_sequences(data, seq_length):
    x=[]
    y=[]
    for i in range(len(data)-seq_length):
        x.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(x), np.array(y)

x, y=create_sequences(scaled, WINDOW_SIZE)
y = y[:, :1]

print("x shape:", x.shape)
print("y shape:", y.shape)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=False)

print("x_train shape:", x_train.shape)
print("x_test shape:", x_test.shape)    
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)

np.save("x_train.npy", x_train)
np.save("x_test.npy", x_test)
np.save("y_train.npy", y_train)
np.save("y_test.npy", y_test)
