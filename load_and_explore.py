import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv(
    "AirQuality.csv",
    sep=";",
    decimal=",",
    encoding="latin1"
)

print(df.head())

# Tamamen boş sütunları sil
df.dropna(axis=1, how="all", inplace=True)

# Sütun isimlerindeki boşlukları temizle
df.columns = df.columns.str.strip()

print(df.head())

# Tarih ve saati birleştir
df["datetime"] = pd.to_datetime(
    df["Date"].astype(str) + " " + df["Time"].astype(str),
    format="%d/%m/%Y %H.%M.%S",
    errors="coerce"
)

# Geçersiz datetime satırlarını sil
df.dropna(subset=["datetime"], inplace=True)

print(df.head())

# Date ve Time sütunlarını artık kullanmadığımız için sil
df.drop(columns=["Date", "Time"], inplace=True)

print(df.head())

# datetime sütununu index yap
df.set_index("datetime", inplace=True)

print(df.head())

# -200 değerlerini eksik değer olarak kabul et
df.replace(-200, np.nan, inplace=True)

# Zaman bazlı interpolasyon
df.interpolate(method="time", inplace=True)

# Başta/sonda kalan eksik değerleri doldur
df.ffill(inplace=True)
df.bfill(inplace=True)

# Saat ve ay bilgilerini ekle
df["hour"] = df.index.hour
df["month"] = df.index.month

print(df.head())

# Kullanılacak sütunlar
selected_columns = [
    "CO(GT)",
    "PT08.S1(CO)",
    "NMHC(GT)",
    "C6H6(GT)",
    "PT08.S2(NMHC)",
    "NOx(GT)",
    "PT08.S3(NOx)",
    "NO2(GT)",
    "PT08.S4(NO2)",
    "PT08.S5(O3)",
    "T",
    "RH",
    "AH",
    "hour",
    "month"
]

df = df[selected_columns]

print(df.head())

# Eksik değerleri kontrol et
print(
    f"Number of missing values in each column:\n{df.isnull().sum()}"
)

# Missing Values Heatmap
plt.figure(figsize=(12, 6))

plt.title("Missing Values Heatmap")

sns.heatmap(
    df.isnull(),
    cbar=False,
    cmap="viridis"
)

plt.tight_layout()
plt.show()

# Correlation Heatmap
plt.figure(figsize=(14, 10))

sns.heatmap(
    df.corr(),
    annot=True,
    cmap="coolwarm"
)

plt.title("Correlation Heatmap")

plt.tight_layout()
plt.show()

# Boxplot
plt.figure(figsize=(15, 7))

sns.boxplot(data=df)

plt.title("Boxplot of Air Quality Parameters")

plt.xticks(rotation=90)

plt.tight_layout()
plt.show()