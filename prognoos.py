import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import joblib
import keras

# Mudeli ja skeileri loomine
model = keras.models.load_model("best_model.keras")
i_scaler = joblib.load('input_scaler.save')

# Uue hoone pindala
new_area = 1156.5

# Ajatarviku loomine
seasons = [
    ('Talv', datetime(2023, 12, 3), datetime(2023, 12, 9, 23),
     'D:\\TTU_data\\Tallinn, Estonia 2023-12-01 to 2023-12-31.csv'),
    ('Kevad', datetime(2023, 3, 19), datetime(2023, 3, 25, 23), 'D:\\TTU_data\\Tallinn 2023-03-01 to 2023-04-30.csv'),
    ('Suvi', datetime(2023, 7, 23), datetime(2023, 7, 29, 23),
     'D:\\TTU_data\\Tallinn, Estonia 2023-07-01 to 2023-07-31.csv'),
    ('Sügis', datetime(2023, 10, 22), datetime(2023, 10, 28, 23),
     'D:\\TTU_data\\Tallinn, Estonia 2023-10-01 to 2023-10-31.csv')
]

plt.figure(figsize=(12, 8))

#kontrollimine failide olemas, ning filtreerimine aegade kaupa
for i, (season, start_date, end_date, data_file) in enumerate(seasons, 1):
    if not os.path.exists(data_file):
        print(f'Faili {data_file} ei ole, {season} ei kasutatakse')
        continue

    weather = pd.read_csv(data_file)

    if 'Periood' in weather.columns:
        weather['FullTime'] = pd.to_datetime(weather['Periood'])
    else:
        raise ValueError('Aja andmeid ei ole')

    weather_week = weather[(weather['FullTime'] >= start_date) & (weather['FullTime'] <= end_date)].copy()

    predicted = []

    for _, row in weather_week.iterrows():
        time_now = row['FullTime']
        h = time_now.hour
        m = time_now.month

        hour_sin = np.sin(2 * np.pi * h / 24)
        hour_cos = np.cos(2 * np.pi * h / 24)
        month_sin = np.sin(2 * np.pi * m / 12)
        month_cos = np.cos(2 * np.pi * m / 12)

        temp_now = row.get('temp', np.nan)
        if np.isnan(temp_now):
            predicted.append(np.nan)
            continue

        X = np.array([new_area, 0, temp_now, hour_sin, hour_cos, month_sin, month_cos]).reshape(1, -1)

        if X.shape[1] < i_scaler.scale_.shape[0]:
            X = np.hstack([X, np.zeros((1, i_scaler.scale_.shape[0] - X.shape[1]))])

        X_scaled = i_scaler.transform(X)
        X_input = X_scaled.reshape(1, 1, -1)

        y_pred = model.predict(X_input, verbose=0)[0, 0]
        predicted.append(y_pred)

    plt.subplot(2, 2, i)
    plt.plot(predicted, color='blue')
    plt.title(f'{season} (Pindala: {new_area} m²)')
    plt.xlabel('Tunnid')
    plt.ylabel('Tarbimine (kWh)')
    plt.grid(True)

plt.tight_layout()
plt.suptitle('Uue hoone prognoseerimine neljal aastal (Tarbimine)')
plt.subplots_adjust(top=0.9)
plt.show()
