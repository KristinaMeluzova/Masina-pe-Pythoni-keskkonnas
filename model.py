import os

import joblib
import keras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from datetime import datetime
from keras.src.layers import BatchNormalization,LayerNormalization, Dense, Dropout, LSTM
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras.src.models import  Sequential
from keras.src import regularizers
from keras.src.optimizers import Adam
from keras.src.callbacks import EarlyStopping, ModelCheckpoint
import tensorflow.python.keras as tf_keras
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')
from keras import __version__
tf_keras.__version__ = __version__
from scipy.stats import pearsonr

#esimene osa, afailide lugemine ja nende andmete töötlemine
#failide lugemine
electricity_data_input = pd.read_excel("D:\\TTU_data\\Input_data_SOC_TIM_D04.xlsx")
electricity_data_input["Periood"] = electricity_data_input["Periood"].astype(str) + " " + electricity_data_input["time"].astype(str)
electricity_data_input = electricity_data_input.drop(columns=["time"])
electricity_data_input["Periood"] = pd.to_datetime(electricity_data_input["Periood"])

electricity_data_output = pd.read_excel("D:\\TTU_data\\Output_data_S01.xlsx")
electricity_data_output["Periood"] = electricity_data_output["Periood"].astype(str) + " " + electricity_data_output["time"].astype(str)
electricity_data_output = electricity_data_output.drop(columns=["time"])
electricity_data_output["Periood"] = pd.to_datetime(electricity_data_output["Periood"])

elec_data_full = pd.merge(electricity_data_input, electricity_data_output, on="Periood")

area = pd.read_excel("D:\\TTU_data\\Pindala.xlsx")
weather = pd.read_csv("weather_perfect.csv")

#muudatakse ajade andeid aja formaadile
weather['Periood'] = pd.to_datetime(weather["Periood"])

# muudatakse
electricity_long = elec_data_full.melt(
    id_vars=["Periood"],
    var_name='Hoone',
    value_name='Tarbimine'
)

#ühendamine "electricity_long" and "weather"
m_data = pd.merge(electricity_long, weather,  on="Periood")
m_data.to_csv("historic_data.csv")

#ühendamine m_data koos area
fl_data = pd.merge(m_data, area, on=["Hoone"])
full_data = fl_data.drop(columns=["Asukoht"])
full_data["Tund"] = full_data["Periood"].dt.hour
full_data["Nädal"] = full_data["Periood"].dt.weekday
full_data["Kuu"] = full_data["Periood"].dt.month

#kuude ja aegade sin ja cos väärtused
full_data['HourSin'] = np.sin(2 * np.pi * full_data["Tund"] / 24)
full_data['HourCos'] = np.cos(2 * np.pi * full_data["Tund"] / 24)
full_data['MonthSin'] = np.sin(2 * np.pi * full_data["Kuu"] / 12)
full_data['MonthCos'] = np.cos(2 * np.pi * full_data["Kuu"] / 12)

#nädalavahemiku filtreerimine
range_data_jan = full_data.loc[
    (full_data['Periood'] >= datetime(2023, 1,1)) &
   (full_data['Periood'] <= datetime(2023, 1,7, 23))
]

range_data_feb = full_data.loc[
    (full_data['Periood'] >= datetime(2023, 2,12)) &
   (full_data['Periood'] <= datetime(2023, 2,18,23))
]

range_data_mar = full_data.loc[
    (full_data['Periood'] >= datetime(2023, 3,26)) &
   (full_data['Periood'] <= datetime(2023, 4,1,23))
]

range_data_may = full_data.loc[
    (full_data['Periood'] >= datetime(2023, 5,7)) &
   (full_data['Periood'] <= datetime(2023, 5,13,23))
]

range_data_jun = full_data.loc[
    (full_data['Periood'] >= datetime(2023, 6,18)) &
   (full_data['Periood'] <= datetime(2023, 6,24,23))
]

range_data_jul = full_data.loc[
    (full_data['Periood'] >= datetime(2023, 7,30)) &
   (full_data['Periood'] <= datetime(2023, 8,5,23))
]

range_data_sep = full_data.loc[
    (full_data['Periood'] >= datetime(2023, 9,10)) &
   (full_data['Periood'] <= datetime(2023, 9,16,23))
]

range_data_oct = full_data.loc[
    (full_data['Periood'] >= datetime(2023, 10,22)) &
   (full_data['Periood'] <= datetime(2023, 10,28,23))
]
range_data_dec = full_data.loc[
    (full_data['Periood'] >= datetime(2023, 12,3)) &
   (full_data['Periood'] <= datetime(2023, 12,9,23))
]

range_data = pd.concat([range_data_jan, range_data_feb,range_data_mar, range_data_may,range_data_jun,range_data_jul,range_data_sep,range_data_oct,range_data_dec])

#teine osa, sisendandmete ettevalmistamine
necessary_buildings = ["SOC","TIM","D04"]
targ_building = "S01"

inp_array = []
for building in necessary_buildings:
    buildings_input = range_data[range_data["Hoone"] == building].copy()
    buildings_input['Absol_Tarbimine'] = buildings_input['Tarbimine'] * buildings_input['Pindala']
    feat = buildings_input[['Pindala','Absol_Tarbimine','temp', 'HourSin', 'HourCos', 'MonthSin', 'MonthCos']].values
    inp_array.append(feat)

inp_array = np.concatenate(inp_array, axis=0)

targ_build = m_data[m_data["Hoone"] == targ_building]
targ = targ_build[['Tarbimine']].values
#treenimiseks ja testimiseks ettevalmistamine
i_scaler = MinMaxScaler(feature_range=(-1, 1))
inp_scaler = i_scaler.fit_transform(inp_array)

inpu = []
target = []
for i in range(int(len(inp_scaler))  - 1512 - 168):#1512 tuni -63 põeva nagu sisend (1 nädal ühe kuust = 9 nädalat) # 7 päeva- 1 nädal
    if len(targ) > i + 1512 + 168:
        inpu.append(inp_scaler[i:i + 1512])
        target.append(targ[i+ 1512:i + 1512 + 168,0])
Input = np.array(inpu)
Target = np.array(target)
print("Input shape:", Input.shape)
print("Target shape:", Target.shape)
inp_train, inp_test, targ_train, targ_test = train_test_split(Input, Target, train_size=0.7, shuffle=False)


#neljas osa, mudeli treenimine ja kompileerimine

if os.path.exists("best_model.keras"):
    model = keras.models.load_model("best_model.keras")
    print("modul on loodud")
else:
    model = Sequential()
    model.add(LSTM(40,input_shape=(1512,7), return_sequences=True, kernel_regularizer=regularizers.L2(0.001)))
    model.add(LayerNormalization())
    model.add(LSTM(40, return_sequences=False))
    model.add(Dense(30,activation="relu"))
    model.add(Dropout(0.4))
    model.add(Dense(168))
    model.compile(loss="mse", optimizer="Adam", metrics=['mae'])


model.summary()


#checkpoint
cp = ModelCheckpoint("best_model.keras", save_best_only=True, save_weights_only=False)



#varem stop
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

#treenimine
model_train = model.fit(inp_train,targ_train, epochs=1000, batch_size=32,validation_data=(inp_test, targ_test), callbacks=[early_stop, cp])

model_history = pd.DataFrame(model_train.history)
model_history['epoch'] = model_train.epoch

plt.plot(model_history["loss"], label = f"Kõige parem treeningu tulemus on {min(model_history["loss"])} õppetsüklil: {max(model_history['epoch'])}")
plt.ylabel('Mean Squared Error (mse)')
plt.xlabel('epochs')
plt.grid(True)
plt.legend()
plt.show()

#mudeli salvestamine
model.save("new_model.keras",  overwrite=True, include_optimizer=True)
joblib.dump(i_scaler, 'input_scaler.save')

test_predicted = model.predict(inp_test)
scaled_data = np.zeros((test_predicted.shape[0], inp_scaler.shape[1]))
scaled_data[:, 1] = test_predicted.flatten()
norm_test = i_scaler.inverse_transform(scaled_data)[:, 1]
norm_test = norm_test.reshape(-1, 168)
print(f"Test prognoos: {norm_test}")
print(f"Reaalandmed: {targ_test}")
# mudeli hindamine
cor = pearsonr(targ_test.flatten(), norm_test.flatten())

test_mse = mean_squared_error(targ_test, norm_test)
test_rmse = np.sqrt(mean_squared_error(targ_test, norm_test))
test_mae = mean_absolute_error(targ_test, norm_test)
test_r2 = r2_score(targ_test, norm_test)

print(f"Test mse: {test_mse}")
print(f"Test rmse: {test_rmse}")
print(f"Test mae: {test_mae}")
print(f"Test r2: {test_r2}")
print(f"cor: {cor}")
