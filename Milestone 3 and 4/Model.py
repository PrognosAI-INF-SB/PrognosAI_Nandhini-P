================================================================================
AI-PrognosAI: Predictive Maintenance System
Hybrid LSTM Model for RUL Prediction
================================================================================

================================================================================
MILESTONE 1: DATA PREPARATION & FEATURE ENGINEERING
================================================================================

✓ Dataset loaded successfully from 'sensor_data.csv'

--- Dataset Summary ---
Shape: (1440, 4)

Column Types:
time        object
SensorA    float64
SensorB    float64
SensorC    float64
dtype: object

Missing Values:
time         0
SensorA    137
SensorB    235
SensorC     48
dtype: int64

Descriptive Statistics:
           SensorA      SensorB      SensorC
count  1303.000000  1205.000000  1392.000000
mean      9.793641     3.215945     8.013604
std       2.038163     2.843696     2.138642
min       1.416250    -2.740000     1.055965
25%       8.466171     1.114624     6.524029
50%       9.809673     3.210922     8.170571
75%      11.238712     5.464741     9.618772
max      18.107463     9.564000    14.307993

--- Data Cleaning ---
✓ Missing values handled
Remaining missing values: 0

--- Feature Normalization ---
✓ Sensor columns normalized (StandardScaler)
Mean after scaling: [-7.50017332e-16 -7.89491929e-17 -1.57898386e-16]
Std after scaling: [1.0003474 1.0003474 1.0003474]

--- Creating Rolling Window Sequences ---
✓ Sequences created
  Input shape (samples, timesteps, features): (282, 30, 3)

--- Computing RUL Targets ---
✓ RUL targets computed
  RUL shape: (282,)
  RUL range: [0.28, 97.92]
  RUL mean: 49.10 ± 28.29

--- Data Splitting ---
✓ Data split completed
  Training set:   196 samples (69.5%)
  Validation set: 43 samples (15.2%)
  Test set:       43 samples (15.2%)

================================================================================
MILESTONE 1 COMPLETE ✓
================================================================================

================================================================================
MILESTONE 2: HYBRID LSTM MODEL DEVELOPMENT
================================================================================

--- Building Hybrid LSTM Architecture ---
✓ Hybrid LSTM model built

--- Model Architecture ---
Model: "sequential"
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃ Layer (type)                    ┃ Output Shape           ┃       Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ conv1d (Conv1D)                 │ (None, 30, 64)         │           640 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ batch_normalization             │ (None, 30, 64)         │           256 │
│ (BatchNormalization)            │                        │               │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dropout (Dropout)               │ (None, 30, 64)         │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ conv1d_1 (Conv1D)               │ (None, 30, 32)         │         6,176 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ batch_normalization_1           │ (None, 30, 32)         │           128 │
│ (BatchNormalization)            │                        │               │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dropout_1 (Dropout)             │ (None, 30, 32)         │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ bidirectional (Bidirectional)   │ (None, 30, 128)        │        49,664 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dropout_2 (Dropout)             │ (None, 30, 128)        │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ bidirectional_1 (Bidirectional) │ (None, 64)             │        41,216 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dropout_3 (Dropout)             │ (None, 64)             │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense (Dense)                   │ (None, 64)             │         4,160 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dropout_4 (Dropout)             │ (None, 64)             │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_1 (Dense)                 │ (None, 32)             │         2,080 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_2 (Dense)                 │ (None, 1)              │            33 │
└─────────────────────────────────┴────────────────────────┴───────────────┘
 Total params: 104,353 (407.63 KB)
 Trainable params: 104,161 (406.88 KB)
 Non-trainable params: 192 (768.00 B)

--- Setting up Training Callbacks ---
✓ Callbacks configured: EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

--- Training Hybrid LSTM Model ---
Epoch 1/100
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 44ms/step - loss: 3376.8315 - mae: 51.1011 - mse: 3376.8315WARNING:absl:You are saving your model as an HDF5 file via `model.save()` or `keras.saving.save_model(model)`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')` or `keras.saving.save_model(model, 'my_model.keras')`. 
4/4 ━━━━━━━━━━━━━━━━━━━━ 6s 261ms/step - loss: 3357.0120 - mae: 50.9163 - mse: 3357.0120 - val_loss: 3272.3298 - val_mae: 48.4774 - val_mse: 3272.3298 - learning_rate: 0.0010
Epoch 2/100
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 49ms/step - loss: 3349.7815 - mae: 50.8925 - mse: 3349.7815WARNING:absl:You are saving your model as an HDF5 file via `model.save()` or `keras.saving.save_model(model)`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')` or `keras.saving.save_model(model, 'my_model.keras')`. 
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 112ms/step - loss: 3330.2227 - mae: 50.7083 - mse: 3330.2227 - val_loss: 3260.8162 - val_mae: 48.3899 - val_mse: 3260.8162 - learning_rate: 0.0010
Epoch 3/100
3/4 ━━━━━━━━━━━━━━━━━━━━ 0s 79ms/step - loss: 3353.6384 - mae: 50.9896 - mse: 3353.6384WARNING:absl:You are saving your model as an HDF5 file via `model.save()` or `keras.saving.save_model(model)`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')` or `keras.saving.save_model(model, 'my_model.keras')`. 
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 124ms/step - loss: 3300.6990 - mae: 50.4897 - mse: 3300.6990 - val_loss: 3233.7893 - val_mae: 48.1980 - val_mse: 3233.7893 - learning_rate: 0.0010
Epoch 4/100
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 77ms/step - loss: 3268.9336 - mae: 50.2937 - mse: 3268.9336WARNING:absl:You are saving your model as an HDF5 file via `model.save()` or `keras.saving.save_model(model)`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')` or `keras.saving.save_model(model, 'my_model.keras')`. 
4/4 ━━━━━━━━━━━━━━━━━━━━ 1s 126ms/step - loss: 3248.9531 - mae: 50.1041 - mse: 3248.9531 - val_loss: 3172.6316 - val_mae: 47.7709 - val_mse: 3172.6316 - learning_rate: 0.0010
Epoch 5/100
3/4 ━━━━━━━━━━━━━━━━━━━━ 0s 46ms/step - loss: 3216.8860 - mae: 49.9610 - mse: 3216.8860 WARNING:absl:You are saving your model as an HDF5 file via `model.save()` or `keras.saving.save_model(model)`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')` or `keras.saving.save_model(model, 'my_model.keras')`. 
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 74ms/step - loss: 3164.2520 - mae: 49.4552 - mse: 3164.2520 - val_loss: 3058.9226 - val_mae: 46.9602 - val_mse: 3058.9226 - learning_rate: 0.0010
Epoch 6/100
3/4 ━━━━━━━━━━━━━━━━━━━━ 0s 45ms/step - loss: 3098.1873 - mae: 49.0082 - mse: 3098.1873WARNING:absl:You are saving your model as an HDF5 file via `model.save()` or `keras.saving.save_model(model)`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')` or `keras.saving.save_model(model, 'my_model.keras')`. 
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 76ms/step - loss: 3041.5024 - mae: 48.4583 - mse: 3041.5024 - val_loss: 2888.6313 - val_mae: 45.6352 - val_mse: 2888.6313 - learning_rate: 0.0010
Epoch 7/100
3/4 ━━━━━━━━━━━━━━━━━━━━ 0s 58ms/step - loss: 2917.5996 - mae: 47.4866 - mse: 2917.5996WARNING:absl:You are saving your model as an HDF5 file via `model.save()` or `keras.saving.save_model(model)`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')` or `keras.saving.save_model(model, 'my_model.keras')`. 
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 83ms/step - loss: 2862.3904 - mae: 46.9257 - mse: 2862.3904 - val_loss: 2662.3796 - val_mae: 43.6411 - val_mse: 2662.3796 - learning_rate: 0.0010
Epoch 8/100
3/4 ━━━━━━━━━━━━━━━━━━━━ 0s 43ms/step - loss: 2687.6790 - mae: 45.2797 - mse: 2687.6790WARNING:absl:You are saving your model as an HDF5 file via `model.save()` or `keras.saving.save_model(model)`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')` or `keras.saving.save_model(model, 'my_model.keras')`. 
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 73ms/step - loss: 2631.2261 - mae: 44.6609 - mse: 2631.2261 - val_loss: 2385.1021 - val_mae: 40.9148 - val_mse: 2385.1021 - learning_rate: 0.0010
Epoch 9/100
3/4 ━━━━━━━━━━━━━━━━━━━━ 0s 45ms/step - loss: 2395.9021 - mae: 42.1185 - mse: 2395.9021WARNING:absl:You are saving your model as an HDF5 file via `model.save()` or `keras.saving.save_model(model)`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')` or `keras.saving.save_model(model, 'my_model.keras')`. 
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 74ms/step - loss: 2339.6924 - mae: 41.4724 - mse: 2339.6924 - val_loss: 2057.8560 - val_mae: 37.2822 - val_mse: 2057.8560 - learning_rate: 0.0010
Epoch 10/100
3/4 ━━━━━━━━━━━━━━━━━━━━ 0s 44ms/step - loss: 2071.1682 - mae: 38.5282 - mse: 2071.1682WARNING:absl:You are saving your model as an HDF5 file via `model.save()` or `keras.saving.save_model(model)`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')` or `keras.saving.save_model(model, 'my_model.keras')`. 
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 79ms/step - loss: 2017.4448 - mae: 37.8496 - mse: 2017.4448 - val_loss: 1708.8103 - val_mae: 33.6455 - val_mse: 1708.8103 - learning_rate: 0.0010
Epoch 11/100
3/4 ━━━━━━━━━━━━━━━━━━━━ 0s 45ms/step - loss: 1683.9066 - mae: 33.9443 - mse: 1683.9066WARNING:absl:You are saving your model as an HDF5 file via `model.save()` or `keras.saving.save_model(model)`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')` or `keras.saving.save_model(model, 'my_model.keras')`. 
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 76ms/step - loss: 1633.3081 - mae: 33.2436 - mse: 1633.3081 - val_loss: 1380.2899 - val_mae: 30.7487 - val_mse: 1380.2899 - learning_rate: 0.0010
Epoch 12/100
3/4 ━━━━━━━━━━━━━━━━━━━━ 0s 47ms/step - loss: 1290.7316 - mae: 28.6766 - mse: 1290.7316WARNING:absl:You are saving your model as an HDF5 file via `model.save()` or `keras.saving.save_model(model)`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')` or `keras.saving.save_model(model, 'my_model.keras')`. 
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 75ms/step - loss: 1256.4836 - mae: 28.1229 - mse: 1256.4836 - val_loss: 1094.1479 - val_mae: 28.3264 - val_mse: 1094.1479 - learning_rate: 0.0010
Epoch 13/100
3/4 ━━━━━━━━━━━━━━━━━━━━ 0s 44ms/step - loss: 1024.8105 - mae: 24.8485 - mse: 1024.8105WARNING:absl:You are saving your model as an HDF5 file via `model.save()` or `keras.saving.save_model(model)`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')` or `keras.saving.save_model(model, 'my_model.keras')`. 
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 76ms/step - loss: 995.5285 - mae: 24.4413 - mse: 995.5285 - val_loss: 850.4561 - val_mae: 25.6206 - val_mse: 850.4561 - learning_rate: 0.0010
Epoch 14/100
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 47ms/step - loss: 694.3184 - mae: 20.6389 - mse: 694.3184WARNING:absl:You are saving your model as an HDF5 file via `model.save()` or `keras.saving.save_model(model)`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')` or `keras.saving.save_model(model, 'my_model.keras')`. 
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 97ms/step - loss: 685.7667 - mae: 20.5242 - mse: 685.7667 - val_loss: 578.9302 - val_mae: 20.9740 - val_mse: 578.9302 - learning_rate: 0.0010
Epoch 15/100
3/4 ━━━━━━━━━━━━━━━━━━━━ 0s 45ms/step - loss: 514.7179 - mae: 17.8064 - mse: 514.7179WARNING:absl:You are saving your model as an HDF5 file via `model.save()` or `keras.saving.save_model(model)`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')` or `keras.saving.save_model(model, 'my_model.keras')`. 
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 75ms/step - loss: 503.8926 - mae: 17.6436 - mse: 503.8926 - val_loss: 371.0053 - val_mae: 16.2828 - val_mse: 371.0053 - learning_rate: 0.0010
Epoch 16/100
3/4 ━━━━━━━━━━━━━━━━━━━━ 0s 45ms/step - loss: 448.5085 - mae: 16.8676 - mse: 448.5085WARNING:absl:You are saving your model as an HDF5 file via `model.save()` or `keras.saving.save_model(model)`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')` or `keras.saving.save_model(model, 'my_model.keras')`. 
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 75ms/step - loss: 443.5446 - mae: 16.7877 - mse: 443.5446 - val_loss: 284.5752 - val_mae: 13.4084 - val_mse: 284.5752 - learning_rate: 0.0010
Epoch 17/100
3/4 ━━━━━━━━━━━━━━━━━━━━ 0s 46ms/step - loss: 398.0314 - mae: 16.1091 - mse: 398.0314WARNING:absl:You are saving your model as an HDF5 file via `model.save()` or `keras.saving.save_model(model)`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')` or `keras.saving.save_model(model, 'my_model.keras')`. 
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 83ms/step - loss: 394.7263 - mae: 15.9785 - mse: 394.7263 - val_loss: 256.6652 - val_mae: 12.4335 - val_mse: 256.6652 - learning_rate: 0.0010
Epoch 18/100
3/4 ━━━━━━━━━━━━━━━━━━━━ 0s 45ms/step - loss: 343.7899 - mae: 14.9770 - mse: 343.7899 WARNING:absl:You are saving your model as an HDF5 file via `model.save()` or `keras.saving.save_model(model)`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')` or `keras.saving.save_model(model, 'my_model.keras')`. 
4/4 ━━━━━━━━━━━━━━━━━━━━ 1s 78ms/step - loss: 346.7793 - mae: 15.0266 - mse: 346.7793 - val_loss: 251.4800 - val_mae: 12.3233 - val_mse: 251.4800 - learning_rate: 0.0010
Epoch 19/100
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 58ms/step - loss: 349.2975 - mae: 14.7495 - mse: 349.2975 - val_loss: 267.1164 - val_mae: 12.7557 - val_mse: 267.1164 - learning_rate: 0.0010
Epoch 20/100
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 55ms/step - loss: 427.1047 - mae: 16.1989 - mse: 427.1047 - val_loss: 255.1739 - val_mae: 12.4380 - val_mse: 255.1739 - learning_rate: 0.0010
Epoch 21/100
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 57ms/step - loss: 352.6471 - mae: 14.8250 - mse: 352.6471 - val_loss: 253.6440 - val_mae: 12.3708 - val_mse: 253.6440 - learning_rate: 0.0010
Epoch 22/100
3/4 ━━━━━━━━━━━━━━━━━━━━ 0s 46ms/step - loss: 299.9543 - mae: 13.3698 - mse: 299.9543 WARNING:absl:You are saving your model as an HDF5 file via `model.save()` or `keras.saving.save_model(model)`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')` or `keras.saving.save_model(model, 'my_model.keras')`. 
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 77ms/step - loss: 288.6008 - mae: 13.1272 - mse: 288.6008 - val_loss: 238.7991 - val_mae: 12.0094 - val_mse: 238.7991 - learning_rate: 0.0010
Epoch 23/100
3/4 ━━━━━━━━━━━━━━━━━━━━ 0s 54ms/step - loss: 287.8192 - mae: 12.9141 - mse: 287.8192WARNING:absl:You are saving your model as an HDF5 file via `model.save()` or `keras.saving.save_model(model)`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')` or `keras.saving.save_model(model, 'my_model.keras')`. 
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 88ms/step - loss: 283.0518 - mae: 12.6838 - mse: 283.0518 - val_loss: 210.0705 - val_mae: 11.3424 - val_mse: 210.0705 - learning_rate: 0.0010
Epoch 24/100
3/4 ━━━━━━━━━━━━━━━━━━━━ 0s 44ms/step - loss: 242.0109 - mae: 11.2241 - mse: 242.0109WARNING:absl:You are saving your model as an HDF5 file via `model.save()` or `keras.saving.save_model(model)`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')` or `keras.saving.save_model(model, 'my_model.keras')`. 
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 76ms/step - loss: 237.4543 - mae: 11.1328 - mse: 237.4543 - val_loss: 185.4352 - val_mae: 11.0154 - val_mse: 185.4352 - learning_rate: 0.0010
Epoch 25/100
3/4 ━━━━━━━━━━━━━━━━━━━━ 0s 45ms/step - loss: 223.4059 - mae: 10.9072 - mse: 223.4059WARNING:absl:You are saving your model as an HDF5 file via `model.save()` or `keras.saving.save_model(model)`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')` or `keras.saving.save_model(model, 'my_model.keras')`. 
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 76ms/step - loss: 221.8148 - mae: 10.8059 - mse: 221.8148 - val_loss: 161.6398 - val_mae: 10.2178 - val_mse: 161.6398 - learning_rate: 0.0010
Epoch 26/100
3/4 ━━━━━━━━━━━━━━━━━━━━ 0s 46ms/step - loss: 178.8608 - mae: 9.5733 - mse: 178.8608 WARNING:absl:You are saving your model as an HDF5 file via `model.save()` or `keras.saving.save_model(model)`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')` or `keras.saving.save_model(model, 'my_model.keras')`. 
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 81ms/step - loss: 170.6706 - mae: 9.3106 - mse: 170.6706 - val_loss: 145.7176 - val_mae: 9.4198 - val_mse: 145.7176 - learning_rate: 0.0010
Epoch 27/100
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 59ms/step - loss: 138.6486 - mae: 8.9865 - mse: 138.6486 - val_loss: 148.9110 - val_mae: 9.5848 - val_mse: 148.9110 - learning_rate: 0.0010
Epoch 28/100
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 58ms/step - loss: 155.9496 - mae: 9.0161 - mse: 155.9496 - val_loss: 168.2626 - val_mae: 9.9248 - val_mse: 168.2626 - learning_rate: 0.0010
Epoch 29/100
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 59ms/step - loss: 99.7337 - mae: 7.3603 - mse: 99.7337 - val_loss: 169.0623 - val_mae: 9.8410 - val_mse: 169.0623 - learning_rate: 0.0010
Epoch 30/100
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 68ms/step - loss: 104.8337 - mae: 7.7411 - mse: 104.8337 - val_loss: 160.4308 - val_mae: 9.6209 - val_mse: 160.4308 - learning_rate: 0.0010
Epoch 31/100
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 42ms/step - loss: 81.0859 - mae: 6.7735 - mse: 81.0859
Epoch 31: ReduceLROnPlateau reducing learning rate to 0.0005000000237487257.
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 62ms/step - loss: 83.4634 - mae: 6.8393 - mse: 83.4634 - val_loss: 146.7459 - val_mae: 9.1261 - val_mse: 146.7459 - learning_rate: 0.0010
Epoch 32/100
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 57ms/step - loss: 92.6511 - mae: 7.3655 - mse: 92.6511 - val_loss: 146.5294 - val_mae: 9.2322 - val_mse: 146.5294 - learning_rate: 5.0000e-04
Epoch 33/100
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 59ms/step - loss: 94.7624 - mae: 7.4839 - mse: 94.7624 - val_loss: 159.9154 - val_mae: 9.7230 - val_mse: 159.9154 - learning_rate: 5.0000e-04
Epoch 34/100
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 62ms/step - loss: 91.0689 - mae: 7.0329 - mse: 91.0689 - val_loss: 172.1588 - val_mae: 10.3120 - val_mse: 172.1588 - learning_rate: 5.0000e-04
Epoch 35/100
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 57ms/step - loss: 83.1945 - mae: 6.9782 - mse: 83.1945 - val_loss: 173.3988 - val_mae: 10.5706 - val_mse: 173.3988 - learning_rate: 5.0000e-04
Epoch 36/100
3/4 ━━━━━━━━━━━━━━━━━━━━ 0s 47ms/step - loss: 111.7749 - mae: 7.9124 - mse: 111.7749 
Epoch 36: ReduceLROnPlateau reducing learning rate to 0.0002500000118743628.
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 59ms/step - loss: 109.1769 - mae: 7.8145 - mse: 109.1769 - val_loss: 163.8965 - val_mae: 10.4607 - val_mse: 163.8965 - learning_rate: 5.0000e-04
Epoch 37/100
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 107ms/step - loss: 99.4201 - mae: 7.0852 - mse: 99.4201 - val_loss: 156.1615 - val_mae: 10.2077 - val_mse: 156.1615 - learning_rate: 2.5000e-04
Epoch 38/100
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 103ms/step - loss: 115.3195 - mae: 7.9871 - mse: 115.3195 - val_loss: 149.2959 - val_mae: 9.9264 - val_mse: 149.2959 - learning_rate: 2.5000e-04
Epoch 39/100
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 70ms/step - loss: 121.5761 - mae: 8.1095 - mse: 121.5761WARNING:absl:You are saving your model as an HDF5 file via `model.save()` or `keras.saving.save_model(model)`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')` or `keras.saving.save_model(model, 'my_model.keras')`. 
4/4 ━━━━━━━━━━━━━━━━━━━━ 1s 115ms/step - loss: 120.2704 - mae: 8.0382 - mse: 120.2704 - val_loss: 144.7796 - val_mae: 9.6608 - val_mse: 144.7796 - learning_rate: 2.5000e-04
Epoch 40/100
3/4 ━━━━━━━━━━━━━━━━━━━━ 0s 45ms/step - loss: 92.1925 - mae: 7.2492 - mse: 92.1925WARNING:absl:You are saving your model as an HDF5 file via `model.save()` or `keras.saving.save_model(model)`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')` or `keras.saving.save_model(model, 'my_model.keras')`. 
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 76ms/step - loss: 88.7086 - mae: 7.0605 - mse: 88.7086 - val_loss: 138.2334 - val_mae: 9.2872 - val_mse: 138.2334 - learning_rate: 2.5000e-04
Epoch 41/100
3/4 ━━━━━━━━━━━━━━━━━━━━ 0s 45ms/step - loss: 110.6388 - mae: 7.8697 - mse: 110.6388WARNING:absl:You are saving your model as an HDF5 file via `model.save()` or `keras.saving.save_model(model)`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')` or `keras.saving.save_model(model, 'my_model.keras')`. 
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 76ms/step - loss: 108.1239 - mae: 7.7739 - mse: 108.1239 - val_loss: 136.7712 - val_mae: 9.1536 - val_mse: 136.7712 - learning_rate: 2.5000e-04
Epoch 42/100
3/4 ━━━━━━━━━━━━━━━━━━━━ 0s 50ms/step - loss: 96.5987 - mae: 7.4851 - mse: 96.5987  WARNING:absl:You are saving your model as an HDF5 file via `model.save()` or `keras.saving.save_model(model)`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')` or `keras.saving.save_model(model, 'my_model.keras')`. 
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 79ms/step - loss: 92.9709 - mae: 7.2542 - mse: 92.9709 - val_loss: 132.0197 - val_mae: 8.9087 - val_mse: 132.0197 - learning_rate: 2.5000e-04
Epoch 43/100
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 40ms/step - loss: 102.1216 - mae: 7.4937 - mse: 102.1216WARNING:absl:You are saving your model as an HDF5 file via `model.save()` or `keras.saving.save_model(model)`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')` or `keras.saving.save_model(model, 'my_model.keras')`. 
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 79ms/step - loss: 99.2898 - mae: 7.3621 - mse: 99.2898 - val_loss: 127.0676 - val_mae: 8.6825 - val_mse: 127.0676 - learning_rate: 2.5000e-04
Epoch 44/100
3/4 ━━━━━━━━━━━━━━━━━━━━ 0s 45ms/step - loss: 91.0841 - mae: 7.1520 - mse: 91.0841WARNING:absl:You are saving your model as an HDF5 file via `model.save()` or `keras.saving.save_model(model)`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')` or `keras.saving.save_model(model, 'my_model.keras')`. 
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 74ms/step - loss: 90.5680 - mae: 7.0873 - mse: 90.5680 - val_loss: 118.0611 - val_mae: 8.3169 - val_mse: 118.0611 - learning_rate: 2.5000e-04
Epoch 45/100
3/4 ━━━━━━━━━━━━━━━━━━━━ 0s 52ms/step - loss: 102.3928 - mae: 7.2912 - mse: 102.3928WARNING:absl:You are saving your model as an HDF5 file via `model.save()` or `keras.saving.save_model(model)`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')` or `keras.saving.save_model(model, 'my_model.keras')`. 
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 80ms/step - loss: 99.6456 - mae: 7.1880 - mse: 99.6456 - val_loss: 111.9436 - val_mae: 8.1358 - val_mse: 111.9436 - learning_rate: 2.5000e-04
Epoch 46/100
3/4 ━━━━━━━━━━━━━━━━━━━━ 0s 48ms/step - loss: 105.3957 - mae: 7.4909 - mse: 105.3957WARNING:absl:You are saving your model as an HDF5 file via `model.save()` or `keras.saving.save_model(model)`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')` or `keras.saving.save_model(model, 'my_model.keras')`. 
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 80ms/step - loss: 100.0878 - mae: 7.2630 - mse: 100.0878 - val_loss: 104.3621 - val_mae: 7.8402 - val_mse: 104.3621 - learning_rate: 2.5000e-04
Epoch 47/100
3/4 ━━━━━━━━━━━━━━━━━━━━ 0s 46ms/step - loss: 79.0150 - mae: 6.8107 - mse: 79.0150WARNING:absl:You are saving your model as an HDF5 file via `model.save()` or `keras.saving.save_model(model)`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')` or `keras.saving.save_model(model, 'my_model.keras')`. 
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 75ms/step - loss: 79.2796 - mae: 6.8023 - mse: 79.2796 - val_loss: 100.9682 - val_mae: 7.7394 - val_mse: 100.9682 - learning_rate: 2.5000e-04
Epoch 48/100
3/4 ━━━━━━━━━━━━━━━━━━━━ 0s 46ms/step - loss: 63.7426 - mae: 6.0417 - mse: 63.7426WARNING:absl:You are saving your model as an HDF5 file via `model.save()` or `keras.saving.save_model(model)`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')` or `keras.saving.save_model(model, 'my_model.keras')`. 
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 77ms/step - loss: 66.3990 - mae: 6.0986 - mse: 66.3990 - val_loss: 96.8146 - val_mae: 7.6433 - val_mse: 96.8146 - learning_rate: 2.5000e-04
Epoch 49/100
3/4 ━━━━━━━━━━━━━━━━━━━━ 0s 45ms/step - loss: 94.5573 - mae: 7.1569 - mse: 94.5573WARNING:absl:You are saving your model as an HDF5 file via `model.save()` or `keras.saving.save_model(model)`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')` or `keras.saving.save_model(model, 'my_model.keras')`. 
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 78ms/step - loss: 93.5605 - mae: 7.1127 - mse: 93.5605 - val_loss: 91.3204 - val_mae: 7.4523 - val_mse: 91.3204 - learning_rate: 2.5000e-04
Epoch 50/100
3/4 ━━━━━━━━━━━━━━━━━━━━ 0s 46ms/step - loss: 71.8722 - mae: 6.5024 - mse: 71.8722WARNING:absl:You are saving your model as an HDF5 file via `model.save()` or `keras.saving.save_model(model)`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')` or `keras.saving.save_model(model, 'my_model.keras')`. 
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 76ms/step - loss: 72.4799 - mae: 6.5054 - mse: 72.4799 - val_loss: 89.5530 - val_mae: 7.4128 - val_mse: 89.5530 - learning_rate: 2.5000e-04
Epoch 51/100
3/4 ━━━━━━━━━━━━━━━━━━━━ 0s 46ms/step - loss: 88.5470 - mae: 6.7292 - mse: 88.5470WARNING:absl:You are saving your model as an HDF5 file via `model.save()` or `keras.saving.save_model(model)`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')` or `keras.saving.save_model(model, 'my_model.keras')`. 
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 78ms/step - loss: 85.9651 - mae: 6.5844 - mse: 85.9651 - val_loss: 88.0565 - val_mae: 7.3375 - val_mse: 88.0565 - learning_rate: 2.5000e-04
Epoch 52/100
3/4 ━━━━━━━━━━━━━━━━━━━━ 0s 46ms/step - loss: 82.8753 - mae: 6.8051 - mse: 82.8753WARNING:absl:You are saving your model as an HDF5 file via `model.save()` or `keras.saving.save_model(model)`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')` or `keras.saving.save_model(model, 'my_model.keras')`. 
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 76ms/step - loss: 81.2253 - mae: 6.7640 - mse: 81.2253 - val_loss: 84.8196 - val_mae: 7.1947 - val_mse: 84.8196 - learning_rate: 2.5000e-04
Epoch 53/100
3/4 ━━━━━━━━━━━━━━━━━━━━ 0s 66ms/step - loss: 79.0153 - mae: 6.2934 - mse: 79.0153WARNING:absl:You are saving your model as an HDF5 file via `model.save()` or `keras.saving.save_model(model)`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')` or `keras.saving.save_model(model, 'my_model.keras')`. 
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 91ms/step - loss: 75.8467 - mae: 6.2226 - mse: 75.8467 - val_loss: 79.5470 - val_mae: 6.9823 - val_mse: 79.5470 - learning_rate: 2.5000e-04
Epoch 54/100
3/4 ━━━━━━━━━━━━━━━━━━━━ 0s 44ms/step - loss: 58.4978 - mae: 5.8949 - mse: 58.4978WARNING:absl:You are saving your model as an HDF5 file via `model.save()` or `keras.saving.save_model(model)`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')` or `keras.saving.save_model(model, 'my_model.keras')`. 
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 72ms/step - loss: 59.9198 - mae: 5.8960 - mse: 59.9198 - val_loss: 70.9137 - val_mae: 6.6135 - val_mse: 70.9137 - learning_rate: 2.5000e-04
Epoch 55/100
3/4 ━━━━━━━━━━━━━━━━━━━━ 0s 46ms/step - loss: 83.8970 - mae: 6.4860 - mse: 83.8970WARNING:absl:You are saving your model as an HDF5 file via `model.save()` or `keras.saving.save_model(model)`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')` or `keras.saving.save_model(model, 'my_model.keras')`. 
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 76ms/step - loss: 80.9303 - mae: 6.4234 - mse: 80.9303 - val_loss: 59.8134 - val_mae: 6.0460 - val_mse: 59.8134 - learning_rate: 2.5000e-04
Epoch 56/100
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 40ms/step - loss: 78.1225 - mae: 6.5175 - mse: 78.1225WARNING:absl:You are saving your model as an HDF5 file via `model.save()` or `keras.saving.save_model(model)`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')` or `keras.saving.save_model(model, 'my_model.keras')`. 
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 84ms/step - loss: 78.6728 - mae: 6.5398 - mse: 78.6728 - val_loss: 48.3377 - val_mae: 5.3326 - val_mse: 48.3377 - learning_rate: 2.5000e-04
Epoch 57/100
3/4 ━━━━━━━━━━━━━━━━━━━━ 0s 45ms/step - loss: 94.0119 - mae: 6.9125 - mse: 94.0119WARNING:absl:You are saving your model as an HDF5 file via `model.save()` or `keras.saving.save_model(model)`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')` or `keras.saving.save_model(model, 'my_model.keras')`. 
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 76ms/step - loss: 91.5865 - mae: 6.8209 - mse: 91.5865 - val_loss: 43.7473 - val_mae: 5.0701 - val_mse: 43.7473 - learning_rate: 2.5000e-04
Epoch 58/100
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 41ms/step - loss: 74.3500 - mae: 6.4128 - mse: 74.3500WARNING:absl:You are saving your model as an HDF5 file via `model.save()` or `keras.saving.save_model(model)`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')` or `keras.saving.save_model(model, 'my_model.keras')`. 
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 77ms/step - loss: 74.4199 - mae: 6.3712 - mse: 74.4199 - val_loss: 40.7738 - val_mae: 4.8590 - val_mse: 40.7738 - learning_rate: 2.5000e-04
Epoch 59/100
3/4 ━━━━━━━━━━━━━━━━━━━━ 0s 43ms/step - loss: 71.4094 - mae: 6.2607 - mse: 71.4094WARNING:absl:You are saving your model as an HDF5 file via `model.save()` or `keras.saving.save_model(model)`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')` or `keras.saving.save_model(model, 'my_model.keras')`. 
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 72ms/step - loss: 71.7878 - mae: 6.2444 - mse: 71.7878 - val_loss: 38.3131 - val_mae: 4.6409 - val_mse: 38.3131 - learning_rate: 2.5000e-04
Epoch 60/100
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 55ms/step - loss: 82.1752 - mae: 6.4979 - mse: 82.1752 - val_loss: 39.9247 - val_mae: 4.7956 - val_mse: 39.9247 - learning_rate: 2.5000e-04
Epoch 61/100
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 60ms/step - loss: 86.5816 - mae: 6.9118 - mse: 86.5816 - val_loss: 42.9579 - val_mae: 5.0204 - val_mse: 42.9579 - learning_rate: 2.5000e-04
Epoch 62/100
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 58ms/step - loss: 80.8358 - mae: 6.5449 - mse: 80.8358 - val_loss: 43.4932 - val_mae: 5.0569 - val_mse: 43.4932 - learning_rate: 2.5000e-04
Epoch 63/100
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 62ms/step - loss: 69.3706 - mae: 6.0740 - mse: 69.3706 - val_loss: 42.1428 - val_mae: 4.9504 - val_mse: 42.1428 - learning_rate: 2.5000e-04
Epoch 64/100
3/4 ━━━━━━━━━━━━━━━━━━━━ 0s 44ms/step - loss: 73.0488 - mae: 6.4402 - mse: 73.0488WARNING:absl:You are saving your model as an HDF5 file via `model.save()` or `keras.saving.save_model(model)`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')` or `keras.saving.save_model(model, 'my_model.keras')`. 
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 75ms/step - loss: 74.1453 - mae: 6.5015 - mse: 74.1453 - val_loss: 37.8286 - val_mae: 4.6188 - val_mse: 37.8286 - learning_rate: 2.5000e-04
Epoch 65/100
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 39ms/step - loss: 86.7687 - mae: 6.8916 - mse: 86.7687WARNING:absl:You are saving your model as an HDF5 file via `model.save()` or `keras.saving.save_model(model)`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')` or `keras.saving.save_model(model, 'my_model.keras')`. 
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 76ms/step - loss: 86.1355 - mae: 6.8622 - mse: 86.1355 - val_loss: 32.5639 - val_mae: 4.2086 - val_mse: 32.5639 - learning_rate: 2.5000e-04
Epoch 66/100
3/4 ━━━━━━━━━━━━━━━━━━━━ 0s 44ms/step - loss: 79.2939 - mae: 6.6532 - mse: 79.2939WARNING:absl:You are saving your model as an HDF5 file via `model.save()` or `keras.saving.save_model(model)`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')` or `keras.saving.save_model(model, 'my_model.keras')`. 
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 75ms/step - loss: 74.8872 - mae: 6.5046 - mse: 74.8872 - val_loss: 26.2128 - val_mae: 3.6828 - val_mse: 26.2128 - learning_rate: 2.5000e-04
Epoch 67/100
3/4 ━━━━━━━━━━━━━━━━━━━━ 0s 44ms/step - loss: 80.0882 - mae: 6.4822 - mse: 80.0882WARNING:absl:You are saving your model as an HDF5 file via `model.save()` or `keras.saving.save_model(model)`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')` or `keras.saving.save_model(model, 'my_model.keras')`. 
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 87ms/step - loss: 77.9514 - mae: 6.4265 - mse: 77.9514 - val_loss: 23.6246 - val_mae: 3.4861 - val_mse: 23.6246 - learning_rate: 2.5000e-04
Epoch 68/100
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 39ms/step - loss: 79.7832 - mae: 6.3684 - mse: 79.7832WARNING:absl:You are saving your model as an HDF5 file via `model.save()` or `keras.saving.save_model(model)`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')` or `keras.saving.save_model(model, 'my_model.keras')`. 
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 76ms/step - loss: 77.7038 - mae: 6.2967 - mse: 77.7038 - val_loss: 19.9462 - val_mae: 3.1899 - val_mse: 19.9462 - learning_rate: 2.5000e-04
Epoch 69/100
3/4 ━━━━━━━━━━━━━━━━━━━━ 0s 45ms/step - loss: 99.6033 - mae: 7.1147 - mse: 99.6033  WARNING:absl:You are saving your model as an HDF5 file via `model.save()` or `keras.saving.save_model(model)`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')` or `keras.saving.save_model(model, 'my_model.keras')`. 
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 74ms/step - loss: 96.4853 - mae: 6.9811 - mse: 96.4853 - val_loss: 17.5272 - val_mae: 2.9082 - val_mse: 17.5272 - learning_rate: 2.5000e-04
Epoch 70/100
3/4 ━━━━━━━━━━━━━━━━━━━━ 0s 54ms/step - loss: 89.6662 - mae: 6.3682 - mse: 89.6662WARNING:absl:You are saving your model as an HDF5 file via `model.save()` or `keras.saving.save_model(model)`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')` or `keras.saving.save_model(model, 'my_model.keras')`. 
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 79ms/step - loss: 83.8841 - mae: 6.2229 - mse: 83.8841 - val_loss: 15.2545 - val_mae: 2.7036 - val_mse: 15.2545 - learning_rate: 2.5000e-04
Epoch 71/100
3/4 ━━━━━━━━━━━━━━━━━━━━ 0s 47ms/step - loss: 77.4997 - mae: 6.2979 - mse: 77.4997WARNING:absl:You are saving your model as an HDF5 file via `model.save()` or `keras.saving.save_model(model)`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')` or `keras.saving.save_model(model, 'my_model.keras')`. 
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 79ms/step - loss: 73.6425 - mae: 6.1719 - mse: 73.6425 - val_loss: 15.2526 - val_mae: 2.7522 - val_mse: 15.2526 - learning_rate: 2.5000e-04
Epoch 72/100
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 80ms/step - loss: 46.3958 - mae: 5.0074 - mse: 46.3958 - val_loss: 16.1897 - val_mae: 2.9432 - val_mse: 16.1897 - learning_rate: 2.5000e-04
Epoch 73/100
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 109ms/step - loss: 66.4604 - mae: 6.0592 - mse: 66.4604 - val_loss: 17.1422 - val_mae: 3.0879 - val_mse: 17.1422 - learning_rate: 2.5000e-04
Epoch 74/100
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 99ms/step - loss: 62.2072 - mae: 5.9622 - mse: 62.2072 - val_loss: 17.7651 - val_mae: 3.1322 - val_mse: 17.7651 - learning_rate: 2.5000e-04
Epoch 75/100
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 63ms/step - loss: 69.2021 - mae: 5.9130 - mse: 69.2021 - val_loss: 18.2793 - val_mae: 3.0825 - val_mse: 18.2793 - learning_rate: 2.5000e-04
Epoch 76/100
3/4 ━━━━━━━━━━━━━━━━━━━━ 0s 46ms/step - loss: 74.5954 - mae: 6.4167 - mse: 74.5954
Epoch 76: ReduceLROnPlateau reducing learning rate to 0.0001250000059371814.
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 58ms/step - loss: 73.5415 - mae: 6.3471 - mse: 73.5415 - val_loss: 18.7528 - val_mae: 3.0077 - val_mse: 18.7528 - learning_rate: 2.5000e-04
Epoch 77/100
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 56ms/step - loss: 57.0173 - mae: 5.6201 - mse: 57.0173 - val_loss: 18.2571 - val_mae: 2.9207 - val_mse: 18.2571 - learning_rate: 1.2500e-04
Epoch 78/100
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 58ms/step - loss: 70.4298 - mae: 5.9875 - mse: 70.4298 - val_loss: 17.7268 - val_mae: 2.8923 - val_mse: 17.7268 - learning_rate: 1.2500e-04
Epoch 79/100
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 61ms/step - loss: 78.5292 - mae: 6.6081 - mse: 78.5292 - val_loss: 17.3819 - val_mae: 2.8902 - val_mse: 17.3819 - learning_rate: 1.2500e-04
Epoch 80/100
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 59ms/step - loss: 70.8318 - mae: 6.3355 - mse: 70.8318 - val_loss: 16.2737 - val_mae: 2.8105 - val_mse: 16.2737 - learning_rate: 1.2500e-04
Epoch 81/100
3/4 ━━━━━━━━━━━━━━━━━━━━ 0s 44ms/step - loss: 60.6420 - mae: 5.9444 - mse: 60.6420 WARNING:absl:You are saving your model as an HDF5 file via `model.save()` or `keras.saving.save_model(model)`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')` or `keras.saving.save_model(model, 'my_model.keras')`. 
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 75ms/step - loss: 61.8856 - mae: 5.9572 - mse: 61.8856 - val_loss: 15.0075 - val_mae: 2.6898 - val_mse: 15.0075 - learning_rate: 1.2500e-04
Epoch 82/100
3/4 ━━━━━━━━━━━━━━━━━━━━ 0s 44ms/step - loss: 74.4391 - mae: 6.2917 - mse: 74.4391WARNING:absl:You are saving your model as an HDF5 file via `model.save()` or `keras.saving.save_model(model)`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')` or `keras.saving.save_model(model, 'my_model.keras')`. 
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 75ms/step - loss: 72.0291 - mae: 6.1288 - mse: 72.0291 - val_loss: 13.9912 - val_mae: 2.5959 - val_mse: 13.9912 - learning_rate: 1.2500e-04
Epoch 83/100
3/4 ━━━━━━━━━━━━━━━━━━━━ 0s 43ms/step - loss: 82.2336 - mae: 6.4827 - mse: 82.2336WARNING:absl:You are saving your model as an HDF5 file via `model.save()` or `keras.saving.save_model(model)`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')` or `keras.saving.save_model(model, 'my_model.keras')`. 
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 77ms/step - loss: 83.0188 - mae: 6.5108 - mse: 83.0188 - val_loss: 13.6126 - val_mae: 2.5779 - val_mse: 13.6126 - learning_rate: 1.2500e-04
Epoch 84/100
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 55ms/step - loss: 68.6963 - mae: 5.9383 - mse: 68.6963 - val_loss: 13.7823 - val_mae: 2.6014 - val_mse: 13.7823 - learning_rate: 1.2500e-04
Epoch 85/100
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 57ms/step - loss: 70.6812 - mae: 6.2617 - mse: 70.6812 - val_loss: 13.7588 - val_mae: 2.6169 - val_mse: 13.7588 - learning_rate: 1.2500e-04
Epoch 86/100
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 57ms/step - loss: 72.5369 - mae: 6.4344 - mse: 72.5369 - val_loss: 13.8965 - val_mae: 2.6631 - val_mse: 13.8965 - learning_rate: 1.2500e-04
Epoch 87/100
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 56ms/step - loss: 85.7655 - mae: 6.6037 - mse: 85.7655 - val_loss: 13.8143 - val_mae: 2.7036 - val_mse: 13.8143 - learning_rate: 1.2500e-04
Epoch 88/100
3/4 ━━━━━━━━━━━━━━━━━━━━ 0s 46ms/step - loss: 65.7281 - mae: 6.1227 - mse: 65.7281WARNING:absl:You are saving your model as an HDF5 file via `model.save()` or `keras.saving.save_model(model)`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')` or `keras.saving.save_model(model, 'my_model.keras')`. 
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 80ms/step - loss: 64.3038 - mae: 5.9948 - mse: 64.3038 - val_loss: 13.1047 - val_mae: 2.7145 - val_mse: 13.1047 - learning_rate: 1.2500e-04
Epoch 89/100
3/4 ━━━━━━━━━━━━━━━━━━━━ 0s 44ms/step - loss: 68.1890 - mae: 6.3047 - mse: 68.1890WARNING:absl:You are saving your model as an HDF5 file via `model.save()` or `keras.saving.save_model(model)`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')` or `keras.saving.save_model(model, 'my_model.keras')`. 
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 75ms/step - loss: 64.8267 - mae: 6.1338 - mse: 64.8267 - val_loss: 12.5292 - val_mae: 2.7266 - val_mse: 12.5292 - learning_rate: 1.2500e-04
Epoch 90/100
3/4 ━━━━━━━━━━━━━━━━━━━━ 0s 46ms/step - loss: 69.1138 - mae: 5.9929 - mse: 69.1138WARNING:absl:You are saving your model as an HDF5 file via `model.save()` or `keras.saving.save_model(model)`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')` or `keras.saving.save_model(model, 'my_model.keras')`. 
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 81ms/step - loss: 69.0793 - mae: 5.9989 - mse: 69.0793 - val_loss: 12.4097 - val_mae: 2.7220 - val_mse: 12.4097 - learning_rate: 1.2500e-04
Epoch 91/100
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 59ms/step - loss: 76.0005 - mae: 6.4088 - mse: 76.0005 - val_loss: 12.4925 - val_mae: 2.7393 - val_mse: 12.4925 - learning_rate: 1.2500e-04
Epoch 92/100
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 57ms/step - loss: 70.4690 - mae: 5.9705 - mse: 70.4690 - val_loss: 12.8162 - val_mae: 2.7831 - val_mse: 12.8162 - learning_rate: 1.2500e-04
Epoch 93/100
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 58ms/step - loss: 75.2815 - mae: 6.2268 - mse: 75.2815 - val_loss: 13.8734 - val_mae: 2.8891 - val_mse: 13.8734 - learning_rate: 1.2500e-04
Epoch 94/100
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 62ms/step - loss: 67.8450 - mae: 5.9219 - mse: 67.8450 - val_loss: 14.3204 - val_mae: 2.9267 - val_mse: 14.3204 - learning_rate: 1.2500e-04
Epoch 95/100
3/4 ━━━━━━━━━━━━━━━━━━━━ 0s 46ms/step - loss: 64.6490 - mae: 5.8595 - mse: 64.6490
Epoch 95: ReduceLROnPlateau reducing learning rate to 6.25000029685907e-05.
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 58ms/step - loss: 64.1784 - mae: 5.8432 - mse: 64.1784 - val_loss: 15.8003 - val_mae: 3.0404 - val_mse: 15.8003 - learning_rate: 1.2500e-04
Epoch 96/100
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 58ms/step - loss: 67.1334 - mae: 5.9763 - mse: 67.1334 - val_loss: 15.8656 - val_mae: 3.0632 - val_mse: 15.8656 - learning_rate: 6.2500e-05
Epoch 97/100
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 56ms/step - loss: 95.4844 - mae: 6.7636 - mse: 95.4844 - val_loss: 15.4976 - val_mae: 3.0606 - val_mse: 15.4976 - learning_rate: 6.2500e-05
Epoch 98/100
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 60ms/step - loss: 73.8749 - mae: 6.4299 - mse: 73.8749 - val_loss: 15.0845 - val_mae: 3.0407 - val_mse: 15.0845 - learning_rate: 6.2500e-05
Epoch 99/100
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 56ms/step - loss: 61.7707 - mae: 5.9287 - mse: 61.7707 - val_loss: 14.6645 - val_mae: 3.0162 - val_mse: 14.6645 - learning_rate: 6.2500e-05
Epoch 100/100
3/4 ━━━━━━━━━━━━━━━━━━━━ 0s 45ms/step - loss: 62.5345 - mae: 5.5761 - mse: 62.5345 
Epoch 100: ReduceLROnPlateau reducing learning rate to 3.125000148429535e-05.
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 57ms/step - loss: 61.9646 - mae: 5.5500 - mse: 61.9646 - val_loss: 14.2978 - val_mae: 2.9879 - val_mse: 14.2978 - learning_rate: 6.2500e-05
Restoring model weights from the end of the best epoch: 90.

✓ Training completed

--- Training History Visualization ---


================================================================================
MILESTONE 2 COMPLETE ✓
================================================================================

================================================================================
MILESTONE 3: EVALUATION & PERFORMANCE ASSESSMENT
================================================================================

--- Generating Predictions on Test Set ---

--- Performance Metrics ---
Root Mean Square Error (RMSE): 3.6973
Mean Absolute Error (MAE):     3.1491
R² Score:                      0.9837

--- Model Interpretation ---
✓ Average prediction error: ±3.70 cycles
✓ Model explains 98.37% of variance in RUL
✓ Excellent performance: RMSE < 10 cycles

--- Actual vs Predicted RUL Visualization ---


--- Prediction Error Distribution ---


================================================================================
MILESTONE 3 COMPLETE ✓
================================================================================
