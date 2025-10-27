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
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 205ms/step - loss: 3368.4937 - mae: 51.0246 - mse: 3368.4937WARNING:absl:You are saving your model as an HDF5 file via `model.save()` or `keras.saving.save_model(model)`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')` or `keras.saving.save_model(model, 'my_model.keras')`. 
4/4 ━━━━━━━━━━━━━━━━━━━━ 11s 969ms/step - loss: 3348.6748 - mae: 50.8394 - mse: 3348.6748 - val_loss: 3274.8748 - val_mae: 48.5194 - val_mse: 3274.8748 - learning_rate: 0.0010
Epoch 2/100
3/4 ━━━━━━━━━━━━━━━━━━━━ 0s 76ms/step - loss: 3375.5452 - mae: 51.1414 - mse: 3375.5452WARNING:absl:You are saving your model as an HDF5 file via `model.save()` or `keras.saving.save_model(model)`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')` or `keras.saving.save_model(model, 'my_model.keras')`. 
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 112ms/step - loss: 3323.0098 - mae: 50.6468 - mse: 3323.0098 - val_loss: 3263.7979 - val_mae: 48.4394 - val_mse: 3263.7979 - learning_rate: 0.0010
Epoch 3/100
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 72ms/step - loss: 3302.2729 - mae: 50.5246 - mse: 3302.2729WARNING:absl:You are saving your model as an HDF5 file via `model.save()` or `keras.saving.save_model(model)`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')` or `keras.saving.save_model(model, 'my_model.keras')`. 
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 123ms/step - loss: 3282.1016 - mae: 50.3353 - mse: 3282.1016 - val_loss: 3233.9219 - val_mae: 48.2253 - val_mse: 3233.9219 - learning_rate: 0.0010
Epoch 4/100
3/4 ━━━━━━━━━━━━━━━━━━━━ 0s 76ms/step - loss: 3264.7947 - mae: 50.3383 - mse: 3264.7947WARNING:absl:You are saving your model as an HDF5 file via `model.save()` or `keras.saving.save_model(model)`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')` or `keras.saving.save_model(model, 'my_model.keras')`. 
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 114ms/step - loss: 3211.0046 - mae: 49.8293 - mse: 3211.0046 - val_loss: 3162.7708 - val_mae: 47.7145 - val_mse: 3162.7708 - learning_rate: 0.0010
Epoch 5/100
3/4 ━━━━━━━━━━━━━━━━━━━━ 0s 88ms/step - loss: 3168.2266 - mae: 49.6020 - mse: 3168.2266WARNING:absl:You are saving your model as an HDF5 file via `model.save()` or `keras.saving.save_model(model)`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')` or `keras.saving.save_model(model, 'my_model.keras')`. 
4/4 ━━━━━━━━━━━━━━━━━━━━ 1s 121ms/step - loss: 3115.3843 - mae: 49.0945 - mse: 3115.3843 - val_loss: 3040.7319 - val_mae: 46.8225 - val_mse: 3040.7319 - learning_rate: 0.0010
Epoch 6/100
3/4 ━━━━━━━━━━━━━━━━━━━━ 0s 78ms/step - loss: 3052.7012 - mae: 48.7142 - mse: 3052.7012WARNING:absl:You are saving your model as an HDF5 file via `model.save()` or `keras.saving.save_model(model)`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')` or `keras.saving.save_model(model, 'my_model.keras')`. 
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 112ms/step - loss: 2999.6396 - mae: 48.2033 - mse: 2999.6396 - val_loss: 2871.7617 - val_mae: 45.5587 - val_mse: 2871.7617 - learning_rate: 0.0010
Epoch 7/100
3/4 ━━━━━━━━━━━━━━━━━━━━ 0s 75ms/step - loss: 2883.7476 - mae: 47.3327 - mse: 2883.7476WARNING:absl:You are saving your model as an HDF5 file via `model.save()` or `keras.saving.save_model(model)`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')` or `keras.saving.save_model(model, 'my_model.keras')`. 
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 115ms/step - loss: 2827.7087 - mae: 46.7717 - mse: 2827.7087 - val_loss: 2667.7822 - val_mae: 43.9800 - val_mse: 2667.7822 - learning_rate: 0.0010
Epoch 8/100
3/4 ━━━━━━━━━━━━━━━━━━━━ 0s 75ms/step - loss: 2676.4133 - mae: 45.3934 - mse: 2676.4133WARNING:absl:You are saving your model as an HDF5 file via `model.save()` or `keras.saving.save_model(model)`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')` or `keras.saving.save_model(model, 'my_model.keras')`. 
4/4 ━━━━━━━━━━━━━━━━━━━━ 1s 111ms/step - loss: 2624.0066 - mae: 44.8105 - mse: 2624.0066 - val_loss: 2421.8398 - val_mae: 41.6754 - val_mse: 2421.8398 - learning_rate: 0.0010
Epoch 9/100
3/4 ━━━━━━━━━━━━━━━━━━━━ 0s 83ms/step - loss: 2422.9675 - mae: 42.6488 - mse: 2422.9675WARNING:absl:You are saving your model as an HDF5 file via `model.save()` or `keras.saving.save_model(model)`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')` or `keras.saving.save_model(model, 'my_model.keras')`. 
4/4 ━━━━━━━━━━━━━━━━━━━━ 1s 116ms/step - loss: 2371.1809 - mae: 42.0686 - mse: 2371.1809 - val_loss: 2122.4958 - val_mae: 38.2111 - val_mse: 2122.4958 - learning_rate: 0.0010
Epoch 10/100
3/4 ━━━━━━━━━━━━━━━━━━━━ 0s 73ms/step - loss: 2136.1948 - mae: 39.5215 - mse: 2136.1948WARNING:absl:You are saving your model as an HDF5 file via `model.save()` or `keras.saving.save_model(model)`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')` or `keras.saving.save_model(model, 'my_model.keras')`. 
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 111ms/step - loss: 2084.3403 - mae: 38.9640 - mse: 2084.3403 - val_loss: 1793.3372 - val_mae: 34.5795 - val_mse: 1793.3372 - learning_rate: 0.0010
Epoch 11/100
3/4 ━━━━━━━━━━━━━━━━━━━━ 0s 85ms/step - loss: 1802.3835 - mae: 36.3357 - mse: 1802.3835WARNING:absl:You are saving your model as an HDF5 file via `model.save()` or `keras.saving.save_model(model)`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')` or `keras.saving.save_model(model, 'my_model.keras')`. 
4/4 ━━━━━━━━━━━━━━━━━━━━ 1s 119ms/step - loss: 1750.1884 - mae: 35.6645 - mse: 1750.1884 - val_loss: 1443.0712 - val_mae: 31.6972 - val_mse: 1443.0712 - learning_rate: 0.0010
Epoch 12/100
3/4 ━━━━━━━━━━━━━━━━━━━━ 0s 74ms/step - loss: 1429.0552 - mae: 32.3542 - mse: 1429.0552WARNING:absl:You are saving your model as an HDF5 file via `model.save()` or `keras.saving.save_model(model)`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')` or `keras.saving.save_model(model, 'my_model.keras')`. 
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 111ms/step - loss: 1381.0355 - mae: 31.6581 - mse: 1381.0355 - val_loss: 1083.9541 - val_mae: 27.8693 - val_mse: 1083.9541 - learning_rate: 0.0010
Epoch 13/100
3/4 ━━━━━━━━━━━━━━━━━━━━ 0s 83ms/step - loss: 1092.5753 - mae: 27.9839 - mse: 1092.5753WARNING:absl:You are saving your model as an HDF5 file via `model.save()` or `keras.saving.save_model(model)`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')` or `keras.saving.save_model(model, 'my_model.keras')`. 
4/4 ━━━━━━━━━━━━━━━━━━━━ 1s 117ms/step - loss: 1052.7798 - mae: 27.3717 - mse: 1052.7798 - val_loss: 764.9356 - val_mae: 23.8978 - val_mse: 764.9356 - learning_rate: 0.0010
Epoch 14/100
3/4 ━━━━━━━━━━━━━━━━━━━━ 0s 73ms/step - loss: 733.3956 - mae: 22.0013 - mse: 733.3956WARNING:absl:You are saving your model as an HDF5 file via `model.save()` or `keras.saving.save_model(model)`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')` or `keras.saving.save_model(model, 'my_model.keras')`. 
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 111ms/step - loss: 703.1109 - mae: 21.5118 - mse: 703.1109 - val_loss: 505.7966 - val_mae: 18.7173 - val_mse: 505.7966 - learning_rate: 0.0010
Epoch 15/100
3/4 ━━━━━━━━━━━━━━━━━━━━ 0s 74ms/step - loss: 547.3925 - mae: 18.5256 - mse: 547.3925WARNING:absl:You are saving your model as an HDF5 file via `model.save()` or `keras.saving.save_model(model)`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')` or `keras.saving.save_model(model, 'my_model.keras')`. 
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 116ms/step - loss: 526.9413 - mae: 18.1592 - mse: 526.9413 - val_loss: 351.0493 - val_mae: 15.5333 - val_mse: 351.0493 - learning_rate: 0.0010
Epoch 16/100
3/4 ━━━━━━━━━━━━━━━━━━━━ 0s 72ms/step - loss: 407.0808 - mae: 16.2695 - mse: 407.0808WARNING:absl:You are saving your model as an HDF5 file via `model.save()` or `keras.saving.save_model(model)`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')` or `keras.saving.save_model(model, 'my_model.keras')`. 
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 109ms/step - loss: 412.1309 - mae: 16.3987 - mse: 412.1309 - val_loss: 269.8106 - val_mae: 13.4959 - val_mse: 269.8106 - learning_rate: 0.0010
Epoch 17/100
3/4 ━━━━━━━━━━━━━━━━━━━━ 0s 74ms/step - loss: 376.4161 - mae: 15.8422 - mse: 376.4161WARNING:absl:You are saving your model as an HDF5 file via `model.save()` or `keras.saving.save_model(model)`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')` or `keras.saving.save_model(model, 'my_model.keras')`. 
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 110ms/step - loss: 377.4946 - mae: 15.8445 - mse: 377.4946 - val_loss: 253.8347 - val_mae: 12.9310 - val_mse: 253.8347 - learning_rate: 0.0010
Epoch 18/100
3/4 ━━━━━━━━━━━━━━━━━━━━ 0s 76ms/step - loss: 367.2194 - mae: 15.5049 - mse: 367.2194WARNING:absl:You are saving your model as an HDF5 file via `model.save()` or `keras.saving.save_model(model)`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')` or `keras.saving.save_model(model, 'my_model.keras')`. 
4/4 ━━━━━━━━━━━━━━━━━━━━ 1s 111ms/step - loss: 357.2366 - mae: 15.2526 - mse: 357.2366 - val_loss: 212.6765 - val_mae: 11.5680 - val_mse: 212.6765 - learning_rate: 0.0010
Epoch 19/100
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 87ms/step - loss: 303.9923 - mae: 13.6334 - mse: 303.9923 - val_loss: 352.3799 - val_mae: 14.6743 - val_mse: 352.3799 - learning_rate: 0.0010
Epoch 20/100
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 120ms/step - loss: 263.2786 - mae: 12.5940 - mse: 263.2786 - val_loss: 220.5526 - val_mae: 11.4931 - val_mse: 220.5526 - learning_rate: 0.0010
Epoch 21/100
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 115ms/step - loss: 214.2380 - mae: 11.2940 - mse: 214.2380WARNING:absl:You are saving your model as an HDF5 file via `model.save()` or `keras.saving.save_model(model)`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')` or `keras.saving.save_model(model, 'my_model.keras')`. 
4/4 ━━━━━━━━━━━━━━━━━━━━ 1s 198ms/step - loss: 210.4439 - mae: 11.1576 - mse: 210.4439 - val_loss: 179.6931 - val_mae: 10.8373 - val_mse: 179.6931 - learning_rate: 0.0010
Epoch 22/100
4/4 ━━━━━━━━━━━━━━━━━━━━ 1s 153ms/step - loss: 178.8874 - mae: 10.0749 - mse: 178.8874 - val_loss: 226.6385 - val_mae: 12.7729 - val_mse: 226.6385 - learning_rate: 0.0010
Epoch 23/100
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 89ms/step - loss: 146.2380 - mae: 9.1481 - mse: 146.2380 - val_loss: 256.4563 - val_mae: 13.4074 - val_mse: 256.4563 - learning_rate: 0.0010
Epoch 24/100
3/4 ━━━━━━━━━━━━━━━━━━━━ 0s 75ms/step - loss: 109.9539 - mae: 8.0247 - mse: 109.9539WARNING:absl:You are saving your model as an HDF5 file via `model.save()` or `keras.saving.save_model(model)`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')` or `keras.saving.save_model(model, 'my_model.keras')`. 
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 117ms/step - loss: 108.8840 - mae: 7.9734 - mse: 108.8840 - val_loss: 153.8434 - val_mae: 10.0544 - val_mse: 153.8434 - learning_rate: 0.0010
Epoch 25/100
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 87ms/step - loss: 109.6510 - mae: 7.8065 - mse: 109.6510 - val_loss: 154.5598 - val_mae: 10.1704 - val_mse: 154.5598 - learning_rate: 0.0010
Epoch 26/100
4/4 ━━━━━━━━━━━━━━━━━━━━ 1s 92ms/step - loss: 117.5422 - mae: 8.0570 - mse: 117.5422 - val_loss: 187.3091 - val_mae: 11.2070 - val_mse: 187.3091 - learning_rate: 0.0010
Epoch 27/100
4/4 ━━━━━━━━━━━━━━━━━━━━ 1s 87ms/step - loss: 109.4180 - mae: 7.9674 - mse: 109.4180 - val_loss: 237.0448 - val_mae: 12.6582 - val_mse: 237.0448 - learning_rate: 0.0010
Epoch 28/100
4/4 ━━━━━━━━━━━━━━━━━━━━ 1s 96ms/step - loss: 116.1933 - mae: 7.9199 - mse: 116.1933 - val_loss: 213.0631 - val_mae: 11.9301 - val_mse: 213.0631 - learning_rate: 0.0010
Epoch 29/100
3/4 ━━━━━━━━━━━━━━━━━━━━ 0s 73ms/step - loss: 121.3898 - mae: 8.1936 - mse: 121.3898
Epoch 29: ReduceLROnPlateau reducing learning rate to 0.0005000000237487257.
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 87ms/step - loss: 115.6666 - mae: 7.9422 - mse: 115.6666 - val_loss: 174.3529 - val_mae: 10.5513 - val_mse: 174.3529 - learning_rate: 0.0010
Epoch 30/100
3/4 ━━━━━━━━━━━━━━━━━━━━ 0s 74ms/step - loss: 99.7684 - mae: 7.2252 - mse: 99.7684  WARNING:absl:You are saving your model as an HDF5 file via `model.save()` or `keras.saving.save_model(model)`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')` or `keras.saving.save_model(model, 'my_model.keras')`. 
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 114ms/step - loss: 95.7812 - mae: 7.0775 - mse: 95.7812 - val_loss: 145.3355 - val_mae: 9.2765 - val_mse: 145.3355 - learning_rate: 5.0000e-04
Epoch 31/100
3/4 ━━━━━━━━━━━━━━━━━━━━ 0s 73ms/step - loss: 93.5528 - mae: 7.3327 - mse: 93.5528WARNING:absl:You are saving your model as an HDF5 file via `model.save()` or `keras.saving.save_model(model)`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')` or `keras.saving.save_model(model, 'my_model.keras')`. 
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 109ms/step - loss: 93.7290 - mae: 7.3147 - mse: 93.7290 - val_loss: 135.3251 - val_mae: 8.8676 - val_mse: 135.3251 - learning_rate: 5.0000e-04
Epoch 32/100
3/4 ━━━━━━━━━━━━━━━━━━━━ 0s 73ms/step - loss: 117.2411 - mae: 7.7787 - mse: 117.2411WARNING:absl:You are saving your model as an HDF5 file via `model.save()` or `keras.saving.save_model(model)`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')` or `keras.saving.save_model(model, 'my_model.keras')`. 
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 110ms/step - loss: 115.1069 - mae: 7.6924 - mse: 115.1069 - val_loss: 132.4581 - val_mae: 8.9242 - val_mse: 132.4581 - learning_rate: 5.0000e-04
Epoch 33/100
3/4 ━━━━━━━━━━━━━━━━━━━━ 0s 75ms/step - loss: 77.0943 - mae: 6.5034 - mse: 77.0943WARNING:absl:You are saving your model as an HDF5 file via `model.save()` or `keras.saving.save_model(model)`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')` or `keras.saving.save_model(model, 'my_model.keras')`. 
4/4 ━━━━━━━━━━━━━━━━━━━━ 1s 110ms/step - loss: 76.3565 - mae: 6.4889 - mse: 76.3565 - val_loss: 132.1996 - val_mae: 8.9480 - val_mse: 132.1996 - learning_rate: 5.0000e-04
Epoch 34/100
3/4 ━━━━━━━━━━━━━━━━━━━━ 0s 74ms/step - loss: 76.3277 - mae: 6.2543 - mse: 76.3277WARNING:absl:You are saving your model as an HDF5 file via `model.save()` or `keras.saving.save_model(model)`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')` or `keras.saving.save_model(model, 'my_model.keras')`. 
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 110ms/step - loss: 76.0214 - mae: 6.2833 - mse: 76.0214 - val_loss: 128.1989 - val_mae: 8.7924 - val_mse: 128.1989 - learning_rate: 5.0000e-04
Epoch 35/100
3/4 ━━━━━━━━━━━━━━━━━━━━ 0s 84ms/step - loss: 104.1748 - mae: 7.5311 - mse: 104.1748WARNING:absl:You are saving your model as an HDF5 file via `model.save()` or `keras.saving.save_model(model)`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')` or `keras.saving.save_model(model, 'my_model.keras')`. 
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 117ms/step - loss: 98.8981 - mae: 7.3208 - mse: 98.8981 - val_loss: 123.4099 - val_mae: 8.7754 - val_mse: 123.4099 - learning_rate: 5.0000e-04
Epoch 36/100
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 87ms/step - loss: 81.2389 - mae: 6.6009 - mse: 81.2389 - val_loss: 127.1112 - val_mae: 9.1512 - val_mse: 127.1112 - learning_rate: 5.0000e-04
Epoch 37/100
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 94ms/step - loss: 76.0313 - mae: 6.3165 - mse: 76.0313 - val_loss: 135.1292 - val_mae: 9.5667 - val_mse: 135.1292 - learning_rate: 5.0000e-04
Epoch 38/100
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 89ms/step - loss: 78.1182 - mae: 6.4795 - mse: 78.1182 - val_loss: 133.5944 - val_mae: 9.4176 - val_mse: 133.5944 - learning_rate: 5.0000e-04
Epoch 39/100
3/4 ━━━━━━━━━━━━━━━━━━━━ 0s 75ms/step - loss: 71.7015 - mae: 6.5531 - mse: 71.7015WARNING:absl:You are saving your model as an HDF5 file via `model.save()` or `keras.saving.save_model(model)`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')` or `keras.saving.save_model(model, 'my_model.keras')`. 
4/4 ━━━━━━━━━━━━━━━━━━━━ 1s 117ms/step - loss: 70.2199 - mae: 6.4634 - mse: 70.2199 - val_loss: 122.3156 - val_mae: 8.5975 - val_mse: 122.3156 - learning_rate: 5.0000e-04
Epoch 40/100
4/4 ━━━━━━━━━━━━━━━━━━━━ 1s 86ms/step - loss: 68.8467 - mae: 6.3786 - mse: 68.8467 - val_loss: 131.5210 - val_mae: 9.0361 - val_mse: 131.5210 - learning_rate: 5.0000e-04
Epoch 41/100
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 87ms/step - loss: 81.2037 - mae: 6.6056 - mse: 81.2037 - val_loss: 136.5180 - val_mae: 9.1865 - val_mse: 136.5180 - learning_rate: 5.0000e-04
Epoch 42/100
4/4 ━━━━━━━━━━━━━━━━━━━━ 1s 88ms/step - loss: 84.7730 - mae: 6.3999 - mse: 84.7730 - val_loss: 144.1130 - val_mae: 9.0376 - val_mse: 144.1130 - learning_rate: 5.0000e-04
Epoch 43/100
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 91ms/step - loss: 80.1999 - mae: 6.8468 - mse: 80.1999 - val_loss: 153.3587 - val_mae: 9.3994 - val_mse: 153.3587 - learning_rate: 5.0000e-04
Epoch 44/100
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 110ms/step - loss: 89.9296 - mae: 6.9118 - mse: 89.9296
Epoch 44: ReduceLROnPlateau reducing learning rate to 0.0002500000118743628.
4/4 ━━━━━━━━━━━━━━━━━━━━ 1s 150ms/step - loss: 89.6272 - mae: 6.8825 - mse: 89.6272 - val_loss: 160.7884 - val_mae: 9.7901 - val_mse: 160.7884 - learning_rate: 5.0000e-04
Epoch 45/100
4/4 ━━━━━━━━━━━━━━━━━━━━ 1s 161ms/step - loss: 99.3568 - mae: 7.6410 - mse: 99.3568 - val_loss: 152.7207 - val_mae: 9.2378 - val_mse: 152.7207 - learning_rate: 2.5000e-04
Epoch 46/100
4/4 ━━━━━━━━━━━━━━━━━━━━ 1s 152ms/step - loss: 95.1824 - mae: 7.0316 - mse: 95.1824 - val_loss: 143.6681 - val_mae: 9.1207 - val_mse: 143.6681 - learning_rate: 2.5000e-04
Epoch 47/100
4/4 ━━━━━━━━━━━━━━━━━━━━ 1s 151ms/step - loss: 66.4433 - mae: 6.3636 - mse: 66.4433 - val_loss: 133.7346 - val_mae: 9.1308 - val_mse: 133.7346 - learning_rate: 2.5000e-04
Epoch 48/100
3/4 ━━━━━━━━━━━━━━━━━━━━ 0s 85ms/step - loss: 79.3866 - mae: 6.7916 - mse: 79.3866WARNING:absl:You are saving your model as an HDF5 file via `model.save()` or `keras.saving.save_model(model)`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')` or `keras.saving.save_model(model, 'my_model.keras')`. 
4/4 ━━━━━━━━━━━━━━━━━━━━ 1s 123ms/step - loss: 76.8300 - mae: 6.6204 - mse: 76.8300 - val_loss: 121.8936 - val_mae: 8.7880 - val_mse: 121.8936 - learning_rate: 2.5000e-04
Epoch 49/100
3/4 ━━━━━━━━━━━━━━━━━━━━ 0s 75ms/step - loss: 56.6152 - mae: 5.7428 - mse: 56.6152WARNING:absl:You are saving your model as an HDF5 file via `model.save()` or `keras.saving.save_model(model)`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')` or `keras.saving.save_model(model, 'my_model.keras')`. 
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 109ms/step - loss: 56.7897 - mae: 5.7237 - mse: 56.7897 - val_loss: 115.9683 - val_mae: 8.7569 - val_mse: 115.9683 - learning_rate: 2.5000e-04
Epoch 50/100
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 93ms/step - loss: 82.1697 - mae: 6.6557 - mse: 82.1697 - val_loss: 117.1833 - val_mae: 8.8522 - val_mse: 117.1833 - learning_rate: 2.5000e-04
Epoch 51/100
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 89ms/step - loss: 62.4813 - mae: 6.1168 - mse: 62.4813 - val_loss: 117.7454 - val_mae: 8.9245 - val_mse: 117.7454 - learning_rate: 2.5000e-04
Epoch 52/100
3/4 ━━━━━━━━━━━━━━━━━━━━ 0s 76ms/step - loss: 74.2135 - mae: 6.3942 - mse: 74.2135WARNING:absl:You are saving your model as an HDF5 file via `model.save()` or `keras.saving.save_model(model)`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')` or `keras.saving.save_model(model, 'my_model.keras')`. 
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 115ms/step - loss: 74.1214 - mae: 6.3129 - mse: 74.1214 - val_loss: 108.2178 - val_mae: 8.5077 - val_mse: 108.2178 - learning_rate: 2.5000e-04
Epoch 53/100
3/4 ━━━━━━━━━━━━━━━━━━━━ 0s 86ms/step - loss: 67.0471 - mae: 6.3943 - mse: 67.0471WARNING:absl:You are saving your model as an HDF5 file via `model.save()` or `keras.saving.save_model(model)`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')` or `keras.saving.save_model(model, 'my_model.keras')`. 
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 118ms/step - loss: 67.2681 - mae: 6.2896 - mse: 67.2681 - val_loss: 85.6115 - val_mae: 7.3218 - val_mse: 85.6115 - learning_rate: 2.5000e-04
Epoch 54/100
3/4 ━━━━━━━━━━━━━━━━━━━━ 0s 74ms/step - loss: 71.2859 - mae: 6.3627 - mse: 71.2859WARNING:absl:You are saving your model as an HDF5 file via `model.save()` or `keras.saving.save_model(model)`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')` or `keras.saving.save_model(model, 'my_model.keras')`. 
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 111ms/step - loss: 73.8228 - mae: 6.4208 - mse: 73.8228 - val_loss: 73.6800 - val_mae: 6.7206 - val_mse: 73.6800 - learning_rate: 2.5000e-04
Epoch 55/100
3/4 ━━━━━━━━━━━━━━━━━━━━ 0s 85ms/step - loss: 60.9410 - mae: 5.8816 - mse: 60.9410WARNING:absl:You are saving your model as an HDF5 file via `model.save()` or `keras.saving.save_model(model)`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')` or `keras.saving.save_model(model, 'my_model.keras')`. 
4/4 ━━━━━━━━━━━━━━━━━━━━ 1s 119ms/step - loss: 64.1120 - mae: 6.0146 - mse: 64.1120 - val_loss: 65.3105 - val_mae: 6.3744 - val_mse: 65.3105 - learning_rate: 2.5000e-04
Epoch 56/100
3/4 ━━━━━━━━━━━━━━━━━━━━ 0s 76ms/step - loss: 53.0005 - mae: 5.6231 - mse: 53.0005WARNING:absl:You are saving your model as an HDF5 file via `model.save()` or `keras.saving.save_model(model)`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')` or `keras.saving.save_model(model, 'my_model.keras')`. 
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 111ms/step - loss: 54.5709 - mae: 5.6893 - mse: 54.5709 - val_loss: 63.2355 - val_mae: 6.4221 - val_mse: 63.2355 - learning_rate: 2.5000e-04
Epoch 57/100
4/4 ━━━━━━━━━━━━━━━━━━━━ 1s 86ms/step - loss: 66.8883 - mae: 6.1280 - mse: 66.8883 - val_loss: 65.8556 - val_mae: 6.6826 - val_mse: 65.8556 - learning_rate: 2.5000e-04
Epoch 58/100
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 86ms/step - loss: 68.8518 - mae: 6.1411 - mse: 68.8518 - val_loss: 66.1489 - val_mae: 6.7357 - val_mse: 66.1489 - learning_rate: 2.5000e-04
Epoch 59/100
3/4 ━━━━━━━━━━━━━━━━━━━━ 0s 89ms/step - loss: 57.5691 - mae: 5.8209 - mse: 57.5691 WARNING:absl:You are saving your model as an HDF5 file via `model.save()` or `keras.saving.save_model(model)`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')` or `keras.saving.save_model(model, 'my_model.keras')`. 
4/4 ━━━━━━━━━━━━━━━━━━━━ 1s 120ms/step - loss: 58.8915 - mae: 5.8020 - mse: 58.8915 - val_loss: 57.4185 - val_mae: 6.2299 - val_mse: 57.4185 - learning_rate: 2.5000e-04
Epoch 60/100
3/4 ━━━━━━━━━━━━━━━━━━━━ 0s 74ms/step - loss: 51.1739 - mae: 5.5181 - mse: 51.1739WARNING:absl:You are saving your model as an HDF5 file via `model.save()` or `keras.saving.save_model(model)`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')` or `keras.saving.save_model(model, 'my_model.keras')`. 
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 112ms/step - loss: 53.2278 - mae: 5.5909 - mse: 53.2278 - val_loss: 51.8838 - val_mae: 5.8610 - val_mse: 51.8838 - learning_rate: 2.5000e-04
Epoch 61/100
3/4 ━━━━━━━━━━━━━━━━━━━━ 0s 73ms/step - loss: 62.7773 - mae: 5.8709 - mse: 62.7773WARNING:absl:You are saving your model as an HDF5 file via `model.save()` or `keras.saving.save_model(model)`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')` or `keras.saving.save_model(model, 'my_model.keras')`. 
4/4 ━━━━━━━━━━━━━━━━━━━━ 1s 113ms/step - loss: 61.0214 - mae: 5.7717 - mse: 61.0214 - val_loss: 44.6900 - val_mae: 5.3141 - val_mse: 44.6900 - learning_rate: 2.5000e-04
Epoch 62/100
3/4 ━━━━━━━━━━━━━━━━━━━━ 0s 76ms/step - loss: 63.6997 - mae: 5.9588 - mse: 63.6997WARNING:absl:You are saving your model as an HDF5 file via `model.save()` or `keras.saving.save_model(model)`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')` or `keras.saving.save_model(model, 'my_model.keras')`. 
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 112ms/step - loss: 65.1194 - mae: 6.0160 - mse: 65.1194 - val_loss: 40.3752 - val_mae: 4.6674 - val_mse: 40.3752 - learning_rate: 2.5000e-04
Epoch 63/100
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 97ms/step - loss: 73.6710 - mae: 6.4072 - mse: 73.6710 - val_loss: 44.0737 - val_mae: 5.0944 - val_mse: 44.0737 - learning_rate: 2.5000e-04
Epoch 64/100
3/4 ━━━━━━━━━━━━━━━━━━━━ 0s 75ms/step - loss: 76.0696 - mae: 6.6879 - mse: 76.0696WARNING:absl:You are saving your model as an HDF5 file via `model.save()` or `keras.saving.save_model(model)`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')` or `keras.saving.save_model(model, 'my_model.keras')`. 
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 111ms/step - loss: 75.4506 - mae: 6.5914 - mse: 75.4506 - val_loss: 37.5348 - val_mae: 4.7441 - val_mse: 37.5348 - learning_rate: 2.5000e-04
Epoch 65/100
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 69ms/step - loss: 68.9251 - mae: 6.5630 - mse: 68.9251WARNING:absl:You are saving your model as an HDF5 file via `model.save()` or `keras.saving.save_model(model)`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')` or `keras.saving.save_model(model, 'my_model.keras')`. 
4/4 ━━━━━━━━━━━━━━━━━━━━ 1s 122ms/step - loss: 69.3458 - mae: 6.5739 - mse: 69.3458 - val_loss: 28.9200 - val_mae: 4.1490 - val_mse: 28.9200 - learning_rate: 2.5000e-04
Epoch 66/100
3/4 ━━━━━━━━━━━━━━━━━━━━ 0s 74ms/step - loss: 77.1876 - mae: 6.5251 - mse: 77.1876WARNING:absl:You are saving your model as an HDF5 file via `model.save()` or `keras.saving.save_model(model)`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')` or `keras.saving.save_model(model, 'my_model.keras')`. 
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 112ms/step - loss: 74.4225 - mae: 6.3810 - mse: 74.4225 - val_loss: 27.1874 - val_mae: 4.1208 - val_mse: 27.1874 - learning_rate: 2.5000e-04
Epoch 67/100
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 104ms/step - loss: 68.7261 - mae: 6.2348 - mse: 68.7261 - val_loss: 30.1304 - val_mae: 4.5367 - val_mse: 30.1304 - learning_rate: 2.5000e-04
Epoch 68/100
4/4 ━━━━━━━━━━━━━━━━━━━━ 1s 154ms/step - loss: 75.8518 - mae: 6.7245 - mse: 75.8518 - val_loss: 30.8251 - val_mae: 4.6184 - val_mse: 30.8251 - learning_rate: 2.5000e-04
Epoch 69/100
4/4 ━━━━━━━━━━━━━━━━━━━━ 1s 157ms/step - loss: 76.6053 - mae: 6.7508 - mse: 76.6053 - val_loss: 28.4302 - val_mae: 4.4056 - val_mse: 28.4302 - learning_rate: 2.5000e-04
Epoch 70/100
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 112ms/step - loss: 55.6560 - mae: 5.6211 - mse: 55.6560WARNING:absl:You are saving your model as an HDF5 file via `model.save()` or `keras.saving.save_model(model)`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')` or `keras.saving.save_model(model, 'my_model.keras')`. 
4/4 ━━━━━━━━━━━━━━━━━━━━ 1s 195ms/step - loss: 54.2668 - mae: 5.5322 - mse: 54.2668 - val_loss: 25.1314 - val_mae: 4.1761 - val_mse: 25.1314 - learning_rate: 2.5000e-04
Epoch 71/100
3/4 ━━━━━━━━━━━━━━━━━━━━ 0s 125ms/step - loss: 78.2892 - mae: 6.7553 - mse: 78.2892WARNING:absl:You are saving your model as an HDF5 file via `model.save()` or `keras.saving.save_model(model)`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')` or `keras.saving.save_model(model, 'my_model.keras')`. 
4/4 ━━━━━━━━━━━━━━━━━━━━ 1s 150ms/step - loss: 76.6045 - mae: 6.6320 - mse: 76.6045 - val_loss: 24.9423 - val_mae: 4.1671 - val_mse: 24.9423 - learning_rate: 2.5000e-04
Epoch 72/100
3/4 ━━━━━━━━━━━━━━━━━━━━ 0s 76ms/step - loss: 74.5177 - mae: 6.4348 - mse: 74.5177WARNING:absl:You are saving your model as an HDF5 file via `model.save()` or `keras.saving.save_model(model)`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')` or `keras.saving.save_model(model, 'my_model.keras')`. 
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 122ms/step - loss: 72.4509 - mae: 6.3134 - mse: 72.4509 - val_loss: 24.7005 - val_mae: 4.1479 - val_mse: 24.7005 - learning_rate: 2.5000e-04
Epoch 73/100
3/4 ━━━━━━━━━━━━━━━━━━━━ 0s 78ms/step - loss: 50.6599 - mae: 5.6401 - mse: 50.6599WARNING:absl:You are saving your model as an HDF5 file via `model.save()` or `keras.saving.save_model(model)`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')` or `keras.saving.save_model(model, 'my_model.keras')`. 
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 115ms/step - loss: 52.0413 - mae: 5.6144 - mse: 52.0413 - val_loss: 24.6409 - val_mae: 4.1278 - val_mse: 24.6409 - learning_rate: 2.5000e-04
Epoch 74/100
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 90ms/step - loss: 67.5977 - mae: 5.9717 - mse: 67.5977 - val_loss: 25.4103 - val_mae: 4.1861 - val_mse: 25.4103 - learning_rate: 2.5000e-04
Epoch 75/100
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 100ms/step - loss: 49.8602 - mae: 5.3041 - mse: 49.8602 - val_loss: 26.4033 - val_mae: 4.2999 - val_mse: 26.4033 - learning_rate: 2.5000e-04
Epoch 76/100
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 89ms/step - loss: 73.4788 - mae: 6.2322 - mse: 73.4788 - val_loss: 29.0724 - val_mae: 4.5352 - val_mse: 29.0724 - learning_rate: 2.5000e-04
Epoch 77/100
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 90ms/step - loss: 73.4478 - mae: 6.2701 - mse: 73.4478 - val_loss: 29.6149 - val_mae: 4.6387 - val_mse: 29.6149 - learning_rate: 2.5000e-04
Epoch 78/100
3/4 ━━━━━━━━━━━━━━━━━━━━ 0s 81ms/step - loss: 71.1949 - mae: 6.8195 - mse: 71.1949
Epoch 78: ReduceLROnPlateau reducing learning rate to 0.0001250000059371814.
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 93ms/step - loss: 73.2638 - mae: 6.8928 - mse: 73.2638 - val_loss: 26.4867 - val_mae: 4.4325 - val_mse: 26.4867 - learning_rate: 2.5000e-04
Epoch 79/100
3/4 ━━━━━━━━━━━━━━━━━━━━ 0s 78ms/step - loss: 76.3868 - mae: 6.3890 - mse: 76.3868WARNING:absl:You are saving your model as an HDF5 file via `model.save()` or `keras.saving.save_model(model)`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')` or `keras.saving.save_model(model, 'my_model.keras')`. 
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 114ms/step - loss: 75.2027 - mae: 6.3169 - mse: 75.2027 - val_loss: 22.6652 - val_mae: 4.1033 - val_mse: 22.6652 - learning_rate: 1.2500e-04
Epoch 80/100
3/4 ━━━━━━━━━━━━━━━━━━━━ 0s 75ms/step - loss: 59.4678 - mae: 5.7172 - mse: 59.4678WARNING:absl:You are saving your model as an HDF5 file via `model.save()` or `keras.saving.save_model(model)`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')` or `keras.saving.save_model(model, 'my_model.keras')`. 
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 118ms/step - loss: 59.2375 - mae: 5.6834 - mse: 59.2375 - val_loss: 21.1068 - val_mae: 3.9467 - val_mse: 21.1068 - learning_rate: 1.2500e-04
Epoch 81/100
3/4 ━━━━━━━━━━━━━━━━━━━━ 0s 75ms/step - loss: 59.0580 - mae: 5.7921 - mse: 59.0580WARNING:absl:You are saving your model as an HDF5 file via `model.save()` or `keras.saving.save_model(model)`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')` or `keras.saving.save_model(model, 'my_model.keras')`. 
4/4 ━━━━━━━━━━━━━━━━━━━━ 1s 113ms/step - loss: 60.1048 - mae: 5.8196 - mse: 60.1048 - val_loss: 19.6204 - val_mae: 3.7785 - val_mse: 19.6204 - learning_rate: 1.2500e-04
Epoch 82/100
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 71ms/step - loss: 46.8657 - mae: 5.1892 - mse: 46.8657WARNING:absl:You are saving your model as an HDF5 file via `model.save()` or `keras.saving.save_model(model)`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')` or `keras.saving.save_model(model, 'my_model.keras')`. 
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 122ms/step - loss: 47.4602 - mae: 5.1995 - mse: 47.4602 - val_loss: 18.6244 - val_mae: 3.6681 - val_mse: 18.6244 - learning_rate: 1.2500e-04
Epoch 83/100
3/4 ━━━━━━━━━━━━━━━━━━━━ 0s 78ms/step - loss: 61.2407 - mae: 5.8056 - mse: 61.2407WARNING:absl:You are saving your model as an HDF5 file via `model.save()` or `keras.saving.save_model(model)`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')` or `keras.saving.save_model(model, 'my_model.keras')`. 
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 117ms/step - loss: 61.3152 - mae: 5.8505 - mse: 61.3152 - val_loss: 16.2530 - val_mae: 3.3962 - val_mse: 16.2530 - learning_rate: 1.2500e-04
Epoch 84/100
3/4 ━━━━━━━━━━━━━━━━━━━━ 0s 74ms/step - loss: 61.1954 - mae: 6.1369 - mse: 61.1954WARNING:absl:You are saving your model as an HDF5 file via `model.save()` or `keras.saving.save_model(model)`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')` or `keras.saving.save_model(model, 'my_model.keras')`. 
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 118ms/step - loss: 61.9079 - mae: 6.1377 - mse: 61.9079 - val_loss: 14.4459 - val_mae: 3.1344 - val_mse: 14.4459 - learning_rate: 1.2500e-04
Epoch 85/100
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 88ms/step - loss: 79.8665 - mae: 6.5447 - mse: 79.8665 - val_loss: 14.4675 - val_mae: 3.0966 - val_mse: 14.4675 - learning_rate: 1.2500e-04
Epoch 86/100
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 86ms/step - loss: 57.5592 - mae: 5.7066 - mse: 57.5592 - val_loss: 15.7864 - val_mae: 3.2273 - val_mse: 15.7864 - learning_rate: 1.2500e-04
Epoch 87/100
3/4 ━━━━━━━━━━━━━━━━━━━━ 0s 78ms/step - loss: 61.0380 - mae: 5.9238 - mse: 61.0380WARNING:absl:You are saving your model as an HDF5 file via `model.save()` or `keras.saving.save_model(model)`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')` or `keras.saving.save_model(model, 'my_model.keras')`. 
4/4 ━━━━━━━━━━━━━━━━━━━━ 1s 115ms/step - loss: 62.2054 - mae: 5.9865 - mse: 62.2054 - val_loss: 14.3773 - val_mae: 3.0524 - val_mse: 14.3773 - learning_rate: 1.2500e-04
Epoch 88/100
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 87ms/step - loss: 53.7230 - mae: 5.4622 - mse: 53.7230 - val_loss: 14.4246 - val_mae: 3.0564 - val_mse: 14.4246 - learning_rate: 1.2500e-04
Epoch 89/100
4/4 ━━━━━━━━━━━━━━━━━━━━ 1s 87ms/step - loss: 59.3412 - mae: 5.7955 - mse: 59.3412 - val_loss: 15.6281 - val_mae: 3.2014 - val_mse: 15.6281 - learning_rate: 1.2500e-04
Epoch 90/100
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 87ms/step - loss: 66.9281 - mae: 5.7073 - mse: 66.9281 - val_loss: 15.7786 - val_mae: 3.2145 - val_mse: 15.7786 - learning_rate: 1.2500e-04
Epoch 91/100
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 97ms/step - loss: 78.0088 - mae: 6.3139 - mse: 78.0088 - val_loss: 16.3533 - val_mae: 3.2822 - val_mse: 16.3533 - learning_rate: 1.2500e-04
Epoch 92/100
3/4 ━━━━━━━━━━━━━━━━━━━━ 0s 78ms/step - loss: 65.2649 - mae: 6.0148 - mse: 65.2649
Epoch 92: ReduceLROnPlateau reducing learning rate to 6.25000029685907e-05.
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 90ms/step - loss: 65.4757 - mae: 6.0056 - mse: 65.4757 - val_loss: 16.8005 - val_mae: 3.3419 - val_mse: 16.8005 - learning_rate: 1.2500e-04
Epoch 93/100
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 87ms/step - loss: 55.7273 - mae: 5.6622 - mse: 55.7273 - val_loss: 16.4864 - val_mae: 3.3084 - val_mse: 16.4864 - learning_rate: 6.2500e-05
Epoch 94/100
4/4 ━━━━━━━━━━━━━━━━━━━━ 1s 148ms/step - loss: 76.3839 - mae: 6.2528 - mse: 76.3839 - val_loss: 15.5611 - val_mae: 3.2185 - val_mse: 15.5611 - learning_rate: 6.2500e-05
Epoch 95/100
4/4 ━━━━━━━━━━━━━━━━━━━━ 1s 159ms/step - loss: 51.4293 - mae: 5.5282 - mse: 51.4293 - val_loss: 15.2964 - val_mae: 3.2046 - val_mse: 15.2964 - learning_rate: 6.2500e-05
Epoch 96/100
4/4 ━━━━━━━━━━━━━━━━━━━━ 1s 121ms/step - loss: 64.3465 - mae: 6.0590 - mse: 64.3465 - val_loss: 15.0835 - val_mae: 3.1902 - val_mse: 15.0835 - learning_rate: 6.2500e-05
Epoch 97/100
3/4 ━━━━━━━━━━━━━━━━━━━━ 0s 75ms/step - loss: 57.5959 - mae: 5.7353 - mse: 57.5959
Epoch 97: ReduceLROnPlateau reducing learning rate to 3.125000148429535e-05.
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 89ms/step - loss: 59.3749 - mae: 5.8308 - mse: 59.3749 - val_loss: 14.6678 - val_mae: 3.1654 - val_mse: 14.6678 - learning_rate: 6.2500e-05
Epoch 98/100
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 99ms/step - loss: 59.3472 - mae: 5.8523 - mse: 59.3472 - val_loss: 14.7098 - val_mae: 3.1786 - val_mse: 14.7098 - learning_rate: 3.1250e-05
Epoch 99/100
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 88ms/step - loss: 65.1084 - mae: 6.1716 - mse: 65.1084 - val_loss: 15.0206 - val_mae: 3.2140 - val_mse: 15.0206 - learning_rate: 3.1250e-05
Epoch 100/100
4/4 ━━━━━━━━━━━━━━━━━━━━ 1s 100ms/step - loss: 65.8362 - mae: 5.9940 - mse: 65.8362 - val_loss: 15.6920 - val_mae: 3.2900 - val_mse: 15.6920 - learning_rate: 3.1250e-05
Restoring model weights from the end of the best epoch: 87.

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
Root Mean Square Error (RMSE): 3.6269
Mean Absolute Error (MAE):     3.2225
R² Score:                      0.9844

--- Model Interpretation ---
✓ Average prediction error: ±3.63 cycles
✓ Model explains 98.44% of variance in RUL
✓ Excellent performance: RMSE < 10 cycles

--- Actual vs Predicted RUL Visualization ---


--- Prediction Error Distribution ---


================================================================================
MILESTONE 3 COMPLETE ✓
================================================================================

================================================================================
MILESTONE 4: RISK THRESHOLDING & MAINTENANCE ALERTS
================================================================================

--- Alert Thresholds Defined ---
⚠ WARNING:  RUL < 40 cycles
🚨 CRITICAL: RUL < 20 cycles

--- Sample Maintenance Alerts (First 20 Test Samples) ---
✓ Sample   1 | Predicted RUL:  68.84 | Actual RUL:  65.95 | Status: NORMAL  
✓ Sample   2 | Predicted RUL:  68.35 | Actual RUL:  65.60 | Status: NORMAL  
⚠ Sample   3 | Predicted RUL:  34.00 | Actual RUL:  35.72 | Status: WARNING 
✓ Sample   4 | Predicted RUL:  57.75 | Actual RUL:  54.83 | Status: NORMAL  
🚨 Sample   5 | Predicted RUL:   3.90 | Actual RUL:   7.23 | Status: CRITICAL
⚠ Sample   6 | Predicted RUL:  23.15 | Actual RUL:  16.96 | Status: WARNING 
✓ Sample   7 | Predicted RUL:  73.09 | Actual RUL:  71.16 | Status: NORMAL  
✓ Sample   8 | Predicted RUL:  88.06 | Actual RUL:  94.79 | Status: NORMAL  
🚨 Sample   9 | Predicted RUL:   8.09 | Actual RUL:  10.70 | Status: CRITICAL
🚨 Sample  10 | Predicted RUL:   4.19 | Actual RUL:   6.18 | Status: CRITICAL
🚨 Sample  11 | Predicted RUL:  18.10 | Actual RUL:  20.43 | Status: CRITICAL
🚨 Sample  12 | Predicted RUL:   5.40 | Actual RUL:   4.79 | Status: CRITICAL
🚨 Sample  13 | Predicted RUL:   9.13 | Actual RUL:  11.05 | Status: CRITICAL
🚨 Sample  14 | Predicted RUL:  17.57 | Actual RUL:  23.91 | Status: CRITICAL
⚠ Sample  15 | Predicted RUL:  23.49 | Actual RUL:  19.39 | Status: WARNING 
✓ Sample  16 | Predicted RUL:  87.84 | Actual RUL:  82.28 | Status: NORMAL  
✓ Sample  17 | Predicted RUL:  88.11 | Actual RUL:  83.32 | Status: NORMAL  
✓ Sample  18 | Predicted RUL:  44.40 | Actual RUL:  47.19 | Status: NORMAL  
✓ Sample  19 | Predicted RUL:  88.02 | Actual RUL:  86.45 | Status: NORMAL  
🚨 Sample  20 | Predicted RUL:   9.85 | Actual RUL:  13.13 | Status: CRITICAL

--- Alert Statistics Across Test Set ---
Total test samples: 43
  ✓ NORMAL:      24 (55.8%)
  ⚠ WARNING:      6 (14.0%)
  🚨 CRITICAL:    13 (30.2%)

--- Alert Level Distribution ---


--- RUL Distribution with Alert Thresholds ---


================================================================================
MILESTONE 4 COMPLETE ✓
================================================================================

================================================================================
FINAL PROJECT SUMMARY & RECOMMENDATIONS
================================================================================

### Model Performance Summary ###
- Architecture: Hybrid LSTM (Conv1D + Bidirectional LSTM)
- RMSE: 3.6269 cycles
- MAE: 3.2225 cycles
- R² Score: 0.9844

### Training Observations ###
- Total epochs trained: 100
- Final training loss: 63.975327
- Final validation loss: 15.691985
- Best validation loss: 14.377288

### Overfitting Analysis ###

⚠ Potential overfitting detected: Validation loss higher than training loss

### Alert System Performance ###
- Normal operations: 55.8% of predictions
- Warnings triggered: 14.0% of predictions
- Critical alerts: 30.2% of predictions

### Improvement Suggestions ###
1. **Data Enhancement:**
   - Collect more labeled failure data for better RUL calibration
   - Add domain-specific features (operating conditions, maintenance history)

2. **Model Architecture:**
   - Experiment with attention mechanisms for better temporal focus
   - Try ensemble methods combining multiple model predictions

3. **Training Optimization:**
   - Implement custom loss functions weighted by proximity to failure
   - Use data augmentation (jittering, warping) for robustness

4. **Alert System:**
   - Implement adaptive thresholds based on equipment criticality
   - Add confidence intervals to predictions for uncertainty quantification

5. **Deployment:**
   - Set up real-time inference pipeline for continuous monitoring
   - Integrate with maintenance scheduling systems
   - Implement model retraining on new failure data

### Next Steps ###
- Validate model on held-out equipment or operational conditions
- Conduct A/B testing comparing predictions with actual maintenance outcomes
- Deploy as edge or cloud service for production monitoring

================================================================================
AI-PrognosAI PROJECT COMPLETE ✓
All milestones successfully executed!
================================================================================
