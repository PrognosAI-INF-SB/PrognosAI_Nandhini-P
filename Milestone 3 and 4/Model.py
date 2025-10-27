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
3/4 ━━━━━━━━━━━━━━━━━━━━ 0s 52ms/step - loss: 3372.5808 - mae: 51.0585 - mse: 3372.5808WARNING:absl:You are saving your model as an HDF5 file via `model.save()` or `keras.saving.save_model(model)`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')` or `keras.saving.save_model(model, 'my_model.keras')`. 
4/4 ━━━━━━━━━━━━━━━━━━━━ 8s 283ms/step - loss: 3318.6304 - mae: 50.5532 - mse: 3318.6304 - val_loss: 3259.4292 - val_mae: 48.3879 - val_mse: 3259.4292 - learning_rate: 0.0010
Epoch 2/100
3/4 ━━━━━━━━━━━━━━━━━━━━ 0s 54ms/step - loss: 3306.4019 - mae: 50.5780 - mse: 3306.4019WARNING:absl:You are saving your model as an HDF5 file via `model.save()` or `keras.saving.save_model(model)`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')` or `keras.saving.save_model(model, 'my_model.keras')`. 
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 86ms/step - loss: 3250.8677 - mae: 50.0565 - mse: 3250.8677 - val_loss: 3210.4685 - val_mae: 48.0284 - val_mse: 3210.4685 - learning_rate: 0.0010
Epoch 3/100
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 54ms/step - loss: 3163.8811 - mae: 49.4491 - mse: 3163.8811WARNING:absl:You are saving your model as an HDF5 file via `model.save()` or `keras.saving.save_model(model)`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')` or `keras.saving.save_model(model, 'my_model.keras')`. 
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 95ms/step - loss: 3142.8760 - mae: 49.2491 - mse: 3142.8760 - val_loss: 3091.6370 - val_mae: 47.1563 - val_mse: 3091.6370 - learning_rate: 0.0010
Epoch 4/100
3/4 ━━━━━━━━━━━━━━━━━━━━ 0s 46ms/step - loss: 3055.6111 - mae: 48.6973 - mse: 3055.6111WARNING:absl:You are saving your model as an HDF5 file via `model.save()` or `keras.saving.save_model(model)`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')` or `keras.saving.save_model(model, 'my_model.keras')`. 
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 79ms/step - loss: 2999.8218 - mae: 48.1534 - mse: 2999.8218 - val_loss: 2905.7261 - val_mae: 45.7112 - val_mse: 2905.7261 - learning_rate: 0.0010
Epoch 5/100
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 42ms/step - loss: 2844.3867 - mae: 46.8609 - mse: 2844.3867WARNING:absl:You are saving your model as an HDF5 file via `model.save()` or `keras.saving.save_model(model)`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')` or `keras.saving.save_model(model, 'my_model.keras')`. 
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 81ms/step - loss: 2824.5457 - mae: 46.6599 - mse: 2824.5457 - val_loss: 2678.7642 - val_mae: 43.8051 - val_mse: 2678.7642 - learning_rate: 0.0010
Epoch 6/100
3/4 ━━━━━━━━━━━━━━━━━━━━ 0s 45ms/step - loss: 2671.7114 - mae: 45.3783 - mse: 2671.7114WARNING:absl:You are saving your model as an HDF5 file via `model.save()` or `keras.saving.save_model(model)`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')` or `keras.saving.save_model(model, 'my_model.keras')`. 
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 80ms/step - loss: 2623.0703 - mae: 44.8764 - mse: 2623.0703 - val_loss: 2421.0952 - val_mae: 41.5046 - val_mse: 2421.0952 - learning_rate: 0.0010
Epoch 7/100
3/4 ━━━━━━━━━━━━━━━━━━━━ 0s 44ms/step - loss: 2407.6338 - mae: 42.9631 - mse: 2407.6338WARNING:absl:You are saving your model as an HDF5 file via `model.save()` or `keras.saving.save_model(model)`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')` or `keras.saving.save_model(model, 'my_model.keras')`. 
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 78ms/step - loss: 2357.5803 - mae: 42.4395 - mse: 2357.5803 - val_loss: 2128.2354 - val_mae: 38.7351 - val_mse: 2128.2354 - learning_rate: 0.0010
Epoch 8/100
3/4 ━━━━━━━━━━━━━━━━━━━━ 0s 46ms/step - loss: 2138.4368 - mae: 40.3552 - mse: 2138.4368WARNING:absl:You are saving your model as an HDF5 file via `model.save()` or `keras.saving.save_model(model)`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')` or `keras.saving.save_model(model, 'my_model.keras')`. 
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 76ms/step - loss: 2095.5679 - mae: 39.8910 - mse: 2095.5679 - val_loss: 1803.5769 - val_mae: 35.4809 - val_mse: 1803.5769 - learning_rate: 0.0010
Epoch 9/100
3/4 ━━━━━━━━━━━━━━━━━━━━ 0s 44ms/step - loss: 1813.4011 - mae: 37.0552 - mse: 1813.4011WARNING:absl:You are saving your model as an HDF5 file via `model.save()` or `keras.saving.save_model(model)`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')` or `keras.saving.save_model(model, 'my_model.keras')`. 
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 73ms/step - loss: 1770.8732 - mae: 36.5480 - mse: 1770.8732 - val_loss: 1441.6742 - val_mae: 31.4356 - val_mse: 1441.6742 - learning_rate: 0.0010
Epoch 10/100
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 44ms/step - loss: 1401.7056 - mae: 32.3343 - mse: 1401.7056WARNING:absl:You are saving your model as an HDF5 file via `model.save()` or `keras.saving.save_model(model)`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')` or `keras.saving.save_model(model, 'my_model.keras')`. 
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 81ms/step - loss: 1382.8358 - mae: 32.0600 - mse: 1382.8358 - val_loss: 1034.8727 - val_mae: 26.0583 - val_mse: 1034.8727 - learning_rate: 0.0010
Epoch 11/100
3/4 ━━━━━━━━━━━━━━━━━━━━ 0s 47ms/step - loss: 1042.7322 - mae: 26.6332 - mse: 1042.7322WARNING:absl:You are saving your model as an HDF5 file via `model.save()` or `keras.saving.save_model(model)`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')` or `keras.saving.save_model(model, 'my_model.keras')`. 
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 77ms/step - loss: 1001.9432 - mae: 25.9643 - mse: 1001.9432 - val_loss: 629.8895 - val_mae: 19.4102 - val_mse: 629.8895 - learning_rate: 0.0010
Epoch 12/100
3/4 ━━━━━━━━━━━━━━━━━━━━ 0s 53ms/step - loss: 658.4479 - mae: 19.9332 - mse: 658.4479WARNING:absl:You are saving your model as an HDF5 file via `model.save()` or `keras.saving.save_model(model)`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')` or `keras.saving.save_model(model, 'my_model.keras')`. 
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 79ms/step - loss: 628.8950 - mae: 19.3568 - mse: 628.8950 - val_loss: 531.9094 - val_mae: 19.5503 - val_mse: 531.9094 - learning_rate: 0.0010
Epoch 13/100
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 61ms/step - loss: 435.3026 - mae: 16.2501 - mse: 435.3026 - val_loss: 602.2176 - val_mae: 20.7853 - val_mse: 602.2176 - learning_rate: 0.0010
Epoch 14/100
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 42ms/step - loss: 392.4214 - mae: 15.8797 - mse: 392.4214WARNING:absl:You are saving your model as an HDF5 file via `model.save()` or `keras.saving.save_model(model)`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')` or `keras.saving.save_model(model, 'my_model.keras')`. 
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 81ms/step - loss: 393.9769 - mae: 15.9268 - mse: 393.9769 - val_loss: 501.6492 - val_mae: 18.5476 - val_mse: 501.6492 - learning_rate: 0.0010
Epoch 15/100
3/4 ━━━━━━━━━━━━━━━━━━━━ 0s 57ms/step - loss: 393.3213 - mae: 16.0367 - mse: 393.3213WARNING:absl:You are saving your model as an HDF5 file via `model.save()` or `keras.saving.save_model(model)`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')` or `keras.saving.save_model(model, 'my_model.keras')`. 
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 88ms/step - loss: 394.0666 - mae: 16.0792 - mse: 394.0666 - val_loss: 308.6184 - val_mae: 14.6661 - val_mse: 308.6184 - learning_rate: 0.0010
Epoch 16/100
3/4 ━━━━━━━━━━━━━━━━━━━━ 0s 56ms/step - loss: 368.4337 - mae: 15.5418 - mse: 368.4337WARNING:absl:You are saving your model as an HDF5 file via `model.save()` or `keras.saving.save_model(model)`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')` or `keras.saving.save_model(model, 'my_model.keras')`. 
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 92ms/step - loss: 371.9857 - mae: 15.5963 - mse: 371.9857 - val_loss: 244.5387 - val_mae: 12.6536 - val_mse: 244.5387 - learning_rate: 0.0010
Epoch 17/100
3/4 ━━━━━━━━━━━━━━━━━━━━ 0s 47ms/step - loss: 253.2345 - mae: 13.0417 - mse: 253.2345WARNING:absl:You are saving your model as an HDF5 file via `model.save()` or `keras.saving.save_model(model)`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')` or `keras.saving.save_model(model, 'my_model.keras')`. 
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 80ms/step - loss: 255.0530 - mae: 13.0159 - mse: 255.0530 - val_loss: 242.3783 - val_mae: 12.3462 - val_mse: 242.3783 - learning_rate: 0.0010
Epoch 18/100
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 62ms/step - loss: 316.8870 - mae: 13.8709 - mse: 316.8870 - val_loss: 288.0675 - val_mae: 13.0817 - val_mse: 288.0675 - learning_rate: 0.0010
Epoch 19/100
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 61ms/step - loss: 302.2948 - mae: 13.4949 - mse: 302.2948 - val_loss: 253.5944 - val_mae: 12.5165 - val_mse: 253.5944 - learning_rate: 0.0010
Epoch 20/100
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 71ms/step - loss: 240.4805 - mae: 11.7784 - mse: 240.4805 - val_loss: 266.7564 - val_mae: 13.2886 - val_mse: 266.7564 - learning_rate: 0.0010
Epoch 21/100
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 70ms/step - loss: 244.8221 - mae: 11.8419 - mse: 244.8221 - val_loss: 264.7570 - val_mae: 13.6168 - val_mse: 264.7570 - learning_rate: 0.0010
Epoch 22/100
3/4 ━━━━━━━━━━━━━━━━━━━━ 0s 55ms/step - loss: 218.5852 - mae: 10.9345 - mse: 218.5852
Epoch 22: ReduceLROnPlateau reducing learning rate to 0.0005000000237487257.
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 69ms/step - loss: 214.0573 - mae: 10.8751 - mse: 214.0573 - val_loss: 272.7716 - val_mae: 13.9242 - val_mse: 272.7716 - learning_rate: 0.0010
Epoch 23/100
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 65ms/step - loss: 190.6335 - mae: 10.4887 - mse: 190.6335 - val_loss: 278.1855 - val_mae: 14.1102 - val_mse: 278.1855 - learning_rate: 5.0000e-04
Epoch 24/100
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 66ms/step - loss: 148.0499 - mae: 9.2177 - mse: 148.0499 - val_loss: 276.9015 - val_mae: 14.1716 - val_mse: 276.9015 - learning_rate: 5.0000e-04
Epoch 25/100
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 74ms/step - loss: 158.9759 - mae: 9.5399 - mse: 158.9759 - val_loss: 277.4677 - val_mae: 14.3180 - val_mse: 277.4677 - learning_rate: 5.0000e-04
Epoch 26/100
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 70ms/step - loss: 132.6875 - mae: 9.0360 - mse: 132.6875 - val_loss: 307.9187 - val_mae: 15.3116 - val_mse: 307.9187 - learning_rate: 5.0000e-04
Epoch 27/100
3/4 ━━━━━━━━━━━━━━━━━━━━ 0s 64ms/step - loss: 202.1475 - mae: 10.4657 - mse: 202.1475
Epoch 27: ReduceLROnPlateau reducing learning rate to 0.0002500000118743628.
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 71ms/step - loss: 191.3264 - mae: 10.2027 - mse: 191.3264 - val_loss: 360.0536 - val_mae: 16.6937 - val_mse: 360.0536 - learning_rate: 5.0000e-04
Epoch 28/100
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 60ms/step - loss: 116.5194 - mae: 8.4562 - mse: 116.5194 - val_loss: 380.1475 - val_mae: 17.1406 - val_mse: 380.1475 - learning_rate: 2.5000e-04
Epoch 29/100
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 58ms/step - loss: 115.6707 - mae: 7.9819 - mse: 115.6707 - val_loss: 383.0670 - val_mae: 17.1970 - val_mse: 383.0670 - learning_rate: 2.5000e-04
Epoch 30/100
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 100ms/step - loss: 120.0121 - mae: 8.1786 - mse: 120.0121 - val_loss: 373.3069 - val_mae: 16.9718 - val_mse: 373.3069 - learning_rate: 2.5000e-04
Epoch 31/100
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 113ms/step - loss: 127.7489 - mae: 8.5507 - mse: 127.7489 - val_loss: 370.5124 - val_mae: 16.8931 - val_mse: 370.5124 - learning_rate: 2.5000e-04
Epoch 32/100
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 76ms/step - loss: 118.9357 - mae: 8.5752 - mse: 118.9357
Epoch 32: ReduceLROnPlateau reducing learning rate to 0.0001250000059371814.
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 115ms/step - loss: 118.5959 - mae: 8.5375 - mse: 118.5959 - val_loss: 364.7393 - val_mae: 16.7435 - val_mse: 364.7393 - learning_rate: 2.5000e-04
Epoch 32: early stopping
Restoring model weights from the end of the best epoch: 17.

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
Root Mean Square Error (RMSE): 14.4222
Mean Absolute Error (MAE):     10.8529
R² Score:                      0.7527

--- Model Interpretation ---
✓ Average prediction error: ±14.42 cycles
✓ Model explains 75.27% of variance in RUL
✓ Good performance: RMSE < 20 cycles

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
✓ Sample   1 | Predicted RUL:  59.33 | Actual RUL:  65.95 | Status: NORMAL  
✓ Sample   2 | Predicted RUL:  59.18 | Actual RUL:  65.60 | Status: NORMAL  
✓ Sample   3 | Predicted RUL:  50.29 | Actual RUL:  35.72 | Status: NORMAL  
✓ Sample   4 | Predicted RUL:  57.71 | Actual RUL:  54.83 | Status: NORMAL  
🚨 Sample   5 | Predicted RUL:  13.62 | Actual RUL:   7.23 | Status: CRITICAL
🚨 Sample   6 | Predicted RUL:  18.17 | Actual RUL:  16.96 | Status: CRITICAL
✓ Sample   7 | Predicted RUL:  59.44 | Actual RUL:  71.16 | Status: NORMAL  
✓ Sample   8 | Predicted RUL:  59.16 | Actual RUL:  94.79 | Status: NORMAL  
🚨 Sample   9 | Predicted RUL:  13.03 | Actual RUL:  10.70 | Status: CRITICAL
🚨 Sample  10 | Predicted RUL:  13.57 | Actual RUL:   6.18 | Status: CRITICAL
🚨 Sample  11 | Predicted RUL:  13.98 | Actual RUL:  20.43 | Status: CRITICAL
🚨 Sample  12 | Predicted RUL:  14.57 | Actual RUL:   4.79 | Status: CRITICAL
🚨 Sample  13 | Predicted RUL:  14.11 | Actual RUL:  11.05 | Status: CRITICAL
🚨 Sample  14 | Predicted RUL:  13.38 | Actual RUL:  23.91 | Status: CRITICAL
🚨 Sample  15 | Predicted RUL:  16.27 | Actual RUL:  19.39 | Status: CRITICAL
✓ Sample  16 | Predicted RUL:  59.26 | Actual RUL:  82.28 | Status: NORMAL  
✓ Sample  17 | Predicted RUL:  59.55 | Actual RUL:  83.32 | Status: NORMAL  
✓ Sample  18 | Predicted RUL:  49.84 | Actual RUL:  47.19 | Status: NORMAL  
✓ Sample  19 | Predicted RUL:  59.19 | Actual RUL:  86.45 | Status: NORMAL  
🚨 Sample  20 | Predicted RUL:  14.12 | Actual RUL:  13.13 | Status: CRITICAL

--- Alert Statistics Across Test Set ---
Total test samples: 43
  ✓ NORMAL:      26 (60.5%)
  ⚠ WARNING:      0 (0.0%)
  🚨 CRITICAL:    17 (39.5%)

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
- RMSE: 14.4222 cycles
- MAE: 10.8529 cycles
- R² Score: 0.7527

### Training Observations ###
- Total epochs trained: 32
- Final training loss: 117.236656
- Final validation loss: 364.739288
- Best validation loss: 242.378265

### Overfitting Analysis ###

✓ Slight underfitting: Model may benefit from increased capacity or training

### Alert System Performance ###
- Normal operations: 60.5% of predictions
- Warnings triggered: 0.0% of predictions
- Critical alerts: 39.5% of predictions

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
