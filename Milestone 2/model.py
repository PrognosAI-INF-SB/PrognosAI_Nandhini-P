Loading dataset...
Dataset loaded with 1440 rows and columns: ['time', 'SensorA', 'SensorB', 'SensorC']
Total missing values before imputation: 420
Missing values imputed using median strategy.
Sensor columns inferred for feature engineering: ['SensorA', 'SensorB', 'SensorC']
Features scaled (StandardScaler).
RUL calculated with cap=125
Sample RUL values (first 5): [125, 125, 125, 125, 125]
Generating rolling windows with size=50, stride=1...
Generated 1391 sequences.

Sample Sequence Feature Window (first sequence):
[[-4.3233805e+00  3.8642612e-01 -2.3210700e+00]
 [-3.2304306e+00 -1.6163524e-03 -2.3570137e+00]
 [-2.1338146e+00 -1.6163524e-03 -2.4229105e+00]
 [-2.1439102e+00 -5.4408230e-02 -1.2547402e+00]
 [-2.7557077e+00 -6.3881570e-01 -3.3122945e+00]
 [-2.0202491e+00 -1.1968350e+00 -1.5303086e+00]
 [ 7.4855159e-03 -8.7450939e-01 -1.3445995e+00]
 [-1.8535072e+00 -1.2267388e+00 -1.5782336e+00]
 [-3.0670936e+00 -5.9520453e-01 -1.1229466e+00]
 [-3.0112424e+00 -2.8907794e-01 -9.4322813e-01]
 [-1.2606572e+00 -3.2669794e-02 -1.4464400e+00]
 [-2.5893354e+00  3.1039611e-01 -1.9855955e+00]
 [-3.3317080e+00 -1.6163524e-03 -1.9706190e+00]
 [-1.8728864e+00  5.9088999e-01 -2.4049387e+00]
 [ 7.4855159e-03  6.2427975e-02 -2.2042530e+00]
 [-2.8375630e+00 -1.5872130e-01 -2.4408824e+00]
 [-1.2925842e+00 -1.0870755e+00 -2.3869669e+00]
 [-7.4874842e-01 -8.5939300e-01 -1.8627878e+00]
 [-2.3381653e+00 -1.0201924e+00 -1.6650975e+00]
 [-1.9871308e-01 -1.6163524e-03  7.2181448e-02]
 [-1.0083452e+00 -9.8903352e-01 -1.1169560e+00]
 [-1.9986365e+00 -4.6226776e-01  7.2181448e-02]
 [-2.4435103e+00 -1.6163524e-03 -1.2846934e+00]
 [-1.3720876e+00  4.6727875e-01 -1.2487496e+00]
 [-1.2723868e+00  5.4697943e-01 -1.5093415e+00]
 [-1.4036456e+00  3.9274085e-01 -1.5273134e+00]
 [ 8.7027347e-01  1.4513555e-01 -1.7609473e+00]
 [-6.8010205e-01 -1.6163524e-03 -2.1593235e+00]
 [-5.1430976e-01 -8.1870681e-01 -2.3180747e+00]
 [ 1.4739224e-01 -7.9377288e-01 -2.4259059e+00]
 [-1.7610710e+00 -1.2931366e+00 -2.3630044e+00]
 [-1.3863993e+00 -1.0850264e+00 -2.3719902e+00]
 [-3.3671644e-01 -7.4732184e-01 -2.1563282e+00]
 [ 9.3637004e-02 -2.8711724e-01 -1.5542711e+00]
 [-1.4684405e+00  3.3757949e-01 -1.9945815e+00]
 [ 6.5159190e-01  6.1165786e-01 -1.3326182e+00]
 [-1.0210929e+00  4.0616900e-01 -1.0660357e+00]
 [-1.5676059e+00  2.4373762e-01 -6.0775357e-01]
 [ 7.4855159e-03  5.9805293e-02 -1.1648810e+00]
 [-3.6003560e-01 -2.9003033e-01 -1.2667215e+00]
 [ 7.4855159e-03 -6.1185622e-01 -1.6770787e+00]
 [-1.3707504e+00 -1.0775974e+00 -1.2277825e+00]
 [-1.4193354e+00 -1.4304692e+00 -1.8448160e+00]
 [ 9.9803299e-01 -1.1871541e+00 -1.8388253e+00]
 [-1.3287354e+00 -1.6163524e-03 -2.4139247e+00]
 [-9.3549508e-01 -3.9170080e-01 -1.8957362e+00]
 [ 2.6977423e-01 -1.6163524e-03 -1.7040365e+00]
 [-4.3959394e-01 -1.6163524e-03 -1.8657832e+00]
 [-3.2742205e-01  1.8861358e-01 -1.1588904e+00]
 [ 6.2526363e-01  8.9603245e-02 -1.4075010e+00]]

Corresponding RUL label for first sequence: 125.0

X shape (samples, timesteps, features): (1391, 50, 3)
y shape (samples,): (1391,)

Starting batch processing with batch size=500...

Batch rows: 0 to 499
Mean sensor values:
SensorA   -0.051051
SensorB   -0.793780
SensorC   -0.971791
dtype: float64
RUL range: 125 to 125

Batch rows: 500 to 999
Mean sensor values:
SensorA   -0.548613
SensorB    0.229646
SensorC    0.471639
dtype: float64
RUL range: 125 to 125

Batch rows: 1000 to 1439
Mean sensor values:
SensorA    0.681436
SensorB    0.641062
SensorC    0.568355
dtype: float64
RUL range: 0 to 125

Milestone 1 processing complete. Proceeding to LSTM training...

Train/Val split: (1112, 50, 3) / (279, 50, 3)
/usr/local/lib/python3.12/dist-packages/keras/src/layers/rnn/rnn.py:199: UserWarning: Do not pass an `input_shape`/`input_dim` argument to a layer. When using Sequential models, prefer using an `Input(shape)` object as the first layer in the model instead.
  super().__init__(**kwargs)
Epoch 1/100
18/18 ━━━━━━━━━━━━━━━━━━━━ 8s 202ms/step - loss: 124.2898 - mse: 15449.2646 - val_loss: 96.5950 - val_mse: 10884.0840 - learning_rate: 0.0010
Epoch 2/100
18/18 ━━━━━━━━━━━━━━━━━━━━ 4s 203ms/step - loss: 117.9769 - mse: 13921.6729 - val_loss: 85.7025 - val_mse: 8820.1826 - learning_rate: 0.0010
Epoch 3/100
18/18 ━━━━━━━━━━━━━━━━━━━━ 3s 151ms/step - loss: 111.9007 - mse: 12525.6816 - val_loss: 79.7877 - val_mse: 7731.5674 - learning_rate: 0.0010
Epoch 4/100
18/18 ━━━━━━━━━━━━━━━━━━━━ 3s 148ms/step - loss: 105.1785 - mse: 11071.2959 - val_loss: 73.4319 - val_mse: 6598.5186 - learning_rate: 0.0010
Epoch 5/100
18/18 ━━━━━━━━━━━━━━━━━━━━ 3s 150ms/step - loss: 97.2507 - mse: 9465.1172 - val_loss: 66.7169 - val_mse: 5451.5942 - learning_rate: 0.0010
Epoch 6/100
18/18 ━━━━━━━━━━━━━━━━━━━━ 4s 243ms/step - loss: 88.2851 - mse: 7805.1992 - val_loss: 59.7359 - val_mse: 4328.9800 - learning_rate: 0.0010
Epoch 7/100
18/18 ━━━━━━━━━━━━━━━━━━━━ 3s 144ms/step - loss: 78.1283 - mse: 6119.0112 - val_loss: 52.7122 - val_mse: 3295.8123 - learning_rate: 0.0010
Epoch 8/100
18/18 ━━━━━━━━━━━━━━━━━━━━ 3s 147ms/step - loss: 66.8627 - mse: 4490.5864 - val_loss: 45.9026 - val_mse: 2428.1323 - learning_rate: 0.0010
Epoch 9/100
18/18 ━━━━━━━━━━━━━━━━━━━━ 3s 148ms/step - loss: 54.1130 - mse: 2955.9961 - val_loss: 39.5102 - val_mse: 1808.3228 - learning_rate: 0.0010
Epoch 10/100
18/18 ━━━━━━━━━━━━━━━━━━━━ 4s 198ms/step - loss: 40.0974 - mse: 1647.4625 - val_loss: 34.0724 - val_mse: 1565.1311 - learning_rate: 0.0010
Epoch 11/100
18/18 ━━━━━━━━━━━━━━━━━━━━ 4s 191ms/step - loss: 24.4438 - mse: 650.5613 - val_loss: 30.0129 - val_mse: 1816.8704 - learning_rate: 0.0010
Epoch 12/100
18/18 ━━━━━━━━━━━━━━━━━━━━ 3s 147ms/step - loss: 10.1305 - mse: 148.4747 - val_loss: 28.8231 - val_mse: 2395.6313 - learning_rate: 0.0010
Epoch 13/100
18/18 ━━━━━━━━━━━━━━━━━━━━ 3s 156ms/step - loss: 6.5058 - mse: 64.0569 - val_loss: 28.8687 - val_mse: 2398.2644 - learning_rate: 0.0010
Epoch 14/100
18/18 ━━━━━━━━━━━━━━━━━━━━ 3s 155ms/step - loss: 6.3899 - mse: 62.1714 - val_loss: 28.3428 - val_mse: 2299.7437 - learning_rate: 0.0010
Epoch 15/100
18/18 ━━━━━━━━━━━━━━━━━━━━ 4s 224ms/step - loss: 6.2501 - mse: 61.4585 - val_loss: 28.3468 - val_mse: 2297.7500 - learning_rate: 0.0010
Epoch 16/100
18/18 ━━━━━━━━━━━━━━━━━━━━ 3s 154ms/step - loss: 5.8837 - mse: 56.4441 - val_loss: 28.3186 - val_mse: 2311.9453 - learning_rate: 0.0010
Epoch 17/100
18/18 ━━━━━━━━━━━━━━━━━━━━ 3s 147ms/step - loss: 6.3097 - mse: 64.0507 - val_loss: 28.2697 - val_mse: 2337.9194 - learning_rate: 0.0010
Epoch 18/100
18/18 ━━━━━━━━━━━━━━━━━━━━ 3s 151ms/step - loss: 5.6762 - mse: 52.2369 - val_loss: 28.2937 - val_mse: 2325.1045 - learning_rate: 0.0010
Epoch 19/100
18/18 ━━━━━━━━━━━━━━━━━━━━ 5s 281ms/step - loss: 6.1768 - mse: 60.3557 - val_loss: 28.2978 - val_mse: 2322.9253 - learning_rate: 0.0010
Epoch 20/100
18/18 ━━━━━━━━━━━━━━━━━━━━ 3s 150ms/step - loss: 6.2608 - mse: 60.3878 - val_loss: 28.2836 - val_mse: 2330.4768 - learning_rate: 0.0010
Epoch 21/100
18/18 ━━━━━━━━━━━━━━━━━━━━ 3s 146ms/step - loss: 6.4098 - mse: 63.8811 - val_loss: 28.3212 - val_mse: 2310.5833 - learning_rate: 0.0010
Epoch 22/100
18/18 ━━━━━━━━━━━━━━━━━━━━ 3s 146ms/step - loss: 6.0092 - mse: 58.9939 - val_loss: 28.2807 - val_mse: 2331.9976 - learning_rate: 0.0010
Epoch 23/100
18/18 ━━━━━━━━━━━━━━━━━━━━ 4s 207ms/step - loss: 6.4324 - mse: 64.4398 - val_loss: 28.2712 - val_mse: 2337.1074 - learning_rate: 5.0000e-04
Epoch 24/100
18/18 ━━━━━━━━━━━━━━━━━━━━ 3s 173ms/step - loss: 5.8421 - mse: 52.5144 - val_loss: 28.2907 - val_mse: 2326.7222 - learning_rate: 5.0000e-04
Epoch 25/100
18/18 ━━━━━━━━━━━━━━━━━━━━ 3s 144ms/step - loss: 6.3201 - mse: 61.4213 - val_loss: 28.3076 - val_mse: 2317.7783 - learning_rate: 5.0000e-04
Epoch 26/100
18/18 ━━━━━━━━━━━━━━━━━━━━ 3s 151ms/step - loss: 6.0556 - mse: 56.7838 - val_loss: 28.2991 - val_mse: 2322.2278 - learning_rate: 5.0000e-04
Epoch 27/100
18/18 ━━━━━━━━━━━━━━━━━━━━ 3s 145ms/step - loss: 6.4984 - mse: 63.1765 - val_loss: 28.3404 - val_mse: 2300.8765 - learning_rate: 5.0000e-04

Validation MAE: 28.2697 | Validation MSE: 2337.9194

Predicted RUL (first 5 val): [124.5999984741211, 124.5999984741211, 124.5999984741211, 124.5999984741211, 124.5999984741211]
Actual RUL     (first 5 val): [125.0, 125.0, 125.0, 125.0, 125.0]

Training complete.
