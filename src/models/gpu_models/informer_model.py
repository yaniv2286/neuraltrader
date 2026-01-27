import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.metrics import mean_squared_error

def prepare_data(X_train, y_train, X_test, window_size=10):
    def reshape(X, y):
        X_seq, y_seq = [], []
        for i in range(len(X) - window_size + 1):
            X_seq.append(X[i:i+window_size].values)
            y_seq.append(y.iloc[i+window_size-1])
        return np.array(X_seq), np.array(y_seq)

    X_train_seq, y_train_seq = reshape(X_train, y_train)
    X_test_seq, _ = reshape(X_test, y_train[:len(X_test)])  # dummy target
    return X_train_seq, y_train_seq, X_test_seq

def build_model(input_shape, lstm_units=64, dropout_rate=0.3, learning_rate=0.001):
    # Simplified Informer: use efficient attention via Conv1D + LSTM
    inputs = layers.Input(shape=input_shape)
    # Conv1D for downsampling (reduces sequence length)
    x = layers.Conv1D(filters=lstm_units, kernel_size=3, activation='relu', padding='causal')(inputs)
    x = layers.LSTM(lstm_units, return_sequences=True)(x)
    x = layers.Dropout(dropout_rate)(x)
    # Self-attention via MultiHeadAttention (efficient for short sequences)
    attn = layers.MultiHeadAttention(num_heads=4, key_dim=lstm_units)(x, x)
    x = layers.Add()([x, attn])
    x = layers.LayerNormalization()(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(dropout_rate)(x)
    outputs = layers.Dense(1)(x)

    model = models.Model(inputs, outputs)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mse')
    return model

def train_and_predict(X_train, y_train, X_test, model, epochs=20, batch_size=16):
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)
    preds = model.predict(X_test).flatten()
    rmse = np.sqrt(mean_squared_error(y_train[-len(preds):], preds))
    return preds, rmse

def predict_future_sequence(model, recent_X_df, steps=1, window_size=10):
    recent_X = recent_X_df.tail(window_size).values
    preds = []
    for _ in range(steps):
        input_seq = recent_X.reshape(1, window_size, -1)
        pred = model.predict(input_seq)[0][0]
        preds.append(pred)
        recent_X = np.vstack([recent_X[1:], [recent_X[-1]]])  # rolling window
    return np.array(preds)
