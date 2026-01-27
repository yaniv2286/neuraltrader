import numpy as np
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from tensorflow.keras import layers, models
import pandas as pd

def reshape_for_patchtst(X, patch_len=8):
    # Convert to NumPy array if it's a DataFrame
    if isinstance(X, pd.DataFrame):
        X = X.to_numpy()

    num_samples, num_features = X.shape
    new_len = (num_features // patch_len) * patch_len
    X = X[:, :new_len]
    return X.reshape((num_samples, new_len // patch_len, patch_len))


def build_model(input_shape, num_heads=4, key_dim=8, dense_units=64, dropout=0.1, lr=0.001):
    inputs = tf.keras.Input(shape=input_shape)
    x = layers.LayerNormalization()(inputs)
    x = layers.MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)(x, x)
    x = layers.Dropout(dropout)(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(dense_units, activation="relu")(x)
    x = layers.Dense(1)(x)
    model = models.Model(inputs, x)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss="mse")
    return model

def prepare_data(X_train, y_train, X_test, patch_len=8):
    X_train_reshaped = reshape_for_patchtst(X_train, patch_len)
    X_test_reshaped = reshape_for_patchtst(X_test, patch_len)
    return X_train_reshaped, y_train, X_test_reshaped

def train_and_predict(X_train, y_train, X_test, model, epochs=50, batch_size=16):
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)
    preds = model.predict(X_test).flatten()
    rmse = np.sqrt(mean_squared_error(y_train[-len(preds):], preds))
    return preds, rmse
