
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

def prepare_data(X_train, y_train, X_test, forecast_horizon=1):
    """
    Scales input and reshapes for N-BEATS model.
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    X_train_nbeats = X_train_scaled.reshape((X_train_scaled.shape[0], X_train_scaled.shape[1], 1))
    X_test_nbeats = X_test_scaled.reshape((X_test_scaled.shape[0], X_test_scaled.shape[1], 1))

    return X_train_nbeats, y_train, X_test_nbeats, scaler

def build_model(input_size=26, hidden_units=128, stacks=2, blocks_per_stack=2):
    """
    Builds a simple N-BEATS-style fully connected deep model.
    """
    inputs = tf.keras.Input(shape=(input_size, 1))
    x = tf.keras.layers.Flatten()(inputs)
    for _ in range(stacks * blocks_per_stack):
        x = tf.keras.layers.Dense(hidden_units, activation='relu')(x)
    outputs = tf.keras.layers.Dense(1)(x)
    model = tf.keras.Model(inputs, outputs)
    model.compile(optimizer="adam", loss="mse")
    return model

def train_and_predict(X_train, y_train, X_test, model, epochs=20, batch_size=16):
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)
    preds = model.predict(X_test).flatten()
    rmse = np.sqrt(mean_squared_error(y_train[-len(preds):], preds))
    return preds, rmse

def predict_future_sequence(model, recent_X_df, steps=1, scaler=None):
    """
    Predict future sequence by feeding the last known values.
    For simplicity, assumes no autoregressive feedback (i.e., static input).
    """
    if scaler:
        recent_X = scaler.transform(recent_X_df.values)
    else:
        recent_X = recent_X_df.values

    preds = []
    for _ in range(steps):
        pred = model.predict(recent_X.reshape(1, -1, 1))[0][0]
        preds.append(pred)
    return np.array(preds)
