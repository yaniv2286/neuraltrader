import os
import tensorflow as tf

def configure_gpu():
    """Configure GPU if available; otherwise fallback to CPU."""
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # Enable memory growth for all GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"Using GPU: {gpus}")
        except RuntimeError as e:
            print(e)
    else:
        print("No GPU detected, using CPU.")
    # Disable noisy TensorFlow logs
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
