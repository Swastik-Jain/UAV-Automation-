import sys
print("Starting verification...", flush=True)

try:
    print("Importing tensorflow...", flush=True)
    import tensorflow as tf
    print(f"TensorFlow version: {tf.__version__}", flush=True)
except Exception as e:
    print(f"TensorFlow import failed: {e}", flush=True)

try:
    print("Importing keras...", flush=True)
    from keras.models import load_model
    print("Keras imported.", flush=True)
except Exception as e:
    print(f"Keras import failed: {e}", flush=True)

try:
    print("Importing airsim...", flush=True)
    import airsim
    print("Airsim imported.", flush=True)
except Exception as e:
    print(f"Airsim import failed: {e}", flush=True)

print("Verification complete.", flush=True)
