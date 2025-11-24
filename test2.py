import tensorflow as tf

# Force a small matrix multiply on the GPU
with tf.device('/GPU:0'):
    a = tf.random.normal([1000, 1000])
    b = tf.random.normal([1000, 1000])
    c = tf.matmul(a, b)

print("Computation finished successfully on GPU 🚀")
