import tensorflow as tf

def custom_loss(y_true, y_pred):

  foo = tf.constant(.75, dtype=tf.float32)
  goo = tf.subtract(y_true, y_pred)
  shoe = tf.reduce_mean(goo)
  return tf.multiply(foo, shoe)
