import tensorflow as tf


def regression_loss(y_true, y_pred):
    smooth_l1 = tf.keras.losses.Huber(
        reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE)
    batch_size = tf.shape(y_pred)[0]
    y_true = tf.reshape(y_true, [batch_size, -1, 4])
    y_pred = tf.reshape(y_pred, [batch_size, -1, 4])
    loss = smooth_l1(y_true, y_pred)
    valid = tf.math.reduce_any(tf.not_equal(y_true, 0.0), axis=-1)
    valid = tf.cast(valid, tf.float32)
    loss = tf.reduce_sum(loss * valid, axis=-1)
    total_pos_boxes = tf.math.maximum(1.0, tf.reduce_sum(valid, axis=-1))
    return tf.math.reduce_mean(tf.truediv(loss, total_pos_boxes))


def classification_loss(y_true, y_pred):
    indices = tf.where(tf.not_equal(
        y_true, tf.constant(-1.0, dtype=tf.float32)))
    target = tf.gather_nd(y_true, indices)
    output = tf.gather_nd(y_pred, indices)
    return tf.losses.BinaryCrossentropy()(target, output)
