import tensorflow as tf


def regression_loss(y_true, y_pred):
    y_pred = tf.reshape(y_pred, (tf.shape(y_pred)[0], -1, 4))
    loss_fn = tf.losses.Huber(reduction=tf.losses.Reduction.NONE)
    loss_for_all = loss_fn(y_true, y_pred)
    loss_for_all = tf.reduce_sum(loss_for_all, axis=-1)
    pos_cond = tf.reduce_any(tf.not_equal(y_true, tf.constant(0.0)), axis=-1)
    pos_mask = tf.cast(pos_cond, dtype=tf.float32)
    loc_loss = tf.reduce_sum(pos_mask * loss_for_all)
    total_pos_bboxes = tf.maximum(1.0, tf.reduce_sum(pos_mask))
    return loc_loss / total_pos_bboxes


def classification_loss(y_true, y_pred):
    indices = tf.where(tf.not_equal(
        y_true, tf.constant(-1.0, dtype=tf.float32)))
    target = tf.gather_nd(y_true, indices)
    output = tf.gather_nd(y_pred, indices)
    return tf.losses.BinaryCrossentropy()(target, output)
