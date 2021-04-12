import tensorflow as tf


def generate_base_anchors(image_size, anchor_ratios, anchor_scales):
    base_anchors = []
    for scale in anchor_scales:
        scale /= image_size
        for ratio in anchor_ratios:
            w = tf.sqrt(scale ** 2 / ratio)
            h = w * ratio
            base_anchors.append([-h / 2, -w / 2, h / 2, w / 2])
    return tf.cast(base_anchors, dtype=tf.float32)


def generate_anchors(image_size, anchor_ratios, anchor_scales, feature_map_shape):
    stride = 1 / feature_map_shape
    grid_coords = tf.cast(
        tf.range(0, feature_map_shape) / feature_map_shape + stride / 2, dtype=tf.float32)
    grid_x, grid_y = tf.meshgrid(grid_coords, grid_coords)
    flat_grid_x, flat_grid_y = tf.reshape(grid_x, (-1, )), tf.reshape(grid_y, (-1, ))
    grid_map = tf.stack([flat_grid_y, flat_grid_x, flat_grid_y, flat_grid_x], axis=-1)
    base_anchors = generate_base_anchors(image_size, anchor_ratios, anchor_scales)
    anchors = tf.reshape(base_anchors, (1, -1, 4)) + tf.reshape(grid_map, (-1, 1, 4))
    anchors = tf.reshape(anchors, (-1, 4))
    return tf.clip_by_value(anchors, 0, 1)
