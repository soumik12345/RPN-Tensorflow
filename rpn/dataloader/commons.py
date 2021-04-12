import tensorflow as tf


def horizontal_flip(image, gt_boxes):
    flipped_image = tf.image.flip_left_right(image)
    flipped_gt_boxes = tf.stack(
        [gt_boxes[..., 0],
         1.0 - gt_boxes[..., 3],
         gt_boxes[..., 2],
         1.0 - gt_boxes[..., 1]], -1
    )
    return flipped_image, flipped_gt_boxes


def randomly_apply_operation(operation, image, gt_boxes):
    return tf.cond(
        tf.greater(tf.random.uniform((), dtype=tf.float32), 0.5),
        lambda: operation(image, gt_boxes), lambda: (image, gt_boxes)
    )


def preprocess_data(image_data, final_height, final_width, apply_augmentation=False, evaluate=False):
    img = image_data["image"]
    gt_boxes = image_data["objects"]["bbox"]
    gt_labels = tf.cast(image_data["objects"]["label"] + 1, tf.int32)
    if evaluate:
        not_diff = tf.logical_not(image_data["objects"]["is_difficult"])
        gt_boxes = gt_boxes[not_diff]
        gt_labels = gt_labels[not_diff]
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, (final_height, final_width))
    if apply_augmentation:
        img, gt_boxes = randomly_apply_operation(horizontal_flip, img, gt_boxes)
    return img, gt_boxes, gt_labels
