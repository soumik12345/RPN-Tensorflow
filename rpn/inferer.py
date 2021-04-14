import numpy as np
from PIL import Image
import tensorflow as tf
from typing import List
from matplotlib import pyplot as plt

from . import build_rpn_model
from .dataloader import generate_anchors, VOCLoaderTest


def get_bboxes_from_deltas(anchors, deltas):
    all_anc_width = anchors[..., 3] - anchors[..., 1]
    all_anc_height = anchors[..., 2] - anchors[..., 0]
    all_anc_ctr_x = anchors[..., 1] + 0.5 * all_anc_width
    all_anc_ctr_y = anchors[..., 0] + 0.5 * all_anc_height
    all_bbox_width = tf.exp(deltas[..., 3]) * all_anc_width
    all_bbox_height = tf.exp(deltas[..., 2]) * all_anc_height
    all_bbox_ctr_x = (deltas[..., 1] * all_anc_width) + all_anc_ctr_x
    all_bbox_ctr_y = (deltas[..., 0] * all_anc_height) + all_anc_ctr_y
    y1 = all_bbox_ctr_y - (0.5 * all_bbox_height)
    x1 = all_bbox_ctr_x - (0.5 * all_bbox_width)
    y2 = all_bbox_height + y1
    x2 = all_bbox_width + x1
    return tf.stack([y1, x1, y2, x2], axis=-1)


class Inferer:

    def __init__(
            self, image_size: int,
            anchor_ratios: List[float],
            anchor_scales: List[int],
            feature_map_shape: int,
            backbone: str,
            model_name: str,
            model_file: str):

        self.image_size = image_size
        anchor_count = len(anchor_ratios) * len(anchor_scales)
        self.model = build_rpn_model(
            image_size=image_size, anchor_count=anchor_count,
            backbone=backbone, model_name=model_name
        )
        self.model.load_weights(model_file, by_name=True)
        self.anchors = generate_anchors(
            image_size=image_size, anchor_ratios=anchor_ratios,
            anchor_scales=anchor_scales, feature_map_shape=feature_map_shape
        )
        self.box_colors = tf.constant([[1, 0, 0, 1]], dtype=tf.float32)

    def _infer(self, images, batch_size, variance):
        bbox_deltas, labels = self.model.predict_on_batch(images)
        bbox_deltas = tf.reshape(bbox_deltas, (batch_size, -1, 4))
        labels = tf.reshape(labels, (batch_size, -1))
        bbox_deltas *= variance
        bboxes = get_bboxes_from_deltas(self.anchors, bbox_deltas)
        _, top_indices = tf.nn.top_k(labels, 10)
        selected_bboxes = tf.gather(bboxes, top_indices, batch_dims=1)
        return selected_bboxes, labels

    def _draw_bboxes(self, images, bboxes):
        images_with_bbox = tf.image.draw_bounding_boxes(images, bboxes, self.box_colors)
        plt.figure(figsize=(15, 10))
        for img_with_bb in images_with_bbox:
            plt.imshow(img_with_bb)
            plt.axis('off')
            plt.show()

    def infer_from_test_dataset(self, data_directory: str, batch_size: int, variance: List[float]):
        test_dataloader = VOCLoaderTest(
            data_directory=data_directory
        ).configure_dataset(
            image_size=self.image_size, batch_size=batch_size
        )
        for images, _, _ in test_dataloader:
            bboxes = self._infer(images=images, batch_size=batch_size, variance=variance)
            self._draw_bboxes(images=images, bboxes=bboxes)

    def infer_from_image(self, image_file: str, variance: List[float]):
        image = Image.open(image_file)
        resized_image = image.resize((self.image_size, self.image_size), Image.LANCZOS)
        image = np.array(resized_image)
        image = tf.image.convert_image_dtype(image, tf.float32)
        image = tf.expand_dims(image, axis=0)
        bboxes, labels = self._infer(images=image, batch_size=1, variance=variance)
        self._draw_bboxes(images=image, bboxes=bboxes)
