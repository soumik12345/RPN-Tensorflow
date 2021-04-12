from typing import List
import tensorflow as tf
import tensorflow_datasets as tfds

from .commons import preprocess_data
from .anchors import generate_anchors


class VOCDataLoader:

    def __init__(self, data_directory: str):
        self.train_data, dataset_info = tfds.load(
            'voc/2007', split='train+validation',
            data_dir=data_directory, with_info=True
        )
        self.val_data, _ = tfds.load(
            'voc/2007', split='test',
            data_dir=data_directory, with_info=True
        )
        self.total_items_train = dataset_info.splits['train'].num_examples
        self.total_items_train += dataset_info.splits['validation'].num_examples
        self.total_items_val = dataset_info.splits['test'].num_examples
        train_2012, dataset_info_2012 = tfds.load(
            'voc/2012', split='train+validation',
            data_dir=data_directory, with_info=True
        )
        self.total_items_train += dataset_info_2012.splits['train'].num_examples
        self.total_items_train += dataset_info_2012.splits['validation'].num_examples
        self.train_data = self.train_data.concatenate(train_2012)
        self.labels = dataset_info.features['labels'].names
        self.total_labels = len(self.labels) + 1
        self.padding_values = (
            tf.constant(0, tf.float32),
            tf.constant(0, tf.float32),
            tf.constant(-1, tf.int32)
        )

    def configure_datasets(
            self, image_size: int, batch_size: int,
            anchor_ratios: List[float], anchor_scales: List[int], feature_map_shape: int):
        self.train_data = self.train_data.map(
            lambda x: preprocess_data(x, image_size, image_size, apply_augmentation=True))
        self.val_data = self.val_data.map(
            lambda x: preprocess_data(x, image_size, image_size))
        self.train_data = self.train_data.padded_batch(
            batch_size, padded_shapes=(
                [None, None, None], [None, None], [None, ]
            ), padding_values=self.padding_values
        )
        self.val_data = self.val_data.padded_batch(
            batch_size, padded_shapes=(
                [None, None, None], [None, None], [None, ]
            ), padding_values=self.padding_values
        )
        anchors = generate_anchors(
            image_size, anchor_ratios,
            anchor_scales, feature_map_shape
        )
