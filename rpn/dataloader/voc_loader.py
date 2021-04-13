from typing import List
import tensorflow as tf
import tensorflow_datasets as tfds

from .commons import preprocess_data
from .anchors import generate_anchors
from .data_generator import data_generator


class VOCDataLoader:

    def __init__(self, data_directory: str, use_voc_2012: bool = False):
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
        if use_voc_2012:
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
            self, image_size: int, batch_size: int, anchor_ratios: List[float],
            anchor_scales: List[int], variance: List[float], feature_map_shape: int):
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
        anchor_count = len(anchor_ratios) * len(anchor_scales)
        train_generator = data_generator(
            self.train_data, anchors,
            feature_map_shape, anchor_count, 128, 128, variance
        )
        val_generator = data_generator(
            self.val_data, anchors,
            feature_map_shape, anchor_count, 128, 128, variance
        )
        return train_generator, val_generator


class VOCLoaderTest:

    def __init__(self, data_directory: str):
        self.test_data, dataset_info = tfds.load(
            'voc/2007', split='test',
            data_dir=data_directory, with_info=True
        )
        self.labels = ['background'] + dataset_info.features['labels'].names
        self.padding_values = (
            tf.constant(0, tf.float32),
            tf.constant(0, tf.float32),
            tf.constant(-1, tf.int32)
        )

    def configure_dataset(self, image_size: int, batch_size: int):
        test_data = self.test_data.map(lambda x: preprocess_data(x, image_size, image_size))
        test_data = test_data.padded_batch(
            batch_size, padded_shapes=(
                [None, None, None], [None, None], [None, ]
            ), padding_values=self.padding_values
        )
        return test_data
