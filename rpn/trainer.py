import os
import tensorflow as tf
from typing import List
from datetime import datetime

from . import build_rpn_model
from .dataloader import VOCDataLoader
from .loss_functions import regression_loss, classification_loss


class Trainer:

    def __init__(
            self, image_size: int,
            feature_map_shape: int,
            anchor_ratios: List[float],
            anchor_scales: List[int],
            total_positive_bboxes: int,
            total_negative_bboxes: int,
            variance: List[float]):

        self.image_size = image_size
        self.feature_map_shape = feature_map_shape
        self.anchor_ratios = anchor_ratios
        self.anchor_scales = anchor_scales
        self.total_positive_bboxes = total_positive_bboxes
        self.total_negative_bboxes = total_negative_bboxes
        self.distribute_strategy = None
        self.get_distribute_strategy()
        self.variance = variance
        self.dataloader = None
        self.batch_size = 0
        self.train_generator, self.val_generator = None, None
        self.model = None

    def get_distribute_strategy(self):
        try:
            gpus = tf.config.list_physical_devices('GPU')
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            if len(gpus) == 1:
                self.distribute_strategy = tf.distribute.OneDeviceStrategy(
                    'GPU:' + gpus[0].name.split(':')[-1]
                )
            elif len(gpus) > 1:
                self.distribute_strategy = tf.distribute.MirroredStrategy(
                    devices=['GPU:' + gpu.name.split(':')[-1] for gpu in gpus]
                )
        except Exception as e:
            print(e)

    def build_dataloader(self, data_directory: str, use_voc_2012: bool, batch_size: int):
        self.dataloader = VOCDataLoader(
            data_directory=data_directory, use_voc_2012=use_voc_2012)
        self.train_generator, self.val_generator = self.dataloader.configure_datasets(
            image_size=self.image_size, batch_size=batch_size, anchor_ratios=self.anchor_ratios,
            anchor_scales=self.anchor_scales, variance=self.variance, feature_map_shape=self.feature_map_shape
        )

    def build_model(
            self, backbone: str, learning_rate: float = 1e-5, model_name: str = 'RegionProposalNetwork'):
        try:
            with self.distribute_strategy.scope():
                self.model = build_rpn_model(
                    self.image_size, len(self.anchor_ratios) * len(self.anchor_scales),
                    backbone=backbone, model_name=model_name + '_' + backbone.upper()
                )
                self.model.compile(
                    optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                    loss=[regression_loss, classification_loss]
                )
                self.model.summary()
        except AttributeError:
            print('Please don\'t train this on CPU, you will die of old age before its done')

    def train(self, epochs: int, model_checkpoint_path: str, model_name: str = 'RegionProposalNetwork'):
        log_directory = os.path.join('logs', datetime.now().strftime("%Y%m%d-%H%M%S"))
        callbacks = [
            tf.keras.callbacks.ModelCheckpoint(
                os.path.join(model_checkpoint_path, model_name),
                monitor='val_loss', save_best_only=True, save_weights_only=True
            ),
            tf.keras.callbacks.TensorBoard(log_dir=log_directory, histogram_freq=1)
        ]
        train_steps_per_epoch = int(self.dataloader.total_items_train // self.batch_size)
        val_steps_per_epoch = int(self.dataloader.total_items_train // self.batch_size)
        self.model.fit(
            self.train_generator, steps_per_epoch=train_steps_per_epoch,
            validation_data=self.val_generator, validation_steps=val_steps_per_epoch,
            epochs=epochs, callbacks=callbacks
        )
