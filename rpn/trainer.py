import os
import wandb
import tensorflow as tf
from typing import List
from datetime import datetime
from wandb.keras import WandbCallback

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
        self.gpus = []
        self.variance = variance
        self.dataloader = None
        self.batch_size = 0
        self.train_generator, self.val_generator = None, None
        self.model = None
        self.log_directory = ''

    def set_gpu_memory_growth(self):
        try:
            self.gpus = tf.config.list_physical_devices('GPU')
            if len(self.gpus) == 0:
                raise ValueError(
                    'Please don\'t train this on CPU, you will die of old age before its done')
            for gpu in self.gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except Exception as e:
            print(e)

    def init_wandb(self, project_name, experiment_name, entity, wandb_api_key):
        self.log_directory = os.path.join('logs', datetime.now().strftime("%Y%m%d-%H%M%S"))
        wandb.tensorboard.patch(root_logdir=self.log_directory)
        if project_name is not None and experiment_name is not None:
            os.environ['WANDB_API_KEY'] = wandb_api_key
            wandb.init(
                project=project_name, name=experiment_name,
                entity=entity, sync_tensorboard=True
            )

    def build_dataloader(self, data_directory: str, use_voc_2012: bool, batch_size: int):
        self.batch_size = batch_size
        self.dataloader = VOCDataLoader(
            data_directory=data_directory, use_voc_2012=use_voc_2012)
        self.train_generator, self.val_generator = self.dataloader.configure_datasets(
            image_size=self.image_size, batch_size=batch_size, anchor_ratios=self.anchor_ratios,
            anchor_scales=self.anchor_scales, variance=self.variance, feature_map_shape=self.feature_map_shape
        )

    def build_model(
            self, backbone: str, learning_rate: float = 1e-5,
            model_name: str = 'RegionProposalNetwork', show_summary: bool = True):
        self.model = build_rpn_model(
            self.image_size, len(self.anchor_ratios) * len(self.anchor_scales),
            backbone=backbone, model_name=model_name + '_' + backbone.upper()
        )
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss=[regression_loss, classification_loss]
        )
        if show_summary:
            self.model.summary()

    def train(self, epochs: int, model_checkpoint_path: str, model_name: str = 'RegionProposalNetwork'):

        callbacks = [
            tf.keras.callbacks.ModelCheckpoint(
                os.path.join(model_checkpoint_path, model_name + '.h5'),
                monitor='val_loss', save_best_only=True, save_weights_only=True
            ),
            tf.keras.callbacks.TensorBoard(
                log_dir=self.log_directory, histogram_freq=1
            ),
            WandbCallback()
        ]
        train_steps_per_epoch = int(self.dataloader.total_items_train // self.batch_size)
        val_steps_per_epoch = int(self.dataloader.total_items_train // self.batch_size)
        self.model.fit(
            self.train_generator, steps_per_epoch=train_steps_per_epoch,
            validation_data=self.val_generator, validation_steps=val_steps_per_epoch,
            epochs=epochs, callbacks=callbacks
        )
