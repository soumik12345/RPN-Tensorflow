import click

from rpn.trainer import Trainer


@click.command()
@click.option('--wandb_project_name', help='Wandb Project Name')
@click.option('--wandb_experiment_name', help='Wandb Experiment Name')
@click.option('--wandb_api_key', help='Wandb API')
@click.option('--use_voc_2012', is_flag=True, help='Include VOC 2012 Dataset in Training')
@click.option('--batch_size', default=8, help='Batch Size')
@click.option('--backbone', default='vgg16', help='Specify the Backbone [vgg16, vgg19 or mobilenet_v2]')
@click.option('--learning_rate', default=1e-5, help='Learning Rate')
@click.option('--epochs', default=75, help='Number of Training Epochs')
def train(use_voc_2012, batch_size, backbone, learning_rate, epochs):
    trainer = Trainer(
        image_size=500, feature_map_shape=31, anchor_ratios=[1., 2., 1. / 2.],
        anchor_scales=[128, 256, 512], total_positive_bboxes=128,
        total_negative_bboxes=128, variance=[0.1, 0.1, 0.2, 0.2]
    )
    trainer.get_distribute_strategy()
    trainer.build_dataloader(data_directory='./datasets', use_voc_2012=use_voc_2012, batch_size=batch_size)
    trainer.build_model(backbone=backbone, learning_rate=learning_rate)
    trainer.train(epochs=epochs, model_checkpoint_path='./checkpoints')


if __name__ == "__main__":
    train()
