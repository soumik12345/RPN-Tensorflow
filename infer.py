import click

from rpn import Inferer


@click.command()
@click.option(
    '--feature_map_shape', default=32,
    help='Shape of the output feature map')
@click.option(
    '--backbone', default='mobilenet_v2',
    help='Specify the Backbone [vgg16, vgg19 or mobilenet_v2]')
@click.option(
    '--model_file', default='./checkpoints/RegionProposalNetwork_MobileNetv2.h5',
    help='Path of pre-trained model file')
@click.option('--image_file', help='Path of the image to be inferred')
def infer(feature_map_shape, backbone, model_file, image_file):
    inferer = Inferer(
        image_size=500, anchor_ratios=[1., 2., 1. / 2.],
        anchor_scales=[128, 256, 512], feature_map_shape=feature_map_shape,
        backbone=backbone, model_name='RPN', model_file=model_file
    )
    inferer.infer_from_image(image_file=image_file, variance=[0.1, 0.1, 0.2, 0.2])


if __name__ == "__main__":
    infer()
