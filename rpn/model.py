import tensorflow as tf


def build_rpn_model(image_size: int, anchor_count: int, backbone: str, model_name: str):
    assert backbone in ['vgg16', 'vgg19', 'mobilenet_v2']
    backbone_model, feature_extractor = None, None
    if backbone == 'vgg16':
        backbone_model = tf.keras.applications.vgg16.VGG16(
            include_top=False, weights='imagenet',
            input_shape=(image_size, image_size, 3)
        )
        feature_extractor = backbone_model.get_layer('block5_conv3')
    elif backbone == 'vgg19':
        backbone_model = tf.keras.applications.vgg19.VGG19(
            include_top=False, weights='imagenet',
            input_shape=(image_size, image_size, 3)
        )
        feature_extractor = backbone_model.get_layer('block5_conv4')
    elif backbone == 'mobilenet_v2':
        backbone_model = tf.keras.applications.mobilenet_v2.MobileNetV2(
            include_top=False, weights='imagenet',
            input_shape=(image_size, image_size, 3)
        )
        feature_extractor = backbone_model.get_layer('block_13_expand_relu')
    rpn_output = tf.keras.layers.Conv2D(
        512, (3, 3), activation='relu',
        padding='same', name='rpn_output'
    )(feature_extractor.output)
    classification_head = tf.keras.layers.Conv2D(
        anchor_count, (1, 1),
        activation='sigmoid', name='rpn_classification_head'
    )(rpn_output)
    regression_head = tf.keras.layers.Conv2D(
        anchor_count * 4, (1, 1),
        activation='linear', name='rpn_reg'
    )(rpn_output)
    return tf.keras.Model(
        inputs=backbone_model.input,
        outputs=[regression_head, classification_head],
        name=model_name
    )
