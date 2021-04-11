import tensorflow as tf


class RegionProposalNetwork(tf.keras.Model):

    def __init__(self, image_size: int, anchor_count: int, backbone: str, **kwargs):
        super(RegionProposalNetwork, self).__init__(**kwargs)
        self.image_size = image_size
        self.backbone = backbone
        self.feature_extractor = self.build_feature_extractor()
        self.convolution = tf.keras.layers.Conv2D(
            512, (3, 3), activation='relu', padding='same', name='rpn_convolution')
        self.classification_head = tf.keras.layers.Conv2D(
            anchor_count, (1, 1), activation='sigmoid', name='classification_head')
        self.regression_head = tf.keras.layers.Conv2D(
            anchor_count * 4, (1, 1), activation='linear', name='regression_head')

    def build_feature_extractor(self):
        if self.backbone == 'vgg16':
            return tf.keras.applications.vgg16.VGG16(
                include_top=False, weights='imagenet',
                input_shape=(self.image_size, self.image_size, 3)
            ).get_layer('block5_conv3')
        elif self.backbone == 'vgg19':
            return tf.keras.applications.vgg19.VGG19(
                include_top=False, weights='imagenet',
                input_shape=(self.image_size, self.image_size, 3)
            ).get_layer('block5_conv4')
        elif self.backbone == 'mobilenet_v2':
            return tf.keras.applications.mobilenet_v2.MobileNetV2(
                include_top=False, weights='imagenet',
                input_shape=(self.image_size, self.image_size, 3)
            ).get_layer('block_13_expand_relu')

    def call(self, inputs, **kwargs):
        features = self.feature_extractor(inputs)
        rpn_out = self.convolution(features)
        classification_output = self.classification_head(rpn_out)
        regression_output = self.regression_head(rpn_out)
        return regression_output, classification_output
