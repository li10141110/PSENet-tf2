import tensorflow as tf
import sys
sys.path.append('../')
# this BottleNeck is for mobilenet family
from models.mobilenet_v3_block import BottleNeck, h_swish
NUM_CLASSES = 10


def conv3x3(out_planes, strides=1):
    """3x3 convolution with padding"""
    return tf.keras.layers.Conv2D(out_planes, kernel_size=3, strides=strides,
                     padding='same', use_bias=False)

class BasicBlock(tf.keras.Model):
    expansion = 1
    
    def __init__(self, inplanes, planes, strides=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(planes, strides)
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.conv2 = conv3x3(planes)
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.downsample = downsample
        self.strides = strides
        
    def call(self, x):
        residual = x
        
        out = tf.nn.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        return tf.nn.relu(out)

# this Bottleneck is for resnet family

class Bottleneck(tf.keras.Model):
    expansion = 4
    
    def __init__(self, inplanes, planes, strides=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(planes, kernel_size=1, use_bias=False)
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.conv2 = tf.keras.layers.Conv2D(planes, kernel_size=3, strides=strides, padding="same", use_bias=False)
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.conv3 = tf.keras.layers.Conv2D(planes*4, kernel_size=1, use_bias=False)
        self.bn3 = tf.keras.layers.BatchNormalization()
        self.downsample = downsample
        self.strides = strides
        
    def call(self, x):
        residual = x
        
        out = tf.nn.relu(self.bn1(self.conv1(x)))
        out = tf.nn.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        
        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        return tf.nn.relu(out)

###！ resnet18/ resnet34 didn't use pretrained model
class ResNet(tf.keras.Model):
    def __init__(self, block, layers, num_classes=7, scale=1):
        self.inplanes = 64
        super(ResNet, self).__init__()
        
        self.conv1 = tf.keras.layers.Conv2D(64, 7, 2, padding="same", input_shape=(640, 640, 3), \
                                            data_format='channels_last', use_bias=False)
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], strides=2)
        self.layer3 = self._make_layer(block, 256, layers[2], strides=2)
        self.layer4 = self._make_layer(block, 512, layers[3], strides=2)
        # self.avgpool
        # self.fc
        
        # Top layer
        self.toplayer = tf.keras.layers.Conv2D(256, kernel_size=1, strides=1, padding="same")  # Reduce channels
        self.toplayer_bn = tf.keras.layers.BatchNormalization()
        
        # Smooth layers
        self.smooth1 = tf.keras.layers.Conv2D(256, kernel_size=3, strides=1, padding="same")
        self.smooth1_bn = tf.keras.layers.BatchNormalization()

        self.smooth2 = tf.keras.layers.Conv2D(256, kernel_size=3, strides=1, padding="same")
        self.smooth2_bn = tf.keras.layers.BatchNormalization()

        self.smooth3 = tf.keras.layers.Conv2D(256, kernel_size=3, strides=1, padding="same")
        self.smooth3_bn = tf.keras.layers.BatchNormalization()

        # Lateral layers
        self.latlayer1 = tf.keras.layers.Conv2D(256, kernel_size=1, strides=1, padding="same")
        self.latlayer1_bn = tf.keras.layers.BatchNormalization()

        self.latlayer2 = tf.keras.layers.Conv2D(256, kernel_size=1, strides=1, padding="same")
        self.latlayer2_bn = tf.keras.layers.BatchNormalization()

        self.latlayer3 = tf.keras.layers.Conv2D(256, kernel_size=1, strides=1, padding="same")
        self.latlayer3_bn = tf.keras.layers.BatchNormalization()

        self.conv2 = tf.keras.layers.Conv2D(256, kernel_size=3, strides=1, padding="same")
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.conv3 = tf.keras.layers.Conv2D(num_classes, kernel_size=1, strides=1, padding="same")

        self.scale = scale
        
        
    def _make_layer(self, block, planes, blocks, strides=1):
        downsample = None
        if strides != 1 or self.inplanes != planes * block.expansion:
            downsample = tf.keras.Sequential([
                tf.keras.layers.Conv2D(planes * block.expansion,
                          kernel_size=1, strides=strides, use_bias=False),
                tf.keras.layers.BatchNormalization()
            ])
            
        layers = []
        layers.append(block(self.inplanes, planes, strides, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
            
        return tf.keras.Sequential(layers)
        
    def _upsample(self, x, y, scale=1):
        _, H, W, _ = y.shape
        return tf.image.resize(x, (H // scale, W // scale))
        
    def _upsample_add(self, x, y):
        return self._upsample(x, y) + y

    
    def call(self, x):
        h = x
        h = tf.nn.relu(self.bn1(self.conv1(h)))
        h = tf.nn.max_pool2d(h, 3, 2, 'SAME', data_format='NHWC')
        
        h = self.layer1(h)
        c2 = h
        h = self.layer2(h)
        c3 = h
        h = self.layer3(h)
        c4 = h
        h = self.layer4(h)
        c5 = h
        
        # Top-down
        p5 = self.toplayer(c5)
        p5 = tf.nn.relu(self.toplayer_bn(p5))
        
        c4 = self.latlayer1(c4)
        c4 = tf.nn.relu(self.latlayer1_bn(c4))
        p4 = self._upsample_add(p5, c4)
        p4 = self.smooth1(p4)
        p4 = tf.nn.relu(self.smooth1_bn(p4))
        
        c3 = self.latlayer2(c3)
        c3 = tf.nn.relu(self.latlayer2_bn(c3))
        p3 = self._upsample_add(p4, c3)
        p3 = self.smooth2(p3)
        p3 = tf.nn.relu(self.smooth2_bn(p3))  
        
        c2 = self.latlayer3(c2)
        c2 = tf.nn.relu(self.latlayer3_bn(c2))
        p2 = self._upsample_add(p3, c2)
        p2 = self.smooth3(p2)
        p2 = tf.nn.relu(self.smooth3_bn(p2))

#       make p2,p3,p4,p5 have the same size
        p3 = self._upsample(p3, p2)
        p4 = self._upsample(p4, p2)
        p5 = self._upsample(p5, p2)
#n=(n,h,w,c)
        out = tf.concat((p2, p3, p4, p5), 3)
        out = self.conv2(out)
        out = tf.nn.relu(self.bn2(out))
        out = self.conv3(out)
        out = self._upsample(out, x, scale=self.scale)
        #output 7 point/class
        return out


class TFPreMobileNetV2(tf.keras.Model):
    def __init__(self, num_classes=7, scale=1):
        super(TFPreMobileNetV2, self).__init__()

        mobilenet = tf.keras.applications.MobileNetV2(input_shape=(640, 640, 3), alpha=1.0, include_top=False, weights='imagenet')
        inds = [27, 54, 116, 150]

        outputs = [mobilenet.layers[i].output for i in inds]
        self.res = tf.keras.Model(inputs=mobilenet.input, outputs=outputs)
        # self.avgpool
        # self.fc

        # Top layer
        self.toplayer = tf.keras.layers.Conv2D(256, kernel_size=1, strides=1, padding="same")  # Reduce channels
        self.toplayer_bn = tf.keras.layers.BatchNormalization()

        # Smooth layers
        self.smooth1 = tf.keras.layers.Conv2D(256, kernel_size=3, strides=1, padding="same")
        self.smooth1_bn = tf.keras.layers.BatchNormalization()

        self.smooth2 = tf.keras.layers.Conv2D(256, kernel_size=3, strides=1, padding="same")
        self.smooth2_bn = tf.keras.layers.BatchNormalization()

        self.smooth3 = tf.keras.layers.Conv2D(256, kernel_size=3, strides=1, padding="same")
        self.smooth3_bn = tf.keras.layers.BatchNormalization()

        # Lateral layers
        self.latlayer1 = tf.keras.layers.Conv2D(256, kernel_size=1, strides=1, padding="same")
        self.latlayer1_bn = tf.keras.layers.BatchNormalization()

        self.latlayer2 = tf.keras.layers.Conv2D(256, kernel_size=1, strides=1, padding="same")
        self.latlayer2_bn = tf.keras.layers.BatchNormalization()

        self.latlayer3 = tf.keras.layers.Conv2D(256, kernel_size=1, strides=1, padding="same")
        self.latlayer3_bn = tf.keras.layers.BatchNormalization()

        self.conv2 = tf.keras.layers.Conv2D(256, kernel_size=3, strides=1, padding="same")
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.conv3 = tf.keras.layers.Conv2D(num_classes, kernel_size=1, strides=1, padding="same")

        self.scale = scale

    def _upsample(self, x, y, scale=1):
        _, H, W, _ = y.shape
        return tf.image.resize(x, (H // scale, W // scale))

    def _upsample_add(self, x, y):
        return self._upsample(x, y) + y

    def call(self, x):
        h = x

        c2, c3, c4, c5 = self.res(h)

        # Top-down
        p5 = self.toplayer(c5)
        p5 = tf.nn.relu(self.toplayer_bn(p5))

        c4 = self.latlayer1(c4)
        c4 = tf.nn.relu(self.latlayer1_bn(c4))
        p4 = self._upsample_add(p5, c4)
        p4 = self.smooth1(p4)
        p4 = tf.nn.relu(self.smooth1_bn(p4))

        c3 = self.latlayer2(c3)
        c3 = tf.nn.relu(self.latlayer2_bn(c3))
        p3 = self._upsample_add(p4, c3)
        p3 = self.smooth2(p3)
        p3 = tf.nn.relu(self.smooth2_bn(p3))

        c2 = self.latlayer3(c2)
        c2 = tf.nn.relu(self.latlayer3_bn(c2))
        p2 = self._upsample_add(p3, c2)
        p2 = self.smooth3(p2)
        p2 = tf.nn.relu(self.smooth3_bn(p2))

        p3 = self._upsample(p3, p2)
        p4 = self._upsample(p4, p2)
        p5 = self._upsample(p5, p2)

        out = tf.concat((p2, p3, p4, p5), 3)
        out = self.conv2(out)
        out = tf.nn.relu(self.bn2(out))
        out = self.conv3(out)
        out = self._upsample(out, x, scale=self.scale)

        return out


class TFPreResNet(tf.keras.Model):
    def __init__(self, layers, num_classes=7, scale=1):
        super(TFPreResNet, self).__init__()
        
        resnet = tf.keras.applications.ResNet50V2(include_top=False, weights='imagenet')
        count = 0
        act_ind = [] #acti layer index
        for subm in resnet.submodules:
            if isinstance(subm, tf.keras.layers.Activation):
                act_ind.append(count)
            count += 1
        
        a = 0
        outputs = []
        for i in layers:
            a += 3 * i
            outputs.append(resnet.layers[act_ind[a]].output)
            
        self.res = tf.keras.Model(inputs=resnet.input, outputs=outputs)
        # self.avgpool
        # self.fc
        
        # Top layer
        self.toplayer = tf.keras.layers.Conv2D(256, kernel_size=1, strides=1, padding="same")  # Reduce channels
        self.toplayer_bn = tf.keras.layers.BatchNormalization()
        
        # Smooth layers
        self.smooth1 = tf.keras.layers.Conv2D(256, kernel_size=3, strides=1, padding="same")
        self.smooth1_bn = tf.keras.layers.BatchNormalization()

        self.smooth2 = tf.keras.layers.Conv2D(256, kernel_size=3, strides=1, padding="same")
        self.smooth2_bn = tf.keras.layers.BatchNormalization()

        self.smooth3 = tf.keras.layers.Conv2D(256, kernel_size=3, strides=1, padding="same")
        self.smooth3_bn = tf.keras.layers.BatchNormalization()

        # Lateral layers
        self.latlayer1 = tf.keras.layers.Conv2D(256, kernel_size=1, strides=1, padding="same")
        self.latlayer1_bn = tf.keras.layers.BatchNormalization()

        self.latlayer2 = tf.keras.layers.Conv2D(256, kernel_size=1, strides=1, padding="same")
        self.latlayer2_bn = tf.keras.layers.BatchNormalization()

        self.latlayer3 = tf.keras.layers.Conv2D(256, kernel_size=1, strides=1, padding="same")
        self.latlayer3_bn = tf.keras.layers.BatchNormalization()

        self.conv2 = tf.keras.layers.Conv2D(256, kernel_size=3, strides=1, padding="same")
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.conv3 = tf.keras.layers.Conv2D(num_classes, kernel_size=1, strides=1, padding="same")

        self.scale = scale
        
        
    def _upsample(self, x, y, scale=1):
        _, H, W, _ = y.shape
        return tf.image.resize(x, (H // scale, W // scale))
        
    def _upsample_add(self, x, y):
        return self._upsample(x, y) + y

    
    def call(self, x):
        h = x
        
        c2, c3, c4, c5 = self.res(h)
        
        # Top-down
        p5 = self.toplayer(c5)
        p5 = tf.nn.relu(self.toplayer_bn(p5))
        
        c4 = self.latlayer1(c4)
        c4 = tf.nn.relu(self.latlayer1_bn(c4))
        p4 = self._upsample_add(p5, c4)
        p4 = self.smooth1(p4)
        p4 = tf.nn.relu(self.smooth1_bn(p4))
        
        c3 = self.latlayer2(c3)
        c3 = tf.nn.relu(self.latlayer2_bn(c3))
        p3 = self._upsample_add(p4, c3)
        p3 = self.smooth2(p3)
        p3 = tf.nn.relu(self.smooth2_bn(p3))  
        
        c2 = self.latlayer3(c2)
        c2 = tf.nn.relu(self.latlayer3_bn(c2))
        p2 = self._upsample_add(p3, c2)
        p2 = self.smooth3(p2)
        p2 = tf.nn.relu(self.smooth3_bn(p2))

        p3 = self._upsample(p3, p2)
        p4 = self._upsample(p4, p2)
        p5 = self._upsample(p5, p2)

        out = tf.concat((p2, p3, p4, p5), 3)
        out = self.conv2(out)
        out = tf.nn.relu(self.bn2(out))
        out = self.conv3(out)
        out = self._upsample(out, x, scale=self.scale)
        
        return out


#
# #todo delete this bug class
# class MobileNetV3Large(tf.keras.Model):
#     def __init__(self,num_classes=7, scale=1):
#
#         #def __init__(self, block, layers, num_classes=7, scale=1):
#
#         super(MobileNetV3Large, self).__init__()
#         #self.conv1 = tf.keras.layers.Conv2D(64, 7, 2, padding="same", input_shape=(640, 640, 3), data_format='channels_last', use_bias=False)
#
#         self.conv1 = tf.keras.layers.Conv2D(filters=16,
#                                             kernel_size=(3, 3),
#                                             strides=2,
#                                             padding="same",input_shape=(640, 640, 3), data_format='channels_last', use_bias=False)
#         self.bn1 = tf.keras.layers.BatchNormalization()
#         self.bneck1 = BottleNeck(in_size=16, exp_size=16, out_size=16, s=1, is_se_existing=False, NL="RE", k=3)
#         self.bneck2 = BottleNeck(in_size=16, exp_size=64, out_size=24, s=2, is_se_existing=False, NL="RE", k=3)
#         self.bneck3 = BottleNeck(in_size=24, exp_size=72, out_size=24, s=1, is_se_existing=False, NL="RE", k=3)
#         self.bneck4 = BottleNeck(in_size=24, exp_size=72, out_size=40, s=2, is_se_existing=True, NL="RE", k=5)
#         self.bneck5 = BottleNeck(in_size=40, exp_size=120, out_size=40, s=1, is_se_existing=True, NL="RE", k=5)
#         self.bneck6 = BottleNeck(in_size=40, exp_size=120, out_size=40, s=1, is_se_existing=True, NL="RE", k=5)
#         self.bneck7 = BottleNeck(in_size=40, exp_size=240, out_size=80, s=2, is_se_existing=False, NL="HS", k=3)
#         self.bneck8 = BottleNeck(in_size=80, exp_size=200, out_size=80, s=1, is_se_existing=False, NL="HS", k=3)
#         self.bneck9 = BottleNeck(in_size=80, exp_size=184, out_size=80, s=1, is_se_existing=False, NL="HS", k=3)
#         self.bneck10 = BottleNeck(in_size=80, exp_size=184, out_size=80, s=1, is_se_existing=False, NL="HS", k=3)
#         self.bneck11 = BottleNeck(in_size=80, exp_size=480, out_size=112, s=1, is_se_existing=True, NL="HS", k=3)
#         self.bneck12 = BottleNeck(in_size=112, exp_size=672, out_size=112, s=1, is_se_existing=True, NL="HS", k=3)
#         self.bneck13 = BottleNeck(in_size=112, exp_size=672, out_size=160, s=2, is_se_existing=True, NL="HS", k=5)
#         self.bneck14 = BottleNeck(in_size=160, exp_size=960, out_size=160, s=1, is_se_existing=True, NL="HS", k=5)
#         #self.bneck15 = BottleNeck(in_size=160, exp_size=960, out_size=160, s=1, is_se_existing=True, NL="HS", k=5)
#
# #         self.conv2 = tf.keras.layers.Conv2D(filters=960,
# #                                             kernel_size=(1, 1),
# #                                             strides=1,
# #                                             padding="same")
# #         self.bn2 = tf.keras.layers.BatchNormalization()
# #         self.avgpool = tf.keras.layers.AveragePooling2D(pool_size=(7, 7),
# #                                                         strides=1)
# #         self.conv3 = tf.keras.layers.Conv2D(filters=1280,
# #                                             kernel_size=(1, 1),
# #                                             strides=1,
# #                                             padding="same")
# #         self.conv4 = tf.keras.layers.Conv2D(filters=NUM_CLASSES,
# #                                             kernel_size=(1, 1),
# #                                             strides=1,
# #                                             padding="same",
# #                                             activation=tf.keras.activations.softmax)
# # #！！！borrow from  resnet
#
#         # Top layer
#         self.toplayer = tf.keras.layers.Conv2D(256, kernel_size=1, strides=1, padding="same")  # Reduce channels
#         self.toplayer_bn = tf.keras.layers.BatchNormalization()
#
#         # Smooth layers
#         self.smooth1 = tf.keras.layers.Conv2D(256, kernel_size=3, strides=1, padding="same")
#         self.smooth1_bn = tf.keras.layers.BatchNormalization()
#
#         self.smooth2 = tf.keras.layers.Conv2D(256, kernel_size=3, strides=1, padding="same")
#         self.smooth2_bn = tf.keras.layers.BatchNormalization()
#
#         self.smooth3 = tf.keras.layers.Conv2D(256, kernel_size=3, strides=1, padding="same")
#         self.smooth3_bn = tf.keras.layers.BatchNormalization()
#
#         # Lateral layers
#         self.latlayer1 = tf.keras.layers.Conv2D(256, kernel_size=1, strides=1, padding="same")
#         self.latlayer1_bn = tf.keras.layers.BatchNormalization()
#
#         self.latlayer2 = tf.keras.layers.Conv2D(256, kernel_size=1, strides=1, padding="same")
#         self.latlayer2_bn = tf.keras.layers.BatchNormalization()
#
#         self.latlayer3 = tf.keras.layers.Conv2D(256, kernel_size=1, strides=1, padding="same")
#         self.latlayer3_bn = tf.keras.layers.BatchNormalization()
#
#         self.conv2 = tf.keras.layers.Conv2D(256, kernel_size=3, strides=1, padding="same")
#         self.bn2 = tf.keras.layers.BatchNormalization()
#         self.conv3 = tf.keras.layers.Conv2D(num_classes, kernel_size=1, strides=1, padding="same")
#
#         self.scale = scale
#
#     def _upsample(self, x, y, scale=1):
#         _, H, W, _ = y.shape
#         return tf.image.resize(x, (H // scale, W // scale))
#
#     def _upsample_add(self, x, y):
#         return self._upsample(x, y) + y
#
#     # ！！！borrow from  resnet
#
#     def call(self, inputs, training=None, mask=None):
#         x_i=inputs
#         x = self.conv1(inputs)
#         x = self.bn1(x, training=training)
#         x = h_swish(x)
#
#         x = self.bneck1(x, training=training)
#         x = self.bneck2(x, training=training)
#         c2 = x
#
#         x = self.bneck3(x, training=training)
#         x = self.bneck4(x, training=training)
#         c3 = x
#         x = self.bneck5(x, training=training)
#         x = self.bneck6(x, training=training)
#         x = self.bneck7(x, training=training)
#         c4 = x
#
#         x = self.bneck8(x, training=training)
#         x = self.bneck9(x, training=training)
#         x = self.bneck10(x, training=training)
#         x = self.bneck11(x, training=training)
#         x = self.bneck12(x, training=training)
#         x = self.bneck13(x, training=training)
#         x = self.bneck14(x, training=training)
#         c5 = x
#
#         # x = self.bneck15(c5, training=training)
#         #
#         # x = self.conv2(x)
#         # x = self.bn2(x, training=training)
#         # x = h_swish(x)
#         # x = self.avgpool(x)
#         # x = self.conv3(x)
#         # x = h_swish(x)
#         # x = self.conv4(x)
#
#         # Top-down
#         p5 = self.toplayer(c5)
#         p5 = tf.nn.relu(self.toplayer_bn(p5))
#
#         c4 = self.latlayer1(c4)
#         c4 = tf.nn.relu(self.latlayer1_bn(c4))
#         p4 = self._upsample_add(p5, c4)
#         p4 = self.smooth1(p4)
#         p4 = tf.nn.relu(self.smooth1_bn(p4))
#
#         c3 = self.latlayer2(c3)
#         c3 = tf.nn.relu(self.latlayer2_bn(c3))
#         p3 = self._upsample_add(p4, c3)
#         p3 = self.smooth2(p3)
#         p3 = tf.nn.relu(self.smooth2_bn(p3))
#
#         c2 = self.latlayer3(c2)
#         c2 = tf.nn.relu(self.latlayer3_bn(c2))
#         p2 = self._upsample_add(p3, c2)
#         p2 = self.smooth3(p2)
#         p2 = tf.nn.relu(self.smooth3_bn(p2))
#
#         p3 = self._upsample(p3, p2)
#         p4 = self._upsample(p4, p2)
#         p5 = self._upsample(p5, p2)
#
#         out = tf.concat((p2, p3, p4, p5), 3)
#         out = self.conv2(out)
#         out = tf.nn.relu(self.bn2(out))
#         out = self.conv3(out)
#         out = self._upsample(out, x_i, scale=self.scale)
# #mb large
#         return out


#get tensor just before_stride2 downsample
class MobileNetV3Large(tf.keras.Model):
    def __init__(self,num_classes=7, scale=1):

        #def __init__(self, block, layers, num_classes=7, scale=1):

        super(MobileNetV3Large, self).__init__()
        #self.conv1 = tf.keras.layers.Conv2D(64, 7, 2, padding="same", input_shape=(640, 640, 3), data_format='channels_last', use_bias=False)

        self.conv1 = tf.keras.layers.Conv2D(filters=16,
                                            kernel_size=(3, 3),
                                            strides=2,
                                            padding="same",input_shape=(640, 640, 3), data_format='channels_last', use_bias=False)
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.bneck1 = BottleNeck(in_size=16, exp_size=16, out_size=16, s=1, is_se_existing=False, NL="RE", k=3)
        self.bneck2 = BottleNeck(in_size=16, exp_size=64, out_size=24, s=2, is_se_existing=False, NL="RE", k=3)
        self.bneck3 = BottleNeck(in_size=24, exp_size=72, out_size=24, s=1, is_se_existing=False, NL="RE", k=3)
        self.bneck4 = BottleNeck(in_size=24, exp_size=72, out_size=40, s=2, is_se_existing=True, NL="RE", k=5)
        self.bneck5 = BottleNeck(in_size=40, exp_size=120, out_size=40, s=1, is_se_existing=True, NL="RE", k=5)
        self.bneck6 = BottleNeck(in_size=40, exp_size=120, out_size=40, s=1, is_se_existing=True, NL="RE", k=5)
        self.bneck7 = BottleNeck(in_size=40, exp_size=240, out_size=80, s=2, is_se_existing=False, NL="HS", k=3)
        self.bneck8 = BottleNeck(in_size=80, exp_size=200, out_size=80, s=1, is_se_existing=False, NL="HS", k=3)
        self.bneck9 = BottleNeck(in_size=80, exp_size=184, out_size=80, s=1, is_se_existing=False, NL="HS", k=3)
        self.bneck10 = BottleNeck(in_size=80, exp_size=184, out_size=80, s=1, is_se_existing=False, NL="HS", k=3)
        self.bneck11 = BottleNeck(in_size=80, exp_size=480, out_size=112, s=1, is_se_existing=True, NL="HS", k=3)
        self.bneck12 = BottleNeck(in_size=112, exp_size=672, out_size=112, s=1, is_se_existing=True, NL="HS", k=3)
        self.bneck13 = BottleNeck(in_size=112, exp_size=672, out_size=160, s=2, is_se_existing=True, NL="HS", k=5)
        self.bneck14 = BottleNeck(in_size=160, exp_size=960, out_size=160, s=1, is_se_existing=True, NL="HS", k=5)
        self.bneck15 = BottleNeck(in_size=160, exp_size=960, out_size=160, s=1, is_se_existing=True, NL="HS", k=5)

        self.conv2 = tf.keras.layers.Conv2D(filters=960,
                                            kernel_size=(1, 1),
                                            strides=1,
                                            padding="same")
        self.bn2 = tf.keras.layers.BatchNormalization()

        # Top layer
        self.toplayer = tf.keras.layers.Conv2D(256, kernel_size=1, strides=1, padding="same")  # Reduce channels
        self.toplayer_bn = tf.keras.layers.BatchNormalization()

        # Smooth layers
        self.smooth1 = tf.keras.layers.Conv2D(256, kernel_size=3, strides=1, padding="same")
        self.smooth1_bn = tf.keras.layers.BatchNormalization()

        self.smooth2 = tf.keras.layers.Conv2D(256, kernel_size=3, strides=1, padding="same")
        self.smooth2_bn = tf.keras.layers.BatchNormalization()

        self.smooth3 = tf.keras.layers.Conv2D(256, kernel_size=3, strides=1, padding="same")
        self.smooth3_bn = tf.keras.layers.BatchNormalization()

        # Lateral layers
        self.latlayer1 = tf.keras.layers.Conv2D(256, kernel_size=1, strides=1, padding="same")
        self.latlayer1_bn = tf.keras.layers.BatchNormalization()

        self.latlayer2 = tf.keras.layers.Conv2D(256, kernel_size=1, strides=1, padding="same")
        self.latlayer2_bn = tf.keras.layers.BatchNormalization()

        self.latlayer3 = tf.keras.layers.Conv2D(256, kernel_size=1, strides=1, padding="same")
        self.latlayer3_bn = tf.keras.layers.BatchNormalization()

        self.conv3 = tf.keras.layers.Conv2D(256, kernel_size=3, strides=1, padding="same")
        self.bn3 = tf.keras.layers.BatchNormalization()
        self.conv4 = tf.keras.layers.Conv2D(num_classes, kernel_size=1, strides=1, padding="same")

        self.scale = scale

    def _upsample(self, x, y, scale=1):
        _, H, W, _ = y.shape
        return tf.image.resize(x, (H // scale, W // scale))

    def _upsample_add(self, x, y):
        return self._upsample(x, y) + y

    # ！！！borrow from  resnet

    def call(self, inputs, training=None, mask=None):
        x_i=inputs
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = h_swish(x)

        x = self.bneck1(x, training=training)
        x = self.bneck2(x, training=training)
        x = self.bneck3(x, training=training)
        c2 = x
        x = self.bneck4(x, training=training)
        x = self.bneck5(x, training=training)
        x = self.bneck6(x, training=training)
        c3 = x
        x = self.bneck7(x, training=training)
        x = self.bneck8(x, training=training)

        x = self.bneck9(x, training=training)
        x = self.bneck10(x, training=training)
        x = self.bneck11(x, training=training)
        x = self.bneck12(x, training=training)
        x = self.bneck13(x, training=training)
        c4 = x
        x = self.bneck14(x, training=training)
        x = self.bneck15(x, training=training)

        x = self.conv2(x)
        x = self.bn2(x, training=training)
        x = h_swish(x)
        c5 = x

        # x = self.avgpool(x)
        # x = self.conv3(x)
        # x = h_swish(x)
        # x = self.conv4(x)

        # Top-down
        p5 = self.toplayer(c5)
        p5 = tf.nn.relu(self.toplayer_bn(p5))

        c4 = self.latlayer1(c4)
        c4 = tf.nn.relu(self.latlayer1_bn(c4))
        p4 = self._upsample_add(p5, c4)
        p4 = self.smooth1(p4)
        p4 = tf.nn.relu(self.smooth1_bn(p4))

        c3 = self.latlayer2(c3)
        c3 = tf.nn.relu(self.latlayer2_bn(c3))
        p3 = self._upsample_add(p4, c3)
        p3 = self.smooth2(p3)
        p3 = tf.nn.relu(self.smooth2_bn(p3))

        c2 = self.latlayer3(c2)
        c2 = tf.nn.relu(self.latlayer3_bn(c2))
        p2 = self._upsample_add(p3, c2)
        p2 = self.smooth3(p2)
        p2 = tf.nn.relu(self.smooth3_bn(p2))

        p3 = self._upsample(p3, p2)
        p4 = self._upsample(p4, p2)
        p5 = self._upsample(p5, p2)

        out = tf.concat((p2, p3, p4, p5), 3)
        out = self.conv3(out)
        out = tf.nn.relu(self.bn3(out))
        out = self.conv4(out)
        out = self._upsample(out, x_i, scale=self.scale)
#mb large
        return out







class MobileNetV3Small(tf.keras.Model):
    def __init__(self,num_classes=7, scale=1):
        super(MobileNetV3Small, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(filters=16,
                                            kernel_size=(3, 3),
                                            strides=2,
                                            padding="same",input_shape=(640, 640, 3), data_format='channels_last', use_bias=False)
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.bneck1 = BottleNeck(in_size=16, exp_size=16, out_size=16, s=2, is_se_existing=True, NL="RE", k=3)
        self.bneck2 = BottleNeck(in_size=16, exp_size=72, out_size=24, s=2, is_se_existing=False, NL="RE", k=3)
        self.bneck3 = BottleNeck(in_size=24, exp_size=88, out_size=24, s=1, is_se_existing=False, NL="RE", k=3)
        self.bneck4 = BottleNeck(in_size=24, exp_size=96, out_size=40, s=2, is_se_existing=True, NL="HS", k=5)
        self.bneck5 = BottleNeck(in_size=40, exp_size=240, out_size=40, s=1, is_se_existing=True, NL="HS", k=5)
        self.bneck6 = BottleNeck(in_size=40, exp_size=240, out_size=40, s=1, is_se_existing=True, NL="HS", k=5)
        self.bneck7 = BottleNeck(in_size=40, exp_size=120, out_size=48, s=1, is_se_existing=True, NL="HS", k=5)
        self.bneck8 = BottleNeck(in_size=48, exp_size=144, out_size=48, s=1, is_se_existing=True, NL="HS", k=5)
        self.bneck9 = BottleNeck(in_size=48, exp_size=288, out_size=96, s=2, is_se_existing=True, NL="HS", k=5)
        self.bneck10 = BottleNeck(in_size=96, exp_size=576, out_size=96, s=1, is_se_existing=True, NL="HS", k=5)
        self.bneck11 = BottleNeck(in_size=96, exp_size=576, out_size=96, s=1, is_se_existing=True, NL="HS", k=5)

        self.conv2 = tf.keras.layers.Conv2D(filters=576,
                                            kernel_size=(1, 1),
                                            strides=1,
                                            padding="same")
        self.bn2 = tf.keras.layers.BatchNormalization()
        # self.avgpool = tf.keras.layers.AveragePooling2D(pool_size=(7, 7),
        #                                                 strides=1)
        # self.conv3 = tf.keras.layers.Conv2D(filters=1280,
        #                                     kernel_size=(1, 1),
        #                                     strides=1,
        #                                     padding="same")
        # self.conv4 = tf.keras.layers.Conv2D(filters=NUM_CLASSES,
        #                                     kernel_size=(1, 1),
        #                                     strides=1,
        #                                     padding="same",
        #                                     activation=tf.keras.activations.softmax)

        # Top layer
        self.toplayer = tf.keras.layers.Conv2D(256, kernel_size=1, strides=1, padding="same")  # Reduce channels
        self.toplayer_bn = tf.keras.layers.BatchNormalization()

        # Smooth layers
        self.smooth1 = tf.keras.layers.Conv2D(256, kernel_size=3, strides=1, padding="same")
        self.smooth1_bn = tf.keras.layers.BatchNormalization()

        self.smooth2 = tf.keras.layers.Conv2D(256, kernel_size=3, strides=1, padding="same")
        self.smooth2_bn = tf.keras.layers.BatchNormalization()

        self.smooth3 = tf.keras.layers.Conv2D(256, kernel_size=3, strides=1, padding="same")
        self.smooth3_bn = tf.keras.layers.BatchNormalization()

        # Lateral layers
        self.latlayer1 = tf.keras.layers.Conv2D(256, kernel_size=1, strides=1, padding="same")
        self.latlayer1_bn = tf.keras.layers.BatchNormalization()

        self.latlayer2 = tf.keras.layers.Conv2D(256, kernel_size=1, strides=1, padding="same")
        self.latlayer2_bn = tf.keras.layers.BatchNormalization()

        self.latlayer3 = tf.keras.layers.Conv2D(256, kernel_size=1, strides=1, padding="same")
        self.latlayer3_bn = tf.keras.layers.BatchNormalization()

        self.conv3 = tf.keras.layers.Conv2D(256, kernel_size=3, strides=1, padding="same")
        self.bn3 = tf.keras.layers.BatchNormalization()
        self.conv4 = tf.keras.layers.Conv2D(num_classes, kernel_size=1, strides=1, padding="same")

        self.scale = scale

    def _upsample(self, x, y, scale=1):
        _, H, W, _ = y.shape
        return tf.image.resize(x, (H // scale, W // scale))

    def _upsample_add(self, x, y):
        return self._upsample(x, y) + y



    def call(self, inputs, training=None, mask=None):
        x_i= inputs
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = h_swish(x)

        x = self.bneck1(x, training=training)
        c2 = x
        x = self.bneck2(x, training=training)
        x = self.bneck3(x, training=training)
        c3 = x
        x = self.bneck4(x, training=training)
        x = self.bneck5(x, training=training)
        x = self.bneck6(x, training=training)
        x = self.bneck7(x, training=training)
        x = self.bneck8(x, training=training)
        c4 = x
        x = self.bneck9(x, training=training)

        x = self.bneck10(x, training=training)
        x = self.bneck11(x, training=training)
        x = self.conv2(x)
        x = self.bn2(x, training=training)
        x = h_swish(x)
        c5 = x


        # Top-down
        p5 = self.toplayer(c5)
        p5 = tf.nn.relu(self.toplayer_bn(p5))

        c4 = self.latlayer1(c4)
        c4 = tf.nn.relu(self.latlayer1_bn(c4))
        p4 = self._upsample_add(p5, c4)
        p4 = self.smooth1(p4)
        p4 = tf.nn.relu(self.smooth1_bn(p4))

        c3 = self.latlayer2(c3)
        c3 = tf.nn.relu(self.latlayer2_bn(c3))
        p3 = self._upsample_add(p4, c3)
        p3 = self.smooth2(p3)
        p3 = tf.nn.relu(self.smooth2_bn(p3))

        c2 = self.latlayer3(c2)
        c2 = tf.nn.relu(self.latlayer3_bn(c2))
        p2 = self._upsample_add(p3, c2)
        p2 = self.smooth3(p2)
        p2 = tf.nn.relu(self.smooth3_bn(p2))

        p3 = self._upsample(p3, p2)
        p4 = self._upsample(p4, p2)
        p5 = self._upsample(p5, p2)

        out = tf.concat((p2, p3, p4, p5), 3)
        out = self.conv3(out)
        out = tf.nn.relu(self.bn3(out))
        out = self.conv4(out)
        out = self._upsample(out, x_i, scale=self.scale)
        # mb small
        return out


# class MobileNetV2_scratch(tf.keras.Model):
#     def __init__(self,num_classes=7, scale=1):
#         super(MobileNetV2, self).__init__()
#         self.conv1 = tf.keras.layers.Conv2D(filters=16,
#                                             kernel_size=(3, 3),
#                                             strides=2,
#                                             padding="same",input_shape=(640, 640, 3), data_format='channels_last', use_bias=False)
#
#
#
#
#
#
#
#
#
#
#         self.conv2 = tf.keras.layers.Conv2D(filters=576,
#                                             kernel_size=(1, 1),
#                                             strides=1,
#                                             padding="same")
#         self.bn2 = tf.keras.layers.BatchNormalization()
#         # self.avgpool = tf.keras.layers.AveragePooling2D(pool_size=(7, 7),
#         #                                                 strides=1)
#         # self.conv3 = tf.keras.layers.Conv2D(filters=1280,
#         #                                     kernel_size=(1, 1),
#         #                                     strides=1,
#         #                                     padding="same")
#         # self.conv4 = tf.keras.layers.Conv2D(filters=NUM_CLASSES,
#         #                                     kernel_size=(1, 1),
#         #                                     strides=1,
#         #                                     padding="same",
#         #                                     activation=tf.keras.activations.softmax)
#
#         # Top layer
#         self.toplayer = tf.keras.layers.Conv2D(256, kernel_size=1, strides=1, padding="same")  # Reduce channels
#         self.toplayer_bn = tf.keras.layers.BatchNormalization()
#
#         # Smooth layers
#         self.smooth1 = tf.keras.layers.Conv2D(256, kernel_size=3, strides=1, padding="same")
#         self.smooth1_bn = tf.keras.layers.BatchNormalization()
#
#         self.smooth2 = tf.keras.layers.Conv2D(256, kernel_size=3, strides=1, padding="same")
#         self.smooth2_bn = tf.keras.layers.BatchNormalization()
#
#         self.smooth3 = tf.keras.layers.Conv2D(256, kernel_size=3, strides=1, padding="same")
#         self.smooth3_bn = tf.keras.layers.BatchNormalization()
#
#         # Lateral layers
#         self.latlayer1 = tf.keras.layers.Conv2D(256, kernel_size=1, strides=1, padding="same")
#         self.latlayer1_bn = tf.keras.layers.BatchNormalization()
#
#         self.latlayer2 = tf.keras.layers.Conv2D(256, kernel_size=1, strides=1, padding="same")
#         self.latlayer2_bn = tf.keras.layers.BatchNormalization()
#
#         self.latlayer3 = tf.keras.layers.Conv2D(256, kernel_size=1, strides=1, padding="same")
#         self.latlayer3_bn = tf.keras.layers.BatchNormalization()
#
#         self.conv3 = tf.keras.layers.Conv2D(256, kernel_size=3, strides=1, padding="same")
#         self.bn3 = tf.keras.layers.BatchNormalization()
#         self.conv4 = tf.keras.layers.Conv2D(num_classes, kernel_size=1, strides=1, padding="same")
#
#         self.scale = scale
#
#     def _upsample(self, x, y, scale=1):
#         _, H, W, _ = y.shape
#         return tf.image.resize(x, (H // scale, W // scale))
#
#     def _upsample_add(self, x, y):
#         return self._upsample(x, y) + y
#
#
#     def call(self, inputs, training=None, mask=None):
#         x_i= inputs
#         x = self.conv1(inputs)
#         x = self.bn1(x, training=training)
#         x = h_swish(x)
#
#         x = self.bneck1(x, training=training)
#         c2 = x
#         x = self.bneck2(x, training=training)
#         x = self.bneck3(x, training=training)
#         c3 = x
#         x = self.bneck4(x, training=training)
#         x = self.bneck5(x, training=training)
#         x = self.bneck6(x, training=training)
#         x = self.bneck7(x, training=training)
#         x = self.bneck8(x, training=training)
#         c4 = x
#         x = self.bneck9(x, training=training)
#
#         x = self.bneck10(x, training=training)
#         x = self.bneck11(x, training=training)
#         x = self.conv2(x)
#         x = self.bn2(x, training=training)
#         x = h_swish(x)
#         c5 = x
#
#
#         # Top-down
#         p5 = self.toplayer(c5)
#         p5 = tf.nn.relu(self.toplayer_bn(p5))
#
#         c4 = self.latlayer1(c4)
#         c4 = tf.nn.relu(self.latlayer1_bn(c4))
#         p4 = self._upsample_add(p5, c4)
#         p4 = self.smooth1(p4)
#         p4 = tf.nn.relu(self.smooth1_bn(p4))
#
#         c3 = self.latlayer2(c3)
#         c3 = tf.nn.relu(self.latlayer2_bn(c3))
#         p3 = self._upsample_add(p4, c3)
#         p3 = self.smooth2(p3)
#         p3 = tf.nn.relu(self.smooth2_bn(p3))
#
#         c2 = self.latlayer3(c2)
#         c2 = tf.nn.relu(self.latlayer3_bn(c2))
#         p2 = self._upsample_add(p3, c2)
#         p2 = self.smooth3(p2)
#         p2 = tf.nn.relu(self.smooth3_bn(p2))
#
#         p3 = self._upsample(p3, p2)
#         p4 = self._upsample(p4, p2)
#         p5 = self._upsample(p5, p2)
#
#         out = tf.concat((p2, p3, p4, p5), 3)
#         out = self.conv3(out)
#         out = tf.nn.relu(self.bn3(out))
#         out = self.conv4(out)
#         out = self._upsample(out, x_i, scale=self.scale)
#         # mb small
#         return out
#
#
#


def mobilenetv2(num_classes=7, scale=1, **kwargs):
    #return TFPreMobileNetV2()
    return TFPreMobileNetV2()

def mobilenetv3_small(num_classes=7, scale=1, **kwargs):
    return MobileNetV3Small()

def mobilenetv3_large(num_classes=7, scale=1, **kwargs):
    return MobileNetV3Large()

        
def resnet18(pretrained=False, **kwargs):
    return ResNet(BasicBlock, [2,2,2,2], **kwargs)
    
def resnet34(pretrained=False, **kwargs):
    return ResNet(BasicBlock, [3,4,6,3], **kwargs)

def resnet50(pretrained=True, **kwargs):
    if pretrained:
        return TFPreResNet([3,4,6,3], **kwargs)
    return ResNet(Bottleneck, [3,4,6,3], **kwargs)

def resnet101(pretrained=True, **kwargs):
    if pretrained:
        return TFPreResNet([3,4,23,3], **kwargs)
    return ResNet(Bottleneck, [3,4,23,3], **kwargs)

def resnet152(pretrained=True, **kwargs):
    if pretrained:
        return TFPreResNet([3,8,36,3], **kwargs)
    return ResNet(Bottleneck, [3,8,36,3], **kwargs)




if __name__ == '__main__':
    model = MobileNetV3Large()
    model.build(input_shape=(None, 224, 224, 3))
    model.summary()