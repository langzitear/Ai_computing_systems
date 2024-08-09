import imageio.v2 as imageio
import numpy as np
from skimage.transform import resize
import scipy.io

import time

from layer.layer1 import FullyConnectedLayer,SoftmaxLossLayer,ReLULayer
from layer. layer2 import ConvolutionalLayer,MaxPoolingLayer,FlattenLayer


class VGG19(object):
    def __init__(self,param_path='./mat/imagenet-vgg-verydeep-19.mat'):
        self.param_path = param_path
        self.param_layer_name = (
            'conv1_1','relu1_1','conv1_2','relu1_2','pool1',
            'conv2_1','relu2_1','conv2_2','relu2_2','pool2',
            'conv3_1','relu3_1','conv3_2','relu3_2','conv3_3','relu3_3','conv3_4','relu3_4','pool3',
            'conv4_1','relu4_1','conv4_2','relu4_2','conv4_3','relu4_3','conv4_4','relu4_4','pool4',
            'conv5_1','relu5_1','conv5_2','relu5_2','conv5_3','relu5_3','conv5_4','relu5_4','pool5',
            'flatten','fc6','relu6',
            'fc7','relu7',
            'fc8','Softmax'
        )

    def load_data(self):
        pass

    def init_param(self):
        pass

    def build_model(self):
        self.layers = {}
        self.layers['conv1_1'] = ConvolutionalLayer(kernel_size=3,channel_in=3,channel_out=64,padding=1,stride=1)
        self.layers['relu1_1'] = ReLULayer()
        self.layers['conv1_2'] = ConvolutionalLayer(kernel_size=3,channel_in=64,channel_out=64,padding=1,stride=1)
        self.layers['relu1_2'] = ReLULayer()
        self.layers['pool1'] = MaxPoolingLayer(kernel_size = 2, stride = 2)

        self.layers['conv2_1'] = ConvolutionalLayer(kernel_size=3,channel_in=64,channel_out=128,padding=1,stride=1)
        self.layers['relu2_1'] = ReLULayer()
        self.layers['conv2_2'] = ConvolutionalLayer(kernel_size=3,channel_in=128,channel_out=128,padding=1,stride=1)
        self.layers['relu2_2'] = ReLULayer()
        self.layers['pool2'] = MaxPoolingLayer(kernel_size = 2, stride = 2)

        self.layers['conv3_1'] = ConvolutionalLayer(kernel_size=3,channel_in=128,channel_out=256,padding=1,stride=1)
        self.layers['relu3_1'] = ReLULayer()
        self.layers['conv3_2'] = ConvolutionalLayer(kernel_size=3,channel_in=256,channel_out=256,padding=1,stride=1)
        self.layers['relu3_2'] = ReLULayer()
        self.layers['conv3_3'] = ConvolutionalLayer(kernel_size=3,channel_in=256,channel_out=256,padding=1,stride=1)
        self.layers['relu3_3'] = ReLULayer()
        self.layers['conv3_4'] = ConvolutionalLayer(kernel_size=3,channel_in=256,channel_out=256,padding=1,stride=1)
        self.layers['relu3_4'] = ReLULayer()
        self.layers['pool3'] = MaxPoolingLayer(kernel_size = 2, stride = 2)

        self.layers['conv4_1'] = ConvolutionalLayer(kernel_size=3,channel_in=256,channel_out=512,padding=1,stride=1)
        self.layers['relu4_1'] = ReLULayer()
        self.layers['conv4_2'] = ConvolutionalLayer(kernel_size=3,channel_in=512,channel_out=512,padding=1,stride=1)
        self.layers['relu4_2'] = ReLULayer()
        self.layers['conv4_3'] = ConvolutionalLayer(kernel_size=3,channel_in=512,channel_out=512,padding=1,stride=1)
        self.layers['relu4_3'] = ReLULayer()
        self.layers['conv4_4'] = ConvolutionalLayer(kernel_size=3,channel_in=512,channel_out=512,padding=1,stride=1)
        self.layers['relu4_4'] = ReLULayer()
        self.layers['pool4'] = MaxPoolingLayer(kernel_size = 2, stride = 2)

        self.layers['conv5_1'] = ConvolutionalLayer(kernel_size=3,channel_in=512,channel_out=512,padding=1,stride=1)
        self.layers['relu5_1'] = ReLULayer()
        self.layers['conv5_2'] = ConvolutionalLayer(kernel_size=3,channel_in=512,channel_out=512,padding=1,stride=1)
        self.layers['relu5_2'] = ReLULayer()
        self.layers['conv5_3'] = ConvolutionalLayer(kernel_size=3,channel_in=512,channel_out=512,padding=1,stride=1)
        self.layers['relu5_3'] = ReLULayer()
        self.layers['conv5_4'] = ConvolutionalLayer(kernel_size=3,channel_in=512,channel_out=512,padding=1,stride=1)
        self.layers['relu5_4'] = ReLULayer()
        self.layers['pool5'] = MaxPoolingLayer(kernel_size = 2, stride = 2)

        self.layers['flatten'] = FlattenLayer(input_shape=[512,7,7],output_shape=[512*7*7])
        self.layers['fc6'] = FullyConnectedLayer(input_num=512*7*7,out_num=4096)
        self.layers['relu6'] = ReLULayer()

        self.layers['fc7'] = FullyConnectedLayer(input_num=4096,out_num=4096)
        self.layers['relu7'] = ReLULayer()

        self.layers['fc8'] = FullyConnectedLayer(input_num=4096, out_num=1000)
        self.layers['Softmax'] = SoftmaxLossLayer()

        self.update_layer_list = []
        for layer_name in self.layers.keys():
            if 'conv' in layer_name or 'fc' in layer_name:
                self.update_layer_list.append(layer_name)



    def forward(self):   # TODO：神经网络的前向传播
        print('Inferencing...')
        start_time = time.time()
        current = self.input_image
        for idx in range(len(self.param_layer_name)):
            print('Inferencing layer: ' + self.param_layer_name[idx])
            current = self.layers[self.param_layer_name[idx]].forward(current)
        print('Inference time: %f' % (time.time()-start_time))
        return current

    def init_model(self):
        print('Initializing parameters of each layer in vgg-19...')
        for layer_name in self.update_layer_list:
            self.layers[layer_name].init_param()

    def load_model(self):
        print('Loading parameters from file ' + self.param_path)
        params = scipy.io.loadmat(self.param_path)
        #print(params)
        self.image_mean = params['normalization'][0][0][0]
        self.image_mean = np.mean(self.image_mean, axis=(0, 1))
        print('Get image mean: ' + str(self.image_mean))

        for idx in range(43):
            if 'conv' in self.param_layer_name[idx]:
                weight, bias = params['layers'][0][idx][0][0][0][0]
                # matconvnet: weights dim [height, width, in_channel, out_channel]
                # ours: weights dim [in_channel, height, width, out_channel]
                # TODO：调整参数的形状
                weight = np.transpose(weight, [2, 0, 1, 3])
                bias = bias.reshape(-1)
                self.layers[self.param_layer_name[idx]].load_param(weight, bias)
            if idx >= 37 and 'fc' in self.param_layer_name[idx]:
                weight, bias = params['layers'][0][idx-1][0][0][0][0]
                weight = weight.reshape([weight.shape[0] * weight.shape[1] * weight.shape[2], weight.shape[3]])
                self.layers[self.param_layer_name[idx]].load_param(weight, bias)

    def backward(self):
        pass

    def update_param(self):
        pass

    # def load_image(self, image_dir):
    #     print('Loading and preprocessing image from ' + image_dir)
    #     self.input_image = scipy.misc.imread(image_dir)
    #     self.input_image = scipy.misc.imresize(self.input_image,[224,224,3])
    #     self.input_image = np.array(self.input_image).astype(np.float32)
    #     self.input_image -= self.image_mean
    #     self.input_image = np.reshape(self.input_image, [1]+list(self.input_image.shape))
    #     # input dim [N, channel, height, width]
    #     # TODO：调整图片维度顺序
    #     self.input_image = np.transpose(self.input_image, [0, 3, 1, 2])

    def load_image(self, image_dir):
        print('Loading and preprocessing image from ' + image_dir)
        self.input_image = imageio.imread(image_dir)
        # self.input_image = resize(self.input_image, (224, 224, 3), anti_aliasing=True)
        self.input_image = resize(self.input_image, (224, 224, 3), anti_aliasing=True, preserve_range=True)
        self.input_image = np.array(self.input_image).astype(np.float32)
        self.input_image -= self.image_mean
        self.input_image = np.reshape(self.input_image, [1] + list(self.input_image.shape))
        # input dim [N, channel, height, width]
        # 调整图片维度顺序
        self.input_image = np.transpose(self.input_image, [0, 3, 1, 2])

    def save_model(self):
        pass

    def train(self):
        pass

    def evaluate(self):
        # TODO：获取神经网络前向传播的结果
        prob = self.forward()
        top1 = np.argmax(prob[0])

        print('Classification result: id = %d, prob = %f' % (top1, prob[0, top1]))



if __name__ == "__main__":
    vgg19 = VGG19()
    vgg19.build_model()
    vgg19.init_model()
    vgg19.load_model()
    vgg19.load_image('./image/cat3.jpg')

    prob = vgg19.evaluate()
