import numpy as np


class ConvolutionalLayer(object):
    def __init__(self,kernel_size,channel_in,channel_out,padding,stride):
        self.kernel_size = kernel_size
        self.channel_in = channel_in
        self.channel_out = channel_out
        self.padding = padding
        self.stride = stride

    def init_param(self,std = 0.01):
        self.weight = np.random.normal(loc=0.0,scale=std,size=[self.channel_in,self.kernel_size,self.kernel_size,self.channel_out])
        self.bias = np.zeros([self.channel_out])

    def forward(self,input):
        self.input = input # N C H W
        height = self.input.shape[2] + 2 * self.padding
        width = self.input.shape[3] + 2 * self.padding
        self.input_pad = np.zeros([input.shape[0],input.shape[1],height,width])
        self.input_pad[:,:,self.padding:self.padding+input.shape[2],self.padding:self.padding+input.shape[3]] = self.input
        height_out = (input.shape[2] + 2 * self.padding - self.kernel_size) // self.stride + 1
        width_out = (input.shape[3] + 2 * self.padding -self.kernel_size) //  self.stride + 1
        self.output = np.zeros([input.shape[0],self.channel_out,height_out,width_out])
        for idn in range(input.shape[0]):
            for idc in range(self.channel_out):
                for idh in range(height_out):
                    for idw in range(width_out):
                        self.output[idn,idc,idh,idw] = np.sum(self.weight[:,:,:,idc] * self.input_pad[idn, : ,idh * self.stride : idh * self.stride + self.kernel_size,idw * self.stride : idw * self.stride + self.kernel_size]) + self.bias[idc]

        return self.output


    def backward(self):
        pass

    def load_param(self,weight,bias):
        assert self.weight.shape == weight.shape
        assert self.bias.shape == bias.shape
        self.weight = weight
        self.bias = bias

    def update_param(self):
        pass


class MaxPoolingLayer(object):
    def __init__(self,kernel_size,stride):
        self.kernel_size = kernel_size
        self.stride = stride


    def init_param(self):
        pass

    def forward(self,input):
        self.input = input # N C H W
        height_out = (input.shape[2] - self.kernel_size) // self.stride + 1
        width_out = (input.shape[3] - self.kernel_size) // self.stride + 1
        self.output = np.zeros([input.shape[0],input.shape[1],height_out,width_out])
        for idxn  in range(input.shape[0]):
            for idxc in range(input.shape[1]):
                for idxh in range(height_out):
                    for idxw in range(width_out):
                        self.output[idxn,idxc,idxh,idxw] = self.input[idxn,idxc,idxh * self.stride : idxh * self.stride + self.kernel_size, idxw * self.stride: idxw * self.stride + self.kernel_size].max()

        return self.output

    def backward(self):
        pass

    def update_param(self):
        pass

    def load_param(self):
        pass


class FlattenLayer(object):
    def __init__(self,input_shape,output_shape):
        self.input_shape = input_shape
        self.output_shape = output_shape


    def init_param(self):
        pass

    def forward(self,input):
        # input shape is N C H W
        # MatConvNet shape is N H W C
        self.input = np.transpose(input,[0,2,3,1])
        self.output = self.input.reshape([self.input.shape[0]] + list(self.output_shape))
        return self.output

    def backward(self):
        pass
