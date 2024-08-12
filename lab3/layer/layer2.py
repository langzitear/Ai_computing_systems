import numpy as np
import time


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

    # def forward(self,input):
    #     self.input = input # N C H W
    #     height = self.input.shape[2] + 2 * self.padding
    #     width = self.input.shape[3] + 2 * self.padding
    #     self.input_pad = np.zeros([input.shape[0],input.shape[1],height,width])
    #     self.input_pad[:,:,self.padding:self.padding+input.shape[2],self.padding:self.padding+input.shape[3]] = self.input
    #     height_out = (input.shape[2] + 2 * self.padding - self.kernel_size) // self.stride + 1
    #     width_out = (input.shape[3] + 2 * self.padding -self.kernel_size) //  self.stride + 1
    #     self.output = np.zeros([input.shape[0],self.channel_out,height_out,width_out])
    #     for idn in range(input.shape[0]):
    #         for idc in range(self.channel_out):
    #             for idh in range(height_out):
    #                 for idw in range(width_out):
    #                     self.output[idn,idc,idh,idw] = np.sum(self.weight[:,:,:,idc] * self.input_pad[idn, : ,idh * self.stride : idh * self.stride + self.kernel_size,idw * self.stride : idw * self.stride + self.kernel_size]) + self.bias[idc]
    #
    #     return self.output


    def forward(self, input):  # 前向传播的计算
        start_time = time.time()
        self.input = input # [N, C, H, W]
        # TODO: 边界扩充
        height = input.shape[2] + 2 * self.padding
        width = input.shape[3] + 2 * self.padding
        self.input_pad = np.zeros([self.input.shape[0], self.input.shape[1], height, width])
        self.input_pad[:, :, self.padding: self.padding + input.shape[2], self.padding: self.padding + input.shape[3]] = self.input
        height_out = (height - self.kernel_size) // self.stride + 1
        width_out = (width - self.kernel_size) // self.stride + 1
        mat_w = self.kernel_size * self.kernel_size * self.channel_in
        mat_h = height_out * width_out

        self.col = np.empty((input.shape[0], mat_h, mat_w))
        cur = 0
        workers = []
        for x in range(height_out):
            for y in range(width_out):
                bias_x = x * self.stride
                bias_y = y * self.stride
                self.col[:, cur, :] = self.input_pad[:, :, bias_x: bias_x + self.kernel_size, bias_y: bias_y + self.kernel_size].reshape(input.shape[0], -1)
                cur = cur + 1
                # workers.append(threading.Thread(target = self.job, args = (x, y, width_out, input.shape[0])))
                # workers[-1].start()

        # for worker in workers:
        #     worker.join()
        # print(col.shape, self.weight.reshape(-1, self.weight.shape[-1]).shape)
        output = np.matmul(self.col, self.weight.reshape(-1, self.weight.shape[-1])) + self.bias
        # print(output.shape)
        self.output = np.moveaxis(output.reshape(input.shape[0], height_out, width_out, self.channel_out), 3, 1)
        return self.output


    # def backward(self,top_diff):
    #     self.d_weight = np.zeros(self.weight.shape)
    #     self.d_bias = np.zeros(self.bias.shape)
    #     botton_diff = np.zeros(self.input_pad.shape)
    #     for idxn in range(top_diff.shape[0]):
    #         for idxc in range(top_diff.shape[1]):
    #             for idxh in range(top_diff.shape[2]):
    #                 for idxw in range(top_diff.shape[3]):
    #                     self.d_weight[:,:,:,idxc] += (top_diff[idxn,idxc,idxh,idxw]*self.input_pad[idxn,:,idxh*self.stride:idxh*self.stride+self.kernel_size,idxw*self.stride:idxw*self.stride+self.kernel_size])
    #                     self.d_bias[idxc] += top_diff[idxn,idxc,idxh,idxw]
    #                     botton_diff[idxn, :, idxh * self.stride:idxh * self.stride + self.kernel_size,idxw * self.stride:idxw * self.stride + self.kernel_size] +=top_diff[idxn, idxc, idxh, idxw] * self.weight[:, :, :, idxc]
    #     botton_diff = botton_diff[:,:,self.padding:self.padding+self.input.shape[2],self.padding:self.padding+self.input.shape[3]]
    #     return botton_diff

    def backward(self, top_diff):
        # TODO: 改进backward函数，使得计算加速
        # top_diff batch, cout, h, w
        start_time = time.time()

        height_out = (self.input.shape[2] + 2 * self.padding - self.kernel_size) // self.stride + 1
        width_out = (self.input.shape[3] + 2 * self.padding - self.kernel_size) // self.stride + 1

        # cout, batch, h, w
        top_diff_col = np.transpose(top_diff, [1, 0, 2, 3]).reshape(top_diff.shape[1], -1)
        # self.col batch, (h * w), (cin * k * k)

        # what we want, cin, k, k, cout
        tmp = np.transpose(self.col.reshape(-1, self.col.shape[-1]), [1, 0])
        self.d_weight = np.matmul(tmp, top_diff_col.T).reshape(self.channel_in, self.kernel_size, self.kernel_size,
                                                               self.channel_out)
        self.d_bias = top_diff_col.sum(axis=1)

        backward_col = np.empty((top_diff.shape[0], self.input.shape[2] * self.input.shape[3],
                                 self.kernel_size * self.kernel_size * self.channel_out))
        pad_height = ((self.input.shape[2] - 1) * self.stride + self.kernel_size - height_out) // 2
        pad_width = ((self.input.shape[3] - 1) * self.stride + self.kernel_size - width_out) // 2
        top_diff_pad = np.zeros(
            (top_diff.shape[0], top_diff.shape[1], height_out + 2 * pad_height, width_out + 2 * pad_width))
        top_diff_pad[:, :, pad_height: height_out + pad_height, pad_width: width_out + pad_width] = top_diff
        cur = 0
        for x in range(self.input.shape[2]):
            for y in range(self.input.shape[3]):
                bias_x = x * self.stride
                bias_y = y * self.stride
                backward_col[:, cur, :] = top_diff_pad[:, :, bias_x: bias_x + self.kernel_size,
                                          bias_y: bias_y + self.kernel_size].reshape(top_diff.shape[0], -1)
                cur = cur + 1

        # backward_col [batch, height * width, cout * k * k]
        # try to draw a draft and you will know the reason.
        # you shall consider the contribution from top_diff to the original dx
        # if x * kernel[i] has contribution to y, then dy * kernel[size - i] will have contribution
        weight_tmp = np.transpose(self.weight, [3, 1, 2, 0]).reshape(self.channel_out, -1, self.channel_in)[:, ::-1,
                     :].reshape(-1, self.channel_in)
        bottom_diff = np.matmul(backward_col, weight_tmp)
        # [batch, height, width, cin]
        bottom_diff = np.transpose(
            bottom_diff.reshape(top_diff.shape[0], self.input.shape[2], self.input.shape[3], self.input.shape[1]),
            [0, 3, 1, 2])

        self.backward_time = time.time() - start_time
        return bottom_diff


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

    # def forward(self,input):
    #     self.input = input # N C H W
    #     height_out = (input.shape[2] - self.kernel_size) // self.stride + 1
    #     width_out = (input.shape[3] - self.kernel_size) // self.stride + 1
    #     self.output = np.zeros([input.shape[0],input.shape[1],height_out,width_out])
    #     for idxn  in range(input.shape[0]):
    #         for idxc in range(input.shape[1]):
    #             for idxh in range(height_out):
    #                 for idxw in range(width_out):
    #                     self.output[idxn,idxc,idxh,idxw] = self.input[idxn,idxc,idxh * self.stride : idxh * self.stride + self.kernel_size, idxw * self.stride: idxw * self.stride + self.kernel_size].max()
    #
    #     return self.output

    # def backward(self,top_diff):
    #     bottom_diff = np.zeros(self.input.shape)
    #     for idxn in range(top_diff.shape[0]):
    #         for idxc in range(top_diff.shape[1]):
    #             for idxh in range(top_diff.shape[2]):
    #                 for idxw in range(top_diff.shape[3]):
    #                     bias_x = idxh * self.stride
    #                     bias_y = idxw * self.stride
    #                     max_index = np.unravel_index(np.argmax(self.input[idxn, idxc, bias_x: bias_x + self.kernel_size, bias_y: bias_y + self.kernel_size]), [self.kernel_size, self.kernel_size])
    #                     bottom_diff[idxn,idxc,idxh*self.stride+max_index[0],idxw*self.stride+max_index[1]] = top_diff[idxn,idxc,idxh,idxw]
    #     return bottom_diff

    def forward(self, input):
        # TODO: 改进forward函数，使得计算加速
        self.input = input # [N, C, H, W]
        height_out = (self.input.shape[2] - self.kernel_size) // self.stride + 1
        width_out = (self.input.shape[3] - self.kernel_size) // self.stride + 1
        mat_w = self.kernel_size * self.kernel_size
        mat_h = height_out * width_out

        col = np.empty((input.shape[0], self.input.shape[1], mat_h, mat_w))
        cur = 0
        for x in range(height_out):
            for y in range(width_out):
                bias_x = x * self.stride
                bias_y = y * self.stride
                col[:, :, cur, :] = self.input[:, :, bias_x: bias_x + self.kernel_size, bias_y: bias_y + self.kernel_size].reshape(input.shape[0], input.shape[1], -1)
                cur = cur + 1

        self.output = np.max(col, axis=3, keepdims=True)
        max_index = np.argmax(col.reshape(input.shape[0], input.shape[1], height_out, width_out, self.kernel_size * self.kernel_size), axis=4)
        self.max_elements = np.zeros((input.shape[0], self.input.shape[1], height_out, width_out, self.kernel_size * self.kernel_size))
        # https://stackoverflow.com/questions/44143438/numpy-indexing-set-1-to-max-value-and-zeros-to-all-others
        # refering the advanced indexing in numpy
        n, c, h, w = self.max_elements.shape[: 4]
        N, C, H, W = np.ogrid[:n, :c, :h, :w]
        self.max_elements[N, C, H, W, max_index] = 1
        self.output = self.output.reshape(input.shape[0], input.shape[1], height_out, width_out)
        return self.output


    def backward(self, top_diff):
        # TODO: 改进backward函数，使得计算加速
        bottom_diff = np.zeros(self.input.shape)
        contrib = self.max_elements * (top_diff.reshape(list(top_diff.shape) + [1]))
        for x in range(top_diff.shape[2]):
            for y in range(top_diff.shape[3]):
                bias_x = x * self.stride
                bias_y = y * self.stride
                bottom_diff[:, :, bias_x: bias_x + self.kernel_size, bias_y: bias_y + self.kernel_size] += contrib[:, :, x, y, :].reshape(top_diff.shape[0], top_diff.shape[1], self.kernel_size, self.kernel_size)
        return bottom_diff








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
