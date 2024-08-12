import numpy as np


class ContentLossLayer(object):
    def __init__(self):
     print('\tContent loss layer.')
    def forward(self,input_layer,content_layer):
     loss = np.square(input_layer - content_layer).sum() / (2 * input_layer.shape[0] * input_layer.shape[1] * input_layer.shape[2] * input_layer.shape[3])
     return loss

    def backward(self, input_layer, content_layer):
        # TODO： 计算内容损失的反向传播
        assert input_layer.shape == content_layer.shape
        bottom_diff = (input_layer - content_layer) / input_layer.size
        return bottom_diff


class StyleLossLayer(object):
    def __init__(self):
        print("style loss layer")

    def forward(self,input_layer,style_layer):
        style_layer_reshape = np.reshape(style_layer,[style_layer.shape[0],style_layer.shape[1],-1])
        # [N, C, H, W]
        self.gram_style = style_layer_reshape @ np.transpose(style_layer_reshape, [0, 2, 1])

        self.input_layer_reshape = np.reshape(input_layer,[input_layer.shape[0],input_layer.shape[1],-1])
        self.gram_input = np.zeros([input_layer.shape[0],input_layer.shape[1],input_layer.shape[1]])
        # for idxn in range(input_layer.shape[0]):
        #     self.gram_input[idxn,:,:] = self.input_layer_reshape[idxn,:,:] @ np.transpose(self.input_layer_reshape, [0, 2, 1])
        self.gram_input = np.matmul(self.input_layer_reshape, np.transpose(self.input_layer_reshape, [0, 2, 1]))

        style_diff = np.square(self.gram_style - self.gram_input).sum() / (4 * input_layer.shape[0] * np.square(input_layer.shape[1] * input_layer.shape[2] * input_layer.shape[3]))
        loss = style_diff.sum()

        return loss


    def backward(self, input_layer, style_layer):
        self.div = (4 * input_layer.shape[0] * np.square(input_layer.shape[1] * input_layer.shape[2] * input_layer.shape[3]))
        # bottom_diff = np.zeros([input_layer.shape[0], input_layer.shape[1], input_layer.shape[2] * input_layer.shape[3]])
        # for idxn in range(input_layer.shape[0]):
        #     # TODO： 计算风格损失的反向传播
        #     bottom_diff[idxn, :, :] = np.matmul((self.gram_input[idxn, :, :] - self.gram_style[idxn, :, :]).T,
        #                                         self.input_layer_reshape[idxn, :, :]) / self.div / \
        #                               style_layer.shape[0]
        # bottom_diff = np.reshape(bottom_diff, input_layer.shape)
        # return bottom_diff


        self.div = (4 * input_layer.shape[0] * np.square(input_layer.shape[1] * input_layer.shape[2] * input_layer.shape[3]))
        return (np.matmul(np.transpose(self.gram_input - self.gram_style, [0, 2, 1]), self.input_layer_reshape) / (
                    input_layer.shape[0] * self.div)).reshape(input_layer.shape)








