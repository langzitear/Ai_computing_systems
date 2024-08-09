import numpy as np

class FullyConnectedLayer(object):
    def __init__(self,input_num,out_num):
        self.input_num = input_num
        self.out_num = out_num

    def init_param(self):
        self.weight = np.random.normal(0.0, 0.01, (self.input_num, self.out_num))
        self.bias = np.zeros([1,self.out_num])
        # self.bias = np.zeros([self.out_num,1])

    def forward(self,input):#input shape [batch_size,image_size]
        self.input = input
        # self.output = self.input @ self.weight + self.bias.T
        self.output = self.input @ self.weight + self.bias
        return self.output

    def backward(self,top_diff):
        self.d_weight = self.input.T @ top_diff
        # print("top_diff shape is:",top_diff.shape)
        column_vector = np.ones((1,top_diff.shape[0]))
        self.d_bias = column_vector @  top_diff

        # column_vector = np.ones((top_diff.shape[0],1))
        # self.d_bias = top_diff.T @ column_vector

        # self.d_bias =  top_diff.T
        bottom_diff = top_diff @ self.weight.T
        return  bottom_diff

    # def load_param(self,para):
    #     self.weight = para['w']
    #     self.bias  = para['b']

    def load_param(self, weight, bias): # 参数加载
        assert self.weight.shape == weight.shape
        assert self.bias.shape == bias.shape
        self.weight=weight
        self.bias=bias

    def update_param(self,lr):
        self.weight = self.weight - lr * self.d_weight
        self.bias = self.bias - lr * self.d_bias

    def save_param(self):
        return self.weight,self.bias


class ReLULayer(object):
    def __init_(self):
        print("RulyLayer is init!!!!")

    def forward(self,input):
        self.input = input
        self.output  = np.maximum(0,self.input)
        return self.output

    def backward(self,top_diff):
        bottom_diff = top_diff * (self.input > 0.)
        return bottom_diff



class SoftmaxLossLayer(object):
    def __init_(self):
        print("softmax loss layer is init!!!!!")


    def forward(self,input):
        input_max = np.max(input, axis=1, keepdims=True) #[batch_size,1]
        input_exp = np.exp(input - input_max)#[batch_size,out_classes]
        exp_sum = np.sum(input_exp, axis=1, keepdims=True)#[batch_size,1]
        self.prob = input_exp / exp_sum #[batch_size,out_classes]
        return self.prob #[batch_size,out_classes]

    def get_loss(self,label):
        self.batch_size = self.prob.shape[0]
        self.label_onehot = np.zeros_like(self.prob)
        self.label_onehot[np.arange(self.batch_size),label] = 1.0
        loss = - np.sum(self.label_onehot * np.log(self.prob)) / self.batch_size
        return loss

    def backward(self):
        return (self.prob - self.label_onehot)/self.batch_size