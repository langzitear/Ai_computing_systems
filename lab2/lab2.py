import numpy as np
from PIL import Image

from lab3.layer.layer1 import FullyConnectedLayer , ReLULayer ,SoftmaxLossLayer
import struct

MNIST_DIR = "./mnist_data/"
TRAIN_DATA = "train-images-idx3-ubyte"
TRAIN_LABEL = "train-labels-idx1-ubyte"
TEST_DATA = "t10k-images-idx3-ubyte"
TEST_LABEL = "t10k-labels-idx1-ubyte"


class Nlp(object):
    def __init__(self,input_size = 784,hidden1=400,hidden2=200,out_class=10,lr=0.01,max_epoch = 20,batch_size = 100):
        self.lr = lr
        self.hidden1 = hidden1
        self.hidden2 = hidden2
        self.out_class = out_class
        self.max_epoch = max_epoch
        self.batch_size = batch_size
        self.lowest_loss = float("inf")
        self.input_size = input_size

    def load_mnist(self,input_file,is_image = False):
        bin_file = open(input_file,'rb')
        bin_data = bin_file.read()
        bin_file.close()

        if is_image:
            fmt_head = '>iiii'
            magic,num_images,num_rows,num_cols = struct.unpack_from(fmt_head,bin_data,offset=0)
        else:
            fmt_head = '>ii'
            magic,num_images = struct.unpack_from(fmt_head,bin_data,offset=0)
            num_rows,num_cols = 1,1

        self.num_images = num_images
        self.num_rows = num_rows
        self.num_cols = num_cols
        data_size = num_images * num_cols * num_rows
        mat_data = struct.unpack_from(">"+str(data_size)+"B", bin_data , offset=struct.calcsize(fmt_head))
        mat_data = np.reshape(mat_data,[num_images,num_rows*num_cols])

        print("mnist file {} parse done!!!".format(input_file))
        return  mat_data

    def get_image(self,image_data):
        image_data = np.reshape(image_data,[image_data.shape[0],self.num_rows,self.num_cols])
        # image_data_40 = image_data[:100,:,:]
        image_data_40 = image_data
        # 确保数据类型是uint8
        image_data_40 = image_data_40.astype(np.uint8)

        num_images0, num_rows, num_cols = image_data_40.shape
        print(num_images0)
        print(num_rows)
        print(num_cols)

        # 保存每张图片
        for i in range(num_images0):
            # 取出单张图片数据
            single_image_data = image_data_40[i]

            # 转换为Pillow Image对象
            img = Image.fromarray(single_image_data)

            # 保存图片
            img.save(f'image_{i}.png')


    def load_data(self):
        train_images = self.load_mnist(MNIST_DIR+TRAIN_DATA,is_image=True)
        train_labels = self.load_mnist(MNIST_DIR+TRAIN_LABEL,is_image=False)
        test_images = self.load_mnist(MNIST_DIR+TEST_DATA,is_image=True)
        test_labels = self.load_mnist(MNIST_DIR+TEST_LABEL,is_image=False)

        self.train_data = np.append(train_images,train_labels,axis=1)
        self.test_data = np.append(test_images,test_labels,axis=1)

        # print("train_data is : ",self.train_data)
        # print("test_data is : ",self.test_data)


    def shuffle_data(self):
        np.random.shuffle(self.train_data)



    def build_net(self):
        self.fc1 = FullyConnectedLayer(self.input_size,self.hidden1)
        self.ru1 = ReLULayer()
        self.fc2 = FullyConnectedLayer(self.hidden1,self.hidden2)
        self.ru2 = ReLULayer()
        self.fc3 = FullyConnectedLayer(self.hidden2,self.out_class)
        self.soft1 = SoftmaxLossLayer()
        self.init_list = [self.fc1,self.fc2,self.fc3]

    def init_net(self):
        for fc in self.init_list:
            fc.init_param()

    def forward(self,input):
        self.fc1_out = self.fc1.forward(input)
        self.ru1_out = self.ru1.forward(self.fc1_out)
        self.fc2_out = self.fc2.forward(self.ru1_out)
        self.ru2_out = self.ru2.forward(self.fc2_out)
        self.fc3_out = self.fc3.forward(self.ru2_out)
        self.soft1_out = self.soft1.forward(self.fc3_out)
        return self.soft1_out

    def backward(self):
        self.soft1_bw_out = self.soft1.backward()#[batch_size,out_classes]
        self.fc3_bw_out = self.fc3.backward(self.soft1_bw_out)
        self.ru2_bw_out = self.ru2.backward(self.fc3_bw_out)
        self.fc2_bw_out = self.fc2.backward(self.ru2_bw_out)
        self.ru1_bw_out = self.ru1.backward(self.fc2_bw_out)
        self.fc1_bw_out = self.fc1.backward(self.ru1_bw_out)
        return self.fc1_bw_out


    def update_param(self,lr):
        for fc in self.init_list:
            fc.update_param(lr)


    def save_model(self, param_dir):
        # print('Saving parameters to file ' + param_dir)
        params = {}
        params['w1'], params['b1'] = self.fc1.save_param()
        params['w2'], params['b2'] = self.fc2.save_param()
        params['w3'], params['b3'] = self.fc3.save_param()
        # print( params)
        np.save(param_dir, params)

    def train(self):
        batch_time = self.train_data.shape[0]//self.batch_size
        for i in range(self.max_epoch):
            self.shuffle_data()
            for j in range(batch_time):
                batch_images = self.train_data[j*self.batch_size:(j+1)*self.batch_size,:-1]
                batch_label = self.train_data[j*self.batch_size:(j+1)*self.batch_size,-1]
                prob = self.forward(batch_images)
                loss = self.soft1.get_loss(batch_label)
                self.backward()
                self.update_param(self.lr)
                if  j % 100 ==0:
                    print("opoch {},batch {} is done,loss is {:.6f}".format(i,j*100,loss))

                if (loss < self.lowest_loss):
                    self.lowest_loss = loss
                    # print('find lowest loss, saving model')
                    self.save_model('./npy/mlp-%d-%d-%depoch.npy' % (self.hidden1, self.hidden2, self.max_epoch))

    def load_model(self, param_dir):
        print('Loading parameters from file ' + param_dir)
        params=np.load(param_dir,allow_pickle=True).item()
        #####weight参数
        self.fc1.load_param(params['w1'],params['b1'])
        self.fc2.load_param(params['w2'],params['b2'])
        self.fc3.load_param(params['w3'],params['b3'])



    def save_net(self):
        pass

    def evaluate(self):
        pred_results = np.zeros([self.test_data.shape[0]])
        for idx in range(int(self.test_data.shape[0]/self.batch_size)):
            batch_images=self.test_data[idx*self.batch_size:(idx+1)*self.batch_size, :-1]
            prob = self.forward(batch_images)
            pred_labels=np.argmax(prob,axis=1)
            pred_results[idx*self.batch_size:(idx+1)*self.batch_size]=pred_labels
        accuracy = np.mean(pred_results==self.test_data[:,-1])
        print('Accuracy in test  set:%f' % accuracy)



        # # 获取分类错误的索引
        # incorrect_indices = np.where(pred_results != self.test_data[:, -1])[0]
        #
        # # 提取分类错误的图像、真值标签和预测标签
        # incorrect_images = self.test_data[incorrect_indices, :-1]
        # true_labels = self.test_data[incorrect_indices, -1].astype(int)
        # predicted_labels = pred_results[incorrect_indices]
        #
        # # 保存分类错误的图像
        # for i, (image, true_label, predicted_label) in enumerate(zip(incorrect_images, true_labels, predicted_labels)):
        #     # 将图像重塑为原始尺寸
        #     reshaped_image = image.reshape(28, 28)
        #
        #     # 转换为PIL Image对象
        #     img = Image.fromarray(reshaped_image.astype(np.uint8))
        #
        #     # 保存图片，使用真实标签和预测标签作为文件名的一部分
        #     img.save(f'incorrect_image_true_{true_label}_predicted_{predicted_label}_{i}.png')

if __name__ == "__main__":
    nlp = Nlp()
    nlp.load_data() #数据加载和预处理
    nlp.build_net() #构建网络
    nlp.init_net()  # 初始化网络
    nlp.train()  #开始训练
    nlp.evaluate() #评估网络性能

    # nlp = Nlp()
    # nlp.load_data() #数据加载和预处理
    # nlp.build_net() #构建网络
    # nlp.init_net()  # 初始化网络
    # nlp.load_model("mlp-100-100-20epoch.npy")
    # nlp.evaluate() #评估网络性能
