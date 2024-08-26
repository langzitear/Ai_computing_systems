import torch
import torch.nn as nn
import struct
import numpy as np
from torch.utils.data import Dataset, DataLoader

MNIST_DIR = "./mnist_data/"
TRAIN_DATA = "train-images-idx3-ubyte"
TRAIN_LABEL = "train-labels-idx1-ubyte"
TEST_DATA = "t10k-images-idx3-ubyte"
TEST_LABEL = "t10k-labels-idx1-ubyte"

class COCODataSet(Dataset):
    def __init__(self,train=True):
        super().__init__()
        self.train = train
        if self.train:
            self.train_images = self.load_mnist(MNIST_DIR + TRAIN_DATA, is_image=True)
            self.train_labels = self.load_mnist(MNIST_DIR + TRAIN_LABEL, is_image=False)
        else:
            self.test_images = self.load_mnist(MNIST_DIR + TEST_DATA, is_image=True)
            self.test_labels = self.load_mnist(MNIST_DIR + TEST_LABEL, is_image=False)

    def __len__(self):
        if self.train:
            return len(self.train_images)
        else:
            return len(self.test_images)

    def __getitem__(self, item):
        if self.train:
            return self.train_images[item],self.train_labels[item]
        else:
            return self.test_images[item],self.test_labels[item]


    def load_mnist(self, input_file, is_image=False):
        bin_file = open(input_file, 'rb')
        bin_data = bin_file.read()
        bin_file.close()

        if is_image:
            fmt_head = '>iiii'
            magic, num_images, num_rows, num_cols = struct.unpack_from(fmt_head, bin_data, offset=0)
        else:
            fmt_head = '>ii'
            magic, num_images = struct.unpack_from(fmt_head, bin_data, offset=0)
            num_rows, num_cols = 1, 1

        self.num_images = num_images
        self.num_rows = num_rows
        self.num_cols = num_cols
        data_size = num_images * num_cols * num_rows
        mat_data = struct.unpack_from(">" + str(data_size) + "B", bin_data, offset=struct.calcsize(fmt_head))
        mat_data = np.reshape(mat_data, [num_images, num_rows * num_cols])

        print("mnist file {} parse done!!!".format(input_file))
        return mat_data




class MLP(nn.Module):
    def __init__(self, input_image):
        super().__init__()
        self.fc1_out = 1000
        self.fc2_out = 800

        self.layers = nn.Sequential(
            nn.Linear(input_image, self.fc1_out),
            nn.ReLU(),
            nn.Linear(self.fc1_out, self.fc2_out),
            nn.ReLU(),
            nn.Linear(self.fc2_out, 10)  # 不再使用Softmax
        )

    def forward(self, x):
        return self.layers(x)  # 返回logits


def test_model(model, test_loader, device):
    model.eval()  # 设置模型为评估模式
    correct = 0
    total = 0

    with torch.no_grad():  # 不计算梯度,提高推理效率
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            images = images.view(images.size(0), -1).to(torch.float)  # 展平图像并转换为浮点型
            if labels.ndim == 2:
                labels = labels.squeeze(1)  # 从标签中删除单一维度

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'Accuracy on test set: {accuracy:.2f}%')

    model.train()  # 将模型切换回训练模式



if __name__ == '__main__':

    data_set = COCODataSet()
    data_load = DataLoader(data_set, batch_size=100, shuffle=True)

    test_set = COCODataSet(train=False)
    test_loader = DataLoader(data_set, batch_size=100, shuffle=False)

    for images, labels in data_load:
        print(images.shape)
        input_image_size = images.view(images.size(0), -1).size(1)  # 展平图像并获取特征数量
        # input_image_size = input_image_size.item()  # 将张量转换为标量
        break  # 只需获取一次特征数量

    mlp_net = MLP(input_image_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mlp_net.to(device)

    optimizer = torch.optim.Adam(mlp_net.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    num_epochs = 40  # 设定训练的epoch数量

    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, (image, label) in enumerate(data_load):
            image, label = image.to(device), label.to(device)  # 移动数据到GPU
            image = image.view(image.size(0), -1).to(torch.float)  # 将图像展平并转换为浮点型
            # print(labels.shape)
            # 确保标签是一维张量
            if label.ndim == 2:
                label = label.squeeze(1)  # 从标签中删除单一维度

            out = mlp_net(image)
            loss = criterion(out, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        average_loss = running_loss / len(data_load)
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {average_loss:.6f}')

        # 每个epoch结束时保存模型
        torch.save(mlp_net.state_dict(), './pth/mlp_train.pth')

        test_model(mlp_net, test_loader, device)





