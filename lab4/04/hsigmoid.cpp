#include <torch/extension.h>
#include <vector>
#include <cmath> // 包含 exp 函数
using namespace std;

float sigmoid(float x) {
    return 1.0 / (1.0 + std::exp(-x)); // 计算 Sigmoid 值
}


torch::Tensor hsigmoid_cpu(const torch::Tensor & dets){
    auto input_data = dets.accessor<float,2>();
    int input_batch_size = input_data.size(0);
    int input_size = input_data.size(1);
    vector <float> output_data(input_batch_size * input_size);
    for(int i=0;i<input_batch_size;i++){
        for(int j =0 ;j < input_size ; j++){
            float x = input_data[i][j];
            output_data[i*input_size+j] = sigmoid(x);
        }
    }

    auto opts = torch::TensorOptions().dtype(torch::kFloat32);
    auto foo = torch::from_blob(output_data.data(),{int64_t(output_data.size())}, opts).clone();

    return foo;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){
    m.doc() = "This is a hsigmoid func just like sigmoid"; // 模块文档
    m.def("hsigmoid_cpu", &hsigmoid_cpu, "A function that computes the hsigmoid_cpu of a number(cpu)"); // 绑定函数
}