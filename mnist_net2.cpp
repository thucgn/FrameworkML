/*************************************************************************
	> File Name: mnist_net.cpp
	> Author: cgn
	> Mail: 
	> Created Time: Sat Dec  1 12:16:01 2018
 ************************************************************************/

#include "read_mnist.h"
#include "loss.h"
#include "layers.h"
#include "initializer.h"
#include "activation.h"
#include<iostream>
using namespace std;

int main()
{
    vector<double> train_labels, test_labels;
    vector< vector<double> > train_images, test_images;

    LOGINFO("begin read data");
    read_Mnist_Label("./MNIST_data/train-labels-idx1-ubyte", train_labels);
    read_Mnist_Label("./MNIST_data/t10k-labels-idx1-ubyte", test_labels);
    read_Mnist_Images("./MNIST_data/train-images-idx3-ubyte", train_images);
    read_Mnist_Images("./MNIST_data/t10k-images-idx3-ubyte", test_images);



    LOGINFO("begin check data");
    CHECK(train_images.size() == 60000 && train_images[0].size() == 784);
    CHECK(train_labels.size() == 60000);
    CHECK(test_images.size() == 10000 && test_images[0].size() == 784);
    CHECK(test_labels.size() == 10000);

    int batch = 600;
    int width = 784;

    Tensor<double> tx(2, 10000, width);
    for(int i = 0;i < 10000;i ++)
        for(int j = 0;j < width; j ++)
            tx(i, j) = test_images[i][j]/255.0;
    Tensor<double> ty(2, 10000, 1);
    for(int i = 0;i < 10000;i ++)
        ty(i,test_labels[i]) = 1;

    Tensor<double> x(2, batch, width);
    for(int i = 0;i < batch;i ++)
        for(int j = 0;j < width; j ++)
            x(i, j) = train_images[i][j]/255.0;
    
    Tensor<double> y(2, batch, 10); 
    for(int i = 0;i < batch;i ++)
        y(i,train_labels[i]) = 1;

    Tensor<double> outY(2, batch, 10);
    Tensor<double> toutY(2, 10000, 10);
    Tensor<double> outXCopy(2, batch, 10);

    Tensor<double> loss(2, batch, 1); 
    Tensor<double> layer1loss(2, batch, 10);
    Tensor<double> saliency(2, batch, width);

    Initializer<double> initializer;
    Relu<double> relu_act;
    Sigmoid<double> sigmoid_act;
    Softmax<double> softmax_act(&outXCopy);

    FC<double> fc1(width, 10, &initializer, &softmax_act);
    SoftmaxLoss<double> softmax_loss;

    x.print();

    for(int i = 0;i < 1000;i ++)
    {
        //forward
        fc1.forward(&x, &outY);
        softmax_loss.forward(softmax_act.bottom_copy, &y, &loss);
        loss.print();

        double sum = 0.0;
        for(int i = 0; i < loss.size; i ++)
            sum += loss.data[i];
        printf("iter: %d, sum: %f, mean loss: %f\n", i, sum,  (sum/loss.size));
        //fflush(stdout);
        
        //backward
        softmax_loss.backward(&outY, &y, &layer1loss);
        fc1.backward(&layer1loss, &outY, &x, &saliency);

    }

    LOGINFO("predict train:");
    fc1.forward(&x, &outY);

    for(int i = 0;i < 10; i ++)
    {
        for(int j = 0;j < outY.axes(1); j ++)
            std::cout << outY(i,j) << ' ';
        std::cout << std::endl;
    }
    std::cout << "==========" << std::endl;
    for(int i = 0;i < 10; i ++)
    {
        for(int j = 0;j < outY.axes(1); j ++)
            std::cout << y(i,j) << ' ';
        std::cout << std::endl;
    }

    int correct = 0;
    double max_value;
    int max_id;
    for(int i = 0;i < outY.axes(0); i ++)
    {
        max_id = 0;
        max_value = 0.0;
        for(int j = 0;j < outY.axes(1); j ++)    
            if(max_value < outY(i,j))
            {
                max_value = outY(i,j);
                max_id = j;
            }
        if(y(i, max_id) == 1)
            correct ++;
    }
    LOG("acc : %f", correct*1.0/outY.axes(0));

    LOGINFO("predict test:");
    fc1.forward(&tx, &toutY);
    correct = 0;
    for(int i = 0;i < toutY.axes(0); i ++)
    {
        max_value = 0.0;
        max_id = 0;
        for(int j = 0;j < toutY.axes(1); j ++)    
            if(max_value < toutY(i,j))
            {
                max_value = toutY(i,j);
                max_id = j;
            }
        if(ty(max_id) == 1)
            correct ++;
    }
    LOG("acc : %f", correct*1.0/outY.size);


    return 0;
}


