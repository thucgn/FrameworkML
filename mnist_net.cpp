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

    int batch = 60000;
    int width = 784;

    Tensor<double> x(2, batch, width);
    for(int i = 0;i < batch;i ++)
        for(int j = 0;j < width; j ++)
            x(i, j) = train_images[i][j]/255.0;
    
    Tensor<double> y(2, batch, 10); 
    for(int i = 0;i < batch;i ++)
        y(i,train_labels[i]) = 1;

    Tensor<double> layer1out(2, batch, 128);
    Tensor<double> layer2out(2, batch, 10);
    Tensor<double> layer2out_copy(2, batch, 10);
    Tensor<double> loss(2, batch, 1);

    Tensor<double> layer2loss(2, batch, 10); 
    Tensor<double> layer1loss(2, batch, 128);
    Tensor<double> saliency(2, batch, width);

    Initializer<double> initializer;
    Relu<double> relu_act;
    Sigmoid<double> sigmoid_act;
    Softmax<double> softmax_act(&layer2out_copy);

    FC<double> fc1(width, 128, &initializer, &relu_act);
    FC<double> fc2(128, 10, &initializer, &softmax_act);
    SoftmaxLoss<double> softmax_loss;

    x.print();

    for(int i = 0;i < 5;i ++)
    {
        //forward
        //LOG("iter %d", i);
        //LOGINFO("fc1");
        fc1.forward(&x, &layer1out);
        layer1out.print();
        //LOGINFO("fc2");
        fc2.forward(&layer1out, &layer2out);
        layer2out.print();
        //LOGINFO("loss");
        softmax_loss.forward(softmax_act.bottom_copy , &y, &loss);
        loss.print();

        double sum = 0.0;
        for(int i = 0; i < loss.size; i ++)
            sum += loss.data[i];
        printf("iter: %d, sum: %f, mean loss: %f\n", i, sum,  (sum/loss.size));
        //fflush(stdout);
        
        //backward
        //LOGINFO("bac_loss");
        softmax_loss.backward(&layer2out, &y, &layer2loss);
        layer2loss.print();
        //LOGINFO("bac_fc2");
        fc2.backward(&layer2loss, &layer2out, &layer1out, &layer1loss);
        layer1loss.print();
        //LOGINFO("bac_fc1");
        fc1.backward(&layer1loss, &layer1out, &x, &saliency);

    }


    return 0;
}


