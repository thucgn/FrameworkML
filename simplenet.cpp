/*************************************************************************
	> File Name: simplenet.cpp
	> Author: cgn
	> Mail: 
	> Created Time: Fri Nov 30 15:44:22 2018
 ************************************************************************/

#include "loss.h"
#include "layers.h"
#include "initializer.h"
#include "activation.h"
#include<iostream>
using namespace std;

int main()
{
    Tensor<double> x(2, 4, 2);
    for(int i = 0;i < 4; i ++)
        for(int j = 0;j < 2; j ++)
            x(i, j) = (i <= 1) ? (i*j) : ((i-j>1) ? 1 : 0);

    Tensor<double> y(2, 4, 1);
    y(3,0) = 1;


    Tensor<double> layer1out(2, 4, 4);
    Tensor<double> layer2out(2, 4, 1);
    Tensor<double> loss(2, 4, 1);

    Tensor<double> layer2loss(2, 4, 1);
    Tensor<double> layer1loss(2, 4, 4);
    Tensor<double> saliency(2, 4, 2);

    Initializer<double> initializer;
    Sigmoid<double> act;
    FC<double> fc1(2, 4, &initializer, &act);
    FC<double> fc2(4, 1, &initializer, &act);
    SquareLoss<double> squareloss;

    for(int i = 0;i < 10000; i ++)
    {
        // forward
        LOG("iter %d", i);
        fc1.forward(&x, &layer1out);
        //LOGINFO("layer1out");
        //layer1out.print();
        fc2.forward(&layer1out, &layer2out);
        //LOGINFO("layer2out");
        //layer2out.print();
        squareloss.forward(&layer2out, &y, &loss);
        //LOGINFO("loss");
        //loss.print();


        // backward
        squareloss.backward(&layer2out, &y, &layer2loss); 
        //LOGINFO("layer2loss");
        //layer2loss.print();
        fc2.backward(&layer2loss, &layer2out, &layer1out, &layer1loss);
        //LOGINFO("layer1loss");
        //layer1loss.print();
        fc1.backward(&layer1loss, &layer1out, &x, &saliency);
        //LOGINFO("saliency");
        //saliency.print();
    }

    fc1.forward(&x, &layer1out);
    fc2.forward(&layer1out, &layer2out);
    
    cout << "x:" << endl;
    for(int i = 0;i < 4; i ++)
        for(int j = 0;j < 2; j ++)
            cout << x(i, j) << ' ';
    cout << endl;
    cout << "y:" << endl;
    for(int i = 0;i < 4;i ++)
        for(int j = 0;j < 1; j ++)
            cout << y(i,j) << ' ';
    cout << endl;
    cout << "y':" << endl;
    for(int i = 0;i < layer2out.axes(0);i ++)
        for(int j = 0;j < layer2out.axes(1); j ++)
            cout << layer2out(i,j) << ' ';
    cout << endl;

    return 0;
}



