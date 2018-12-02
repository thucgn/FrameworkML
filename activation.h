/*************************************************************************
	> File Name: activation.h
	> Author: 
	> Mail: 
	> Created Time: Fri Nov 30 11:29:36 2018
 ************************************************************************/

#ifndef _ACTIVATION_H
#define _ACTIVATION_H

#include "tensor.h"
#include "util.h"
#include <string.h>

enum ActivationType { SIGMOID, RELU, SOFTMAX, UNKNOWN};

template <typename T>
class Activation
{
public:
    virtual void forward(Tensor<T>* in, Tensor<T>* out) = 0;
    virtual void backward(Tensor<T>* in, Tensor<T>* out_grad) = 0;
    virtual T forward(T in) = 0;
    virtual T backward(T in) = 0;
    virtual ActivationType type() = 0;
};

template <typename T>
class Sigmoid : public Activation<T>
{
public:
    ActivationType type() { return SIGMOID; }
    void forward(Tensor<T>* in, Tensor<T>* out)
    {
        CHECK(in->size == out->size);
        for(int i = 0;i < in->size; i ++)
            out->data[i] = sigmoid(in->data[i]);
    }
    void backward(Tensor<T>* in, Tensor<T>* out_grad)
    {
        CHECK(in->size == out_grad->size);
        for(int i = 0;i < in->size; i ++)
            out_grad->data[i] = sigmoid_grad(in->data[i]);
    }
    T forward(T in) { return sigmoid(in); }
    T backward(T in) { return sigmoid_grad(in); }
};

template <typename T>
class Relu : public Activation<T>
{
public:
    ActivationType type() { return RELU; }
    void forward(Tensor<T>* bottom, Tensor<T>* top)
    {
        CHECK(top->size == bottom->size);
        for(int i = 0;i < bottom->size; i ++)
            top->data[i] = LMath::max(T(0), bottom->data[i]);
    }
    void backward(Tensor<T>* top, Tensor<T>* bottom_grad)
    {
        CHECK(top->size == bottom_grad->size);
        for(int i = 0;i < top->size; i ++)
            bottom_grad->data[i] = top->data[i] > 0 ? 1 : 0;
    }
    T forward(T in) { return LMath::max(T(0), in); }
    T backward(T in) { return in > 0 ? 1 : 0; }
};

template <typename T>
class Softmax : public Activation<T>
{
public:
    Tensor<T>* bottom_copy;
    Softmax(Tensor<T>* bottom_copy):
        bottom_copy(bottom_copy)
    {}

    ActivationType type() { return SOFTMAX; }
    void forward(Tensor<T>* bottom, Tensor<T>* top)
    {
        CHECK(top->size == bottom->size);
        CHECK(sizeof(bottom->data[0]) == sizeof(bottom_copy->data[0]));
        CHECK(bottom_copy->same_shape_as(*bottom));

        memcpy(bottom_copy->data, bottom->data, sizeof(T) * bottom->size);


        double max_num = 0.0;
        double sum = 0.0;

        for(int i = 0;i < bottom->axes(0); i ++)
        {
            sum = 0.0;
            max_num = 0.0;
            for(int j = 0;j < bottom->axes(1); j ++)
                if((*bottom)(i,j) > max_num)
                    max_num = (*bottom)(i, j);
            for(int j = 0;j < bottom->axes(1); j ++)
            {
                (*top)(i, j) = exp((*bottom)(i, j) - max_num);
                sum += (*top)(i, j);
            }
            for(int j = 0;j < bottom->axes(1); j ++)
                (*top)(i, j) = (*top)(i,j) / sum;
        }
    }

    void backward(Tensor<T>* bottom, Tensor<T>* top)
    {
        // just copy bottom to top; because cross_entropy has calculated the
        // gradient of dLoss/dz
        CHECK(top->size == bottom->size);
        for(int i = 0;i < top->size; i ++)
            top->data[i] = bottom->data[i];
    }
    T forward(T in) { CHECK(0 && "softmax cannot process signal element)"); }
    T backward(T in) { CHECK(0 && "softmax cannot process signal element)"); }
};


#endif
