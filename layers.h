/*************************************************************************
	> File Name: layers.h
	> Author: cgn
	> Mail: 
	> Created Time: Thu Nov 29 19:47:33 2018
 ************************************************************************/ 
#ifndef _LAYERS_H
#define _LAYERS_H

#include "util.h"
#include "initializer.h"
#include "tensor.h"
#include "activation.h"
#include <vector>
#include <string>
#include <string.h>

class Layer
{
public:
};

template <typename T>
class BaseLayer : public Layer
{
public:

};

template <typename T>
class Flatten : public BaseLayer< Flatten<T> >
{
public:
};

template <typename T>
class FC : public BaseLayer< FC<T> >
{
public:
    int ni, no;
    Activation<T>* act;
    double lr;
    Tensor<T>* weight;
    Tensor<T>* bias;
    Tensor<T>* grad_w;
    Tensor<T>* grad_b;

    FC(int ni, int no, Initializer<T>* initializer, Activation<T>* act, double lr=0.1): 
        ni(ni), no(no), act(act),
        lr(lr)
    {
        weight = new Tensor<T>(2, ni, no);
        initializer->visit(weight, ni);
        /*if(ni == 4)
        {
            for(int i = 0;i < weight->size; i ++)
                weight->data[i] = 2;
        }
        else
        {
            for(int i = 0;i < weight->size; i ++)
                weight->data[i] = 3;
        }*/
        grad_w = new Tensor<T>(2, ni, no);
        bias = new Tensor<T>(1, no);
        grad_b = new Tensor<T>(1, no);
    }

    ~FC() 
    {
        if(weight) 
            delete weight;
        if(bias)
            delete bias;
        if(grad_w)
            delete grad_w;
        if(grad_b)
            delete grad_b;
    }
    
    void forward(Tensor<T>* in, Tensor<T>* out)
    {
        CHECK(in->axes(1) == ni && out->axes(1) == no && in->axes(0)==out->axes(0));
        for(int ib = 0;ib < in->axes(0); ib ++)
            for(int io = 0;io < no; io ++)
            {
                T sum = 0;
                for(int ii = 0;ii < ni; ii ++)
                    sum += (*in)(ib, ii) * (*weight)(ii, io);
                sum += (*bias)(io);
                //(*out)(ib, io) = act->forward(sum);
                (*out)(ib, io) = sum;
            }
        act->forward(out, out);

    }

    
    void backward(Tensor<T>* grad_y,Tensor<T>* y, Tensor<T>* x, Tensor<T>* grad_x)
    {
        Tensor<T>* grad_z = new Tensor<T>();
        grad_z->reshape(y);

        // calculate grad_z
        if(act->type() == SIGMOID)
        {
            // sigmoid
            // grad_z = grad_y * y * (1 - y)
            act->backward(y, grad_z);
            for(int i = 0;i < grad_y->size; i ++)
                grad_z->data[i] = grad_y->data[i] * grad_z->data[i];
        }
        else if(act->type() == RELU)
        {
            // relu
            act->backward(y, grad_z);
            for(int i = 0;i < grad_y->size; i ++)
                grad_z->data[i] = grad_y->data[i] * grad_z->data[i];
        }
        else if(act->type() == SOFTMAX)
        {
            // softmax
            // note!!! grad_y is indeed grad_z due to the special softmax loss
            act->backward(grad_y, grad_z);
        }else
        {
            CHECK(0 && "unrecognized act");
        }

        // grad_w = xT * grad_z
        gemm2DTransX(x, grad_z, grad_w);

        // grad_x = grad_z * wT
        gemm2DTransY(grad_z, weight, grad_x);

        // grad_b = reduce(grad_z)
        grad_b->set_zeros();
        for(int i = 0; i < grad_z->axes(0); i ++)
        {
            for(int j = 0;j < grad_z->axes(1); j ++)
                (*grad_b)(j) += (*grad_z)(i,j);
        }


        // update w
        (*grad_w) *= lr;
        (*weight) -= (*grad_w);

        // update b
        (*grad_b) *= lr;
        (*bias) -= (*grad_b);

        delete grad_z;
    }

};

#endif
