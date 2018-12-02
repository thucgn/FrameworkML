/*************************************************************************
	> File Name: loss.h
	> Author: 
	> Mail: 
	> Created Time: Fri Nov 30 15:31:13 2018
 ************************************************************************/

#ifndef _LOSS_H
#define _LOSS_H
#include "tensor.h"
#include "util.h"

template <typename T>
class Loss
{
public:
    virtual void forward(Tensor<T>* y, Tensor<T>* t,  Tensor<T>* loss) = 0;
    virtual void backward(Tensor<T>* y, Tensor<T>* t, Tensor<T>* grad_y) = 0;

};

template <typename T>
class SquareLoss : public Loss<T>
{
public:
    void forward(Tensor<T>* y, Tensor<T>* t,  Tensor<T>* loss)
    {
        CHECK(y->size == t->size && y->size == loss->size);
        for(int i = 0;i < loss->size; i ++)
            loss->data[i] = 
                0.5 * (y->data[i]-t->data[i]) * (y->data[i]-t->data[i]);
    }
    void backward(Tensor<T>* y, Tensor<T>* t, Tensor<T>* grad_y)
    {
        CHECK(y->size == t->size && y->size == grad_y->size);
        for(int i = 0;i < grad_y->size; i ++)
            grad_y->data[i] = y->data[i] - t->data[i];
        
    }
};

template <typename T>
class SoftmaxLoss : public Loss<T>
{
public:
    /*void forward(Tensor<T>* y, Tensor<T>* label, Tensor<T>* loss)
    {
        CHECK(y->size == label->size && y->axes(0) == loss->size);
        for(int i = 0;i < loss->size; i ++)
            for(int j = 0;j < label->axes(1); j ++)
                if((*label)(i, j) == 1)
                    loss->data[i] = 0 - log((*y)(i, j));
    }*/
    /*
     * \bref note !!! input is z
     */
    void forward(Tensor<T>* z, Tensor<T>* label, Tensor<T>* loss)
    {
        loss->set_zeros();
        CHECK(z->size == label->size && z->axes(0) == loss->size);
        double max_num = 0.0;
        double sum = 0.0;
        for(int i = 0; i < z->axes(0); i ++)
        {
            sum = max_num = 0.0;    
            for(int j = 0;j < z->axes(1); j ++)
                if((*z)(i,j) > max_num)
                    max_num = (*z)(i,j);
            for(int j = 0;j < z->axes(1); j ++)
                sum += exp((*z)(i,j) - max_num);
            for(int j = 0;j < z->axes(1); j ++)
                if((*label)(i,j) == 1)
                    (*loss)(i,0) = max_num - (*z)(i,j) + log(sum);
        }
    }
    /**
     * \bref output is gradient of z
     */
    void backward(Tensor<T>* y, Tensor<T>* label, Tensor<T>* grad_z)
    {
        CHECK(y->size == label->size && label->size == grad_z->size);
        for(int i = 0;i < grad_z->axes(0); i ++)
            for(int j = 0;j < grad_z->axes(1); j ++)
                (*grad_z)(i, j) = (*y)(i, j) - (*label)(i,j);
    }
};



#endif
