/*************************************************************************
	> File Name: tensor.h
	> Author: 
	> Mail: 
	> Created Time: Thu Nov 29 23:12:33 2018
 ************************************************************************/

#ifndef _TENSOR_H
#define _TENSOR_H

#include "util.h"
#include <stdio.h>
#include <string.h>
#include <iostream>

template <typename T>
class Tensor
{
public:
    int dim;
    size_t d3, d2, d1, d0;
    size_t size;
    T* data;
    bool own;

    Tensor(int dim, int d3, int d2=1, int d1=1, int d0=1):
        dim(dim), d3(d3), d2(d2), d1(d1), d0(d0)
    {
        size = this->d3 * this->d2 * this->d1 * this->d0;
        data = new T[size];
        CHECK(data);
        memset(data, 0, sizeof(T) * size);
        own = true;
    };


    Tensor(T* od, int dim, int d3, int d2=1, int d1=1, int d0=1):
        dim(dim), d3(d3), d2(d2), d1(d1), d0(d0)
    {
        size = this->d3 * this->d2 * this->d1 * this->d0;
        data = od;
        own = false;
    };

    Tensor():
        dim(0), d3(0), d2(0), d1(0), d0(0), 
        size(0), data(NULL)
    {
        own = false;
    }

    ~Tensor() 
    { 
        if(data && own) 
            delete [] data;
    }

    T& operator()(int i0) { 
        //LOG("id %d, dim %d, size %d, i0 %d", lid, dim, size, i0);
        //CHECK(dim==1 && i0 < size); 
        return data[i0]; 
    }
    T& operator()(int i1, int i0) { 
        //LOG("%d", size);  
        //LOG("index i1 %d, d2 %d, i0 %d,  %d",i1,d2,i0, (i1*d2+i0));
        //CHECK(dim==2 && (i1*d2+i0)<size);
        return data[i1*d2 + i0]; 
    }
    T& operator()(int i2, int i1, int i0) { 
        //CHECK(dim==3 && ((i2*d2+i1)*d1+i0) < size); 
        return data[(i2*d2+i1)*d1 + i0]; 
    }
    T& operator()(int i3, int i2, int i1, int i0) { 
        //CHECK(dim==4);
        //int index = ((i3*d2+i2)*d1+i1)*d0 + i0;
        //CHECK(index < size);
        return data[((i3*d2+i2)*d1+i1)*d0 + i0]; 
    }

    int axes(int axes) 
    {
        switch(axes)
        {
            case 0:
                return d3;
            case 1:
                return d2;
            case 2:
                return d1;
            case 3:
                return d0;
            default:
                CHECK(0);
        }
        return -1;
    }

    bool same_shape_as(const Tensor& o)
    {
        bool ret = dim == o.dim && d3==o.d3 && d2 == o.d2 &&
            d1 == o.d1 && size == o.size;
        return ret;
    }

    Tensor& operator-=(const Tensor& o)
    {
        CHECK( d3==o.d3 && d2==o.d2 && d1==o.d1 && d0==o.d0);
        for(int i = 0;i < size; i ++)
            data[i] -= o.data[i];
        return *this;
    }

    Tensor& operator*=(const T e)
    {
        for(int i = 0;i < size; i ++)
            data[i] *= e;
        return *this;
    }

    void reshape(const Tensor* o)
    {
        dim = o->dim;
        d3 = o->d3;
        d2 = o->d2;
        d1 = o->d1;
        d0 = o->d0;
        size = o->size;
        if(data && own)
            delete [] data;
        data =  new T[size];
        CHECK(data);
        memset(data, 0, sizeof(T) * size);
        own = true;
    }

    void set_zeros()
    {
        if(data)
            memset(data, 0, sizeof(T) * size);
    }

    void print()
    {
        return;
        if(dim == 2)
        for(int i = 0;i < d3; i ++)
        {
            for(int j = 0;j < d2; j ++)
                std::cout << data[i*d2 + j] << ' ';
            std::cout << std::endl;
        }
        if(dim == 1)
        {
        for(int i = 0;i < size; i ++)
            std::cout << data[i] << ' ';
        std::cout << std::endl;
        }
            
    }
    
};

template <typename T>
void gemm2D(Tensor<T>* x, Tensor<T>* y, Tensor<T>* z)
{
    CHECK(x->dim == 2 && y->dim == 2);
    CHECK((x->d3==z->d3) &&(x->d2==y->d3) && (y->d2==z->d2));
    const int M = x->d3;
    const int K = x->d2;
    const int N = y->d2;

    z->set_zeros();

    for(int k = 0;k < K; k ++)
        for(int m = 0; m < M; m ++)
            for(int n = 0; n < N; n ++)
                (*z)(m, n) += (*x)(m,k) * (*y)(k, n);
}

template <typename T>
void gemm2DTransX(Tensor<T>* x, Tensor<T>* y, Tensor<T>* z)
{
    CHECK(x->dim == 2 && y->dim == 2);
    CHECK((x->axes(1)==z->axes(0)) &&(x->axes(0)==y->axes(0)) && (y->axes(1)==z->axes(1)));
    const int M = x->axes(1);
    const int K = x->axes(0);
    const int N = y->axes(1);

    z->set_zeros();

    for(int k = 0;k < K; k ++)
        for(int m = 0; m < M; m ++)
            for(int n = 0; n < N; n ++)
                (*z)(m, n) += (*x)(k,m) * (*y)(k, n);
}

template <typename T>
void gemm2DTransY(Tensor<T>* x, Tensor<T>* y, Tensor<T>* z)
{
    CHECK(x->dim == 2 && y->dim == 2);
    CHECK((x->axes(0)==z->axes(0)) &&(x->axes(1)==y->axes(1)) && (y->axes(0)==z->axes(1)));
    const int M = x->axes(0);
    const int K = x->axes(1);
    const int N = y->axes(0);

    z->set_zeros();

    for(int k = 0;k < K; k ++)
        for(int m = 0; m < M; m ++)
            for(int n = 0; n < N; n ++)
                (*z)(m, n) += (*x)(m,k) * (*y)(n, k);
}



#endif
