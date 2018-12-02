/*************************************************************************
	> File Name: initializer.h
	> Author: 
	> Mail: 
	> Created Time: Thu Nov 29 20:50:31 2018
 ************************************************************************/

#ifndef _INITIALIZER_H
#define _INITIALIZER_H

#include "tensor.h"
#include <string.h>
#include <stdlib.h>
#include <sys/time.h>
#include <math.h>

template <typename T>
class Initializer
{
public:
    virtual void visit(T* data, int len, size_t filter_size)
    {
        srand((unsigned)time(NULL));            
        double scale = sqrt(3.0 / double(filter_size));
        for(int i = 0;i < len; i ++)
            data[i] = rand()*scale/RAND_MAX;
    }

    virtual void visit(Tensor<T>* tensor, int filter_size)
    {
        srand((unsigned)time(NULL));            
        double scale = sqrt(3.0 / double(filter_size));
        for(int i = 0;i < tensor->size; i ++)
            tensor->data[i] = rand()*scale/RAND_MAX;
    }

    virtual void visit(Tensor<T>* tensor)
    {
        memset(tensor->data, 0, sizeof(T) * tensor->size);
    }
};

#endif
