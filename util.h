/*************************************************************************
	> File Name: util.h
	> Author: cgn
	> Mail: 
	> Created Time: Thu Nov 29 16:50:30 2018
 ************************************************************************/

#ifndef _UTIL_H
#define _UTIL_H

#include <stdio.h>
#include <stdarg.h>
#include <stdlib.h>
#include <sys/time.h>
#include <time.h>
#include <math.h>

#define LOG(format, ...) do { \
    printf("[%s, %s:%d %s] " format "\n", \
        __TIME__, __FILE__, __LINE__, __FUNCTION__, ##__VA_ARGS__); \
}while(0)

#define LOGINFO(message) do { \
    printf("[%s, %s:%d %s] %s\n", \
        __TIME__, __FILE__, __LINE__, __FUNCTION__, message); \
}while(0)

#define CHECK(flag) do { \
    if(!(flag)) { \
      printf("[%s, %s:%d %s] check failure: %s \n", \
        __TIME__, __FILE__, __LINE__, __FUNCTION__, #flag); \
        exit(0); \
    } \
} while(0)

#define MARK_TIME(t) gettimeofday(&t, NULL)
#define DIFF_TIME(tt, ts) (((tt).tv_sec-(ts).tv_sec) + ((tt).tv_usec*1e-6 - (ts).tv_usec*1e-6))

typedef struct timeval TIME_T;

typedef unsigned long ulong;
typedef unsigned int uint;

typedef double DT;

template <typename T>
inline T sigmoid(T e) {
    return 1 / (1 + exp(-e));
}
template <typename T>
inline T sigmoid_grad(T e) {
    return e * (1-e);
}

namespace LMath{

template <typename T>
inline T max(T o1, T o2)
{
    return o1 > o1 ? o1 : o2;
}


}


#endif

