#ifndef CONV_H
#define CONV_H
#include <iostream>
#define K 1  //卷积和的大小 (1,1)
#define Tr 128 // 输入H
#define Tc 16  // 输入K
#define Tout 64 // 输出通道
#define Tin 14  // 输入通道
#define S 1    // 步长
#define B 4    // batch_size大小


// 原始算法
void conv(){
    int K = 1;
    int Tr = 128;
    int Tc = 16;
    int Tout = 64;
    int Tin = 14;
    int S = 1;
    //当K=1的时候,和S=1的时候,可以简写
    Batch_size:
    for(int b=0; b < B;b++){
        Kernel_Row:
        for(int kr=0;kr<K;kr++)
        {
            Kernel_Col:
            for(int kc=0;kc<K;kc++)
            {
                Row:
                for(int r=0;r<Tr;r++)
                {
                    Column:
                    for(int c=0;c<;c++)
                    {
                        Out_channel:
                        for(int out=0;out<Tout;out++)
                        {
                            in_channel:
                            for(int in=0;in<Tin;in++)
                            {
                                Out[out][r][c]+=In[in][S*r+kr][S*c+kc]*W[out][in][kr][kc];
                            }
                        }
                    }
                }
            }
        }         
    }  
}

// 针对于K=1和S=1的卷积
void conv_1(){
    int K = 1;
    int Tr = 128;
    int Tc = 16;
    int Tout = 64;
    int Tin = 14;
    int S = 1;
    //当K=1的时候,和S=1的时候,可以简写
    Batch_size:
    for(int b=0; b < B;b++){
        Row:
        for(int r=0;r<Tr;r++)
        {
            Column:
            for(int c=0;c<Tc;c++)
            {
                Out_channel:
                for(int out=0;out<Tout;out++)
                {
                    in_channel:
                    for(int in=0;in<Tin;in++)
                    {
                        Out[b][out][r][c]+=In[b][in][r][c]*W[out][in][0][0];
                    }
                }
            }
        }
    }  
}
