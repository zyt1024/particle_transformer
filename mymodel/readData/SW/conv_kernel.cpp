#include "common.h"



// 第一次层卷积
void edgeConv_conv_1(float input[BATCH_SIZE][CONV_1_IN_C][CONV_ALL_INPUT_H][CONV_ALL_INPUT_K], //(N,14,128,16)
                     float conv_kernel[CONV_1_OUT_C][CONV_1_IN_C][KERSIZE_H][KERSIZE_W], //(64,14,1,1)
                     float output[BATCH_SIZE][CONV_1_OUT_C][CONV_ALL_INPUT_H][CONV_ALL_INPUT_K] //(N,64,128,16)
                     )
{ 
    // stride = 1;
    for(int bs = 0; bs < BATCH_SIZE; bs++){ // 遍历batch_size
        for(int outc = 0; outc < CONV_1_OUT_C; outc++){ // 遍历输出通道
            for(int inc = 0; inc < CONV_1_IN_C; inc++){ //遍历输入通道
                for(int h = 0; h < CONV_ALL_INPUT_H; h++){ // 遍历H
                    for(int k = 0; k < CONV_ALL_INPUT_K; k++){ // 遍历K
                        // 被取出来的区域,针对于stride = 1且kernel的size为(1,1)只有一个
                        float region = input[bs][outc][h][k];
                        // 点乘
                        output[bs][outc][h][w] += region * conv_kernel[outc][inc][0][0];                                 
                    }
                }
            }
        }
    }
    
}