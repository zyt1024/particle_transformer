#include <iostream>

typedef int32_t DTYPE;

int main(){
    const DTYPE weight_C1[8][2][1][1]={1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16};
    for(int i = 0; i < 8;i++){
        for(int j = 0; j < 2;j++){
            for(int l = 0; l < 1;l++){
                for(int m=0;m < 1;m++){
                    std::cout << weight_C1[i][j][l][m] << "   ";
                }
            }
        }
    }

    
    return 0;
}


void myConv(int input[512][14][128][16], int weight[64][14][1][1], int out[512][64][128][16], int bias[64]){
    
       
}