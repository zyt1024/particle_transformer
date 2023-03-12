

def myConv(x):
    #卷积核是1*1,stride是1
    print(0)
    # x.shape [512, 14, 128, 16],设置的卷积参数为Cin=14,Cout=64,kernel=1*1,stride=1
    #经历过卷积之后[512, 64, 128, 16],进行了扩维
    #权重是64,14,1,1,  64个 14通道的卷积核


import torch
aa = torch.tensor([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16])
aa = aa.reshape(8,2,1,1)
print(aa[0][0][0][0])
print(aa[0][1][0][0])
print(aa[1][0][0][0])
print(aa[1][1][0][0])
