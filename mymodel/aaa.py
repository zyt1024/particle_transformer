import sys,os
# sys.path.append('需要作为模块引入的路径')
# 添加当前路径的前一级文件作为源文件夹
path = os.path.dirname(os.path.dirname(__file__)) 
print(path)
sys.path.append(path)

import torch

#创建一个全1的2*128的向量
x = torch.ones([2, 128])
x = x.reshape([1,2,128])

print(x.shape)

print(x[0,:,:],x[0,:,:].shape)

# xx = torch.tensor(
# [[ 1.3503e-01, -1.4550e-01, -7.7082e-02,  8.7171e-02,  5.7269e-02,
#          -1.0102e-01,  6.6573e-02,  6.5457e-02,  1.3503e-01, -3.7740e-01,
#           1.3327e-02, -3.5718e-01,  1.1996e-01, -1.1659e-01,  1.5008e-01,
#           9.4135e-02,  7.2064e-02,  8.7043e-02, -1.3932e-01,  1.0222e-01,
#           1.5559e-01, -3.1527e-01,  5.2226e-02, -3.7830e-01, -9.0162e-02,
#          -3.5481e-02, -1.4212e-02, -2.0237e-01, -3.9571e-01, -4.4215e-01,
#          -1.4550e-01, -3.4501e-01, -3.6527e-01,  3.1136e-02,  2.4295e-01,
#          -2.0210e-01, -4.7323e-01,  1.3327e-02,  2.8007e-01,  3.4718e-01,
#          -9.4233e-02,  2.4295e-01, -3.4026e-01, -3.4225e-01,  6.0961e-01,
#           1.7956e-01,  4.1946e-01,  2.3285e-01,  1.0000e+09,  1.0000e+09,
#           1.0000e+09,  1.0000e+09,  1.0000e+09,  1.0000e+09,  1.0000e+09,
#           1.0000e+09,  1.0000e+09,  1.0000e+09,  1.0000e+09,  1.0000e+09,
#           1.0000e+09,  1.0000e+09,  1.0000e+09,  1.0000e+09,  1.0000e+09,
#           1.0000e+09,  1.0000e+09,  1.0000e+09,  1.0000e+09,  1.0000e+09,
#           1.0000e+09,  1.0000e+09,  1.0000e+09,  1.0000e+09,  1.0000e+09,
#           1.0000e+09,  1.0000e+09,  1.0000e+09,  1.0000e+09,  1.0000e+09,
#           1.0000e+09,  1.0000e+09,  1.0000e+09,  1.0000e+09,  1.0000e+09,
#           1.0000e+09,  1.0000e+09,  1.0000e+09,  1.0000e+09,  1.0000e+09,
#           1.0000e+09,  1.0000e+09,  1.0000e+09,  1.0000e+09,  1.0000e+09,
#           1.0000e+09,  1.0000e+09,  1.0000e+09,  1.0000e+09,  1.0000e+09,
#           1.0000e+09,  1.0000e+09,  1.0000e+09,  1.0000e+09,  1.0000e+09,
#           1.0000e+09,  1.0000e+09,  1.0000e+09,  1.0000e+09,  1.0000e+09,
#           1.0000e+09,  1.0000e+09,  1.0000e+09,  1.0000e+09,  1.0000e+09,
#           1.0000e+09,  1.0000e+09,  1.0000e+09,  1.0000e+09,  1.0000e+09,
#           1.0000e+09,  1.0000e+09,  1.0000e+09,  1.0000e+09,  1.0000e+09,
#           1.0000e+09,  1.0000e+09,  1.0000e+09],
#         [ 3.0021e-02,  7.1463e-02,  2.7765e-02, -1.5743e-03,  1.1683e-02,
#           2.0369e-02,  7.9145e-03, -1.1382e-02,  3.0021e-02, -6.0396e-01,
#           2.0378e-01, -5.9795e-01,  4.7380e-02,  2.7191e-02,  3.7986e-01,
#           2.2476e-01, -8.7023e-05, -2.6452e-02,  7.1253e-02,  2.1152e-02,
#           2.7681e-01, -6.1757e-01,  3.9073e-02, -5.9451e-01,  3.9817e-01,
#          -1.0541e-02,  2.5514e-01,  5.7106e-02, -6.8303e-01, -4.0419e-01,
#           7.1463e-02, -2.2394e-01, -6.2869e-01,  2.6039e-01,  2.0070e-01,
#          -8.7215e-02, -5.8764e-01,  2.0378e-01,  2.9300e-01,  4.2280e-01,
#          -8.8246e-02,  2.0070e-01, -2.1816e-01, -2.9317e-01, -3.8596e-01,
#          -6.6892e-02, -2.3828e-01, -7.2653e-02,  1.0000e+09,  1.0000e+09,
#           1.0000e+09,  1.0000e+09,  1.0000e+09,  1.0000e+09,  1.0000e+09,
#           1.0000e+09,  1.0000e+09,  1.0000e+09,  1.0000e+09,  1.0000e+09,
#           1.0000e+09,  1.0000e+09,  1.0000e+09,  1.0000e+09,  1.0000e+09,
#           1.0000e+09,  1.0000e+09,  1.0000e+09,  1.0000e+09,  1.0000e+09,
#           1.0000e+09,  1.0000e+09,  1.0000e+09,  1.0000e+09,  1.0000e+09,
#           1.0000e+09,  1.0000e+09,  1.0000e+09,  1.0000e+09,  1.0000e+09,
#           1.0000e+09,  1.0000e+09,  1.0000e+09,  1.0000e+09,  1.0000e+09,
#           1.0000e+09,  1.0000e+09,  1.0000e+09,  1.0000e+09,  1.0000e+09,
#           1.0000e+09,  1.0000e+09,  1.0000e+09,  1.0000e+09,  1.0000e+09,
#           1.0000e+09,  1.0000e+09,  1.0000e+09,  1.0000e+09,  1.0000e+09,
#           1.0000e+09,  1.0000e+09,  1.0000e+09,  1.0000e+09,  1.0000e+09,
#           1.0000e+09,  1.0000e+09,  1.0000e+09,  1.0000e+09,  1.0000e+09,
#           1.0000e+09,  1.0000e+09,  1.0000e+09,  1.0000e+09,  1.0000e+09,
#           1.0000e+09,  1.0000e+09,  1.0000e+09,  1.0000e+09,  1.0000e+09,
#           1.0000e+09,  1.0000e+09,  1.0000e+09,  1.0000e+09,  1.0000e+09,
#           1.0000e+09,  1.0000e+09,  1.0000e+09]],
# )

# print(xx.shape)
import numpy as np
torch.set_printoptions(threshold=np.inf)   # 配置打印的时候不显示省略号

def knn(x, k): # x(N, 2, 128)  k=6
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
    
    idx = pairwise_distance.topk(k=k + 1, dim=-1)[1][:, :, 1:]  # (batch_size, num_points, k)
        
    return idx

# print(knn(xx.reshape([1,2,128]),16))
# print(knn(xx.reshape([1,2,128]),16).shape)


yy = torch.tensor(
    [[[ 7.0468e-02, -3.5568e-02, -1.0625e-02, -2.9171e-02, -3.3797e-02,
         -4.9668e-01, -4.6522e-01, -1.5307e-02, -5.0524e-02, -3.9443e-02,
         -2.9510e-02, -2.4383e-02, -2.5356e-02,  3.6282e-01,  3.3283e-01,
          1.7820e-01,  3.3175e-01, -4.2212e-01,  2.9059e-01,  2.4348e-01,
         -5.5367e-01,  2.5529e-01,  1.7115e-01,  3.8124e-02,  1.9054e-01,
         -3.5324e-01,  1.4207e-01, -3.9782e-01,  3.3560e-01,  5.4863e-01,
         -6.2017e-01,  3.3560e-01, -6.1146e-01,  4.4048e-01, -2.5607e-01,
          2.9296e-01,  1.5728e-01, -6.4314e-01, -2.6155e-01, -6.1391e-01,
         -2.7916e-01,  1.0000e+09,  1.0000e+09,  1.0000e+09,  1.0000e+09,
          1.0000e+09,  1.0000e+09,  1.0000e+09,  1.0000e+09,  1.0000e+09,
          1.0000e+09,  1.0000e+09,  1.0000e+09,  1.0000e+09,  1.0000e+09,
          1.0000e+09,  1.0000e+09,  1.0000e+09,  1.0000e+09,  1.0000e+09,
          1.0000e+09,  1.0000e+09,  1.0000e+09,  1.0000e+09,  1.0000e+09,
          1.0000e+09,  1.0000e+09,  1.0000e+09,  1.0000e+09,  1.0000e+09,
          1.0000e+09,  1.0000e+09,  1.0000e+09,  1.0000e+09,  1.0000e+09,
          1.0000e+09,  1.0000e+09,  1.0000e+09,  1.0000e+09,  1.0000e+09,
          1.0000e+09,  1.0000e+09,  1.0000e+09,  1.0000e+09,  1.0000e+09,
          1.0000e+09,  1.0000e+09,  1.0000e+09,  1.0000e+09,  1.0000e+09,
          1.0000e+09,  1.0000e+09,  1.0000e+09,  1.0000e+09,  1.0000e+09,
          1.0000e+09,  1.0000e+09,  1.0000e+09,  1.0000e+09,  1.0000e+09,
          1.0000e+09,  1.0000e+09,  1.0000e+09,  1.0000e+09,  1.0000e+09,
          1.0000e+09,  1.0000e+09,  1.0000e+09,  1.0000e+09,  1.0000e+09,
          1.0000e+09,  1.0000e+09,  1.0000e+09,  1.0000e+09,  1.0000e+09,
          1.0000e+09,  1.0000e+09,  1.0000e+09,  1.0000e+09,  1.0000e+09,
          1.0000e+09,  1.0000e+09,  1.0000e+09,  1.0000e+09,  1.0000e+09,
          1.0000e+09,  1.0000e+09,  1.0000e+09],
        [ 9.8010e-02,  7.8847e-02,  5.8964e-02,  4.8991e-02,  7.5576e-02,
          1.5077e-01,  4.0231e-02,  4.2613e-02,  3.3940e-02,  6.8856e-02,
          4.7890e-02,  4.6411e-02,  5.9862e-02, -5.9602e-01, -5.9223e-01,
         -5.8280e-01, -5.3974e-01,  3.3947e-02, -5.2857e-01, -6.3361e-01,
          2.9617e-01, -6.5740e-01, -5.2138e-01, -2.1204e-01, -4.4517e-01,
          2.7549e-01, -5.9305e-01, -4.6948e-02, -4.9142e-01, -3.9962e-01,
         -1.1132e-01, -4.9142e-01,  3.0798e-01, -3.5342e-01, -3.1119e-01,
         -6.8861e-01, -2.2024e-01, -7.6455e-02,  4.6112e-01,  1.3402e-02,
         -1.6614e-01,  1.0000e+09,  1.0000e+09,  1.0000e+09,  1.0000e+09,
          1.0000e+09,  1.0000e+09,  1.0000e+09,  1.0000e+09,  1.0000e+09,
          1.0000e+09,  1.0000e+09,  1.0000e+09,  1.0000e+09,  1.0000e+09,
          1.0000e+09,  1.0000e+09,  1.0000e+09,  1.0000e+09,  1.0000e+09,
          1.0000e+09,  1.0000e+09,  1.0000e+09,  1.0000e+09,  1.0000e+09,
          1.0000e+09,  1.0000e+09,  1.0000e+09,  1.0000e+09,  1.0000e+09,
          1.0000e+09,  1.0000e+09,  1.0000e+09,  1.0000e+09,  1.0000e+09,
          1.0000e+09,  1.0000e+09,  1.0000e+09,  1.0000e+09,  1.0000e+09,
          1.0000e+09,  1.0000e+09,  1.0000e+09,  1.0000e+09,  1.0000e+09,
          1.0000e+09,  1.0000e+09,  1.0000e+09,  1.0000e+09,  1.0000e+09,
          1.0000e+09,  1.0000e+09,  1.0000e+09,  1.0000e+09,  1.0000e+09,
          1.0000e+09,  1.0000e+09,  1.0000e+09,  1.0000e+09,  1.0000e+09,
          1.0000e+09,  1.0000e+09,  1.0000e+09,  1.0000e+09,  1.0000e+09,
          1.0000e+09,  1.0000e+09,  1.0000e+09,  1.0000e+09,  1.0000e+09,
          1.0000e+09,  1.0000e+09,  1.0000e+09,  1.0000e+09,  1.0000e+09,
          1.0000e+09,  1.0000e+09,  1.0000e+09,  1.0000e+09,  1.0000e+09,
          1.0000e+09,  1.0000e+09,  1.0000e+09,  1.0000e+09,  1.0000e+09,
          1.0000e+09,  1.0000e+09,  1.0000e+09]]]
)
# print(yy.shape)
# print(knn(yy,16))



def knn_test(x, k): # x(N, 2, 128)  k=6
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
    
    idx = pairwise_distance.topk(k=k + 1, dim=-1)[1][:, :, 1:]  # (batch_size, num_points, k)
        
    return idx