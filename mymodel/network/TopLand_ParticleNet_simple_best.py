''' ParticleNet

Paper: "ParticleNet: Jet Tagging via Particle Clouds" - https://arxiv.org/abs/1902.08570

Adapted from the DGCNN implementation in https://github.com/WangYueFt/dgcnn/blob/master/pytorch/model.py.


针对于此数据集的ParticleNet:https://zenodo.org/record/2603256

'''
import numpy as np
import torch
import torch.nn as nn
import time

def knn(x, k):
    
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
    idx = pairwise_distance.topk(k=k + 1, dim=-1)[1][:, :, 1:]  # (batch_size, num_points, k)
    #[1][:, :, 1:]

    return idx


# v1 is faster on GPU
def get_graph_feature_v1(x, k, idx):
    batch_size, num_dims, num_points = x.size()

    idx_base = torch.arange(0, batch_size, device=x.device).view(-1, 1, 1) * num_points
    idx = idx + idx_base
    idx = idx.view(-1)

    fts = x.transpose(2, 1).reshape(-1, num_dims)  # -> (batch_size, num_points, num_dims) -> (batch_size*num_points, num_dims)
    fts = fts[idx, :].view(batch_size, num_points, k, num_dims)  # neighbors: -> (batch_size*num_points*k, num_dims) -> ...
    fts = fts.permute(0, 3, 1, 2).contiguous()  # (batch_size, num_dims, num_points, k)
    x = x.view(batch_size, num_dims, num_points, 1).repeat(1, 1, 1, k)
    fts = torch.cat((x, fts - x), dim=1)  # ->(batch_size, 2*num_dims, num_points, k)
    return fts


# v2 is faster on CPU
def get_graph_feature_v2(x, k, idx):
    batch_size, num_dims, num_points = x.size()

    idx_base = torch.arange(0, batch_size, device=x.device).view(-1, 1, 1) * num_points
    idx = idx + idx_base
    idx = idx.view(-1)

    fts = x.transpose(0, 1).reshape(num_dims, -1)  # -> (num_dims, batch_size, num_points) -> (num_dims, batch_size*num_points)
    fts = fts[:, idx].view(num_dims, batch_size, num_points, k)  # neighbors: -> (num_dims, batch_size*num_points*k) -> ...
    fts = fts.transpose(1, 0).contiguous()  # (batch_size, num_dims, num_points, k)

    x = x.view(batch_size, num_dims, num_points, 1).repeat(1, 1, 1, k)
    fts = torch.cat((x, fts - x), dim=1)  # ->(batch_size, 2*num_dims, num_points, k)

    return fts


class ParticleNet(nn.Module):

    def __init__(self,
                 input_dims,
                 num_classes,
                 conv_params=[(7, (32, 32, 32)), (7, (64, 64, 64))],
                 fc_params=[(128, 0.1)],
                 use_fusion=True,
                 use_fts_bn=True,
                 use_counts=True,
                 for_inference=False,
                 for_segmentation=False,
                 **kwargs):
        super(ParticleNet, self).__init__(**kwargs)

        self.use_fts_bn = use_fts_bn
        # if self.use_fts_bn: # 默认为True
        #     self.bn_fts = nn.BatchNorm1d(input_dims)
        self.bn_fts = nn.BatchNorm1d(input_dims)


        self.use_counts = use_counts # 为True


        # 初始化第一个Edge_convs
        self.edge_convs0_conv0 = nn.Conv2d(14,64,kernel_size=1,bias=False)
        self.edge_convs0_conv1 = nn.Conv2d(64,64,kernel_size=1,bias=False)
        self.edge_convs0_conv2 = nn.Conv2d(64,64,kernel_size=1,bias=False)

        self.edge_convs0_bn0 = nn.BatchNorm2d(64)
        self.edge_convs0_bn1 = nn.BatchNorm2d(64)
        self.edge_convs0_bn2 = nn.BatchNorm2d(64)

        self.edge_convs0_ac0 = nn.ReLU()
        self.edge_convs0_ac1 = nn.ReLU()
        self.edge_convs0_ac2 = nn.ReLU()

        self.edge_convs0_shot_conv = nn.Conv1d(7,64,kernel_size=1,bias=False)
        self.edge_convs0_shot_bn = nn.BatchNorm1d(64)
        self.edge_convs0_shot_ac = nn.ReLU()

        # 初始化第二个Edge_convs
        self.edge_convs1_conv0 = nn.Conv2d(128,128,kernel_size=1,bias=False)
        self.edge_convs1_conv1 = nn.Conv2d(128,128,kernel_size=1,bias=False)
        self.edge_convs1_conv2 = nn.Conv2d(128,128,kernel_size=1,bias=False)

        self.edge_convs1_bn0 = nn.BatchNorm2d(128)
        self.edge_convs1_bn1 = nn.BatchNorm2d(128)
        self.edge_convs1_bn2 = nn.BatchNorm2d(128)

        self.edge_convs1_ac0 = nn.ReLU()
        self.edge_convs1_ac1 = nn.ReLU()
        self.edge_convs1_ac2 = nn.ReLU()

        self.edge_convs1_shot_conv = nn.Conv1d(64,128,kernel_size=1,bias=False)
        self.edge_convs1_shot_bn = nn.BatchNorm1d(128)
        self.edge_convs1_shot_ac = nn.ReLU()

        # 初始化第三个Edge_convs
        self.edge_convs2_conv0 = nn.Conv2d(256,256,kernel_size=1,bias=False)
        self.edge_convs2_conv1 = nn.Conv2d(256,256,kernel_size=1,bias=False)
        self.edge_convs2_conv2 = nn.Conv2d(256,256,kernel_size=1,bias=False)
    
        self.edge_convs2_bn0 = nn.BatchNorm2d(256)
        self.edge_convs2_bn1 = nn.BatchNorm2d(256)
        self.edge_convs2_bn2 = nn.BatchNorm2d(256)

        self.edge_convs2_ac0 = nn.ReLU()
        self.edge_convs2_ac1 = nn.ReLU()
        self.edge_convs2_ac2 = nn.ReLU()   
       
        self.edge_convs2_shot_conv = nn.Conv1d(128,256,kernel_size=1,bias=False)
        self.edge_convs2_shot_bn = nn.BatchNorm1d(256)
        self.edge_convs2_shot_ac = nn.ReLU()
        


        self.for_segmentation = for_segmentation

        fcs = []
        for idx, layer_param in enumerate(fc_params):
            channels, drop_rate = layer_param
            if idx == 0:
                in_chn = conv_params[-1][1][-1] # 输入通道

            else:
                in_chn = fc_params[idx - 1][0]

            fcs.append(nn.Sequential(nn.Linear(in_chn, channels), nn.ReLU(), nn.Dropout(drop_rate)))

        fcs.append(nn.Linear(fc_params[-1][0], num_classes))
        self.fc = nn.Sequential(*fcs)

        self.for_inference = for_inference

    def forward(self, points, features, mask=None):
        print('\npoints:\n', points.shape,points.type())
        print('features:\n', features.shape,features.type())
        print('mask:\n', mask.shape,mask.type())
        # if mask is None:  # 针对于此数据集、可取消
        #     print("==================mask is None=========================")
        #     mask = (features.abs().sum(dim=1, keepdim=True) != 0)  # (N, 1, P)
        points *= mask
        features *= mask
        coord_shift = (mask == 0) * 1e9
        # if self.use_counts:
        #     counts = mask.float().sum(dim=-1)
        #     counts = torch.max(counts, torch.ones_like(counts))  # >=1


        counts = mask.float().sum(dim=-1) # 将mask转换为float，并将最后一维求和
        

        # 生成与counts相同形状的张量,并且将>=1的取出
        counts = torch.max(counts, torch.ones_like(counts))  # >=1

        # if self.use_fts_bn: # 为TRUE
        #     fts = self.bn_fts(features) * mask
        # else:
        #     fts = features
        

        outputs = []
        # for idx, conv in enumerate(self.edge_convs):
        #     pts = (points if idx == 0 else fts) + coord_shift
        #     fts = conv(pts, fts) * mask
            # if self.use_fusion:
            #     outputs.append(fts)

        """ 将for 展开 """
        #第一个EdgeConvBlock
        start_edge_1 = time.time()
        fts = self.bn_fts(features) * mask  # 7
        pts = points + coord_shift
        print("fts000=",fts)
        print("pts000=",pts)
        print("edge1:",fts.shape,pts.shape)
        #edge1: torch.Size([1, 7, 128]) torch.Size([1, 2, 128])
        # pts = point
        # fts = features


        print("knn0:",pts.shape)
        topk_indices = knn(pts, 16)   # 计算距离，找出最近的K个点

        x = get_graph_feature_v1(fts, 16, topk_indices) #

        print("edge0 input shape:",x.shape)
        x = self.edge_convs0_ac0(self.edge_convs0_bn0(self.edge_convs0_conv0(x)))
        x = self.edge_convs0_ac1(self.edge_convs0_bn1(self.edge_convs0_conv1(x)))
        x = self.edge_convs0_ac2(self.edge_convs0_bn2(self.edge_convs0_conv2(x)))
        
        # x.shape => NCHW (1,64,128,16)
        print("edge0 x=",x.shape)
        x = x.mean(dim=-1)  # 最后一维求均值
        # shortcut  输入1*7*128
        sc = self.edge_convs0_shot_bn(self.edge_convs0_shot_conv(fts))
        print("edge0 sc,x",sc.shape,x.shape)
        #sc:torch.Size([1, 64, 128])  x:torch.Size([1, 64, 128])
        fts = self.edge_convs0_shot_ac(sc + x) * mask
        end_edge_1 = time.time()

        # 第一层Edconv结束

        # fts = self.edge_convs[0](pts, fts) * mask

        #第二个EdgeConvBlock
        pts = fts + coord_shift
        start_edge_2 = time.time()
        print("knn1:",pts.shape)
        topk_indices = knn(pts, 16)   # 计算距离，找出最近的K个点
        x = get_graph_feature_v1(fts, 16, topk_indices) #
        x = self.edge_convs1_ac0(self.edge_convs1_bn0(self.edge_convs1_conv0(x)))
        x = self.edge_convs1_ac1(self.edge_convs1_bn1(self.edge_convs1_conv1(x)))
        x = self.edge_convs1_ac2(self.edge_convs1_bn2(self.edge_convs1_conv2(x)))
        
        x = x.mean(dim=-1)
        # shortcut
        sc = self.edge_convs1_shot_bn(self.edge_convs1_shot_conv(fts))

        fts = self.edge_convs1_shot_ac(sc + x) * mask
        # fts = self.edge_convs[1](pts, fts) * mask      
        end_edge_2 = time.time()

        #第三个EdgeConvBlock
        start_edge_3 = time.time()
        pts = fts + coord_shift
        start_knn = time.time()
        print("knn2:",pts.shape)
        topk_indices = knn(pts, 16)   # 计算距离，找出最近的K个点
        x = get_graph_feature_v1(fts, 16, topk_indices) #
        end_knn = time.time()
        x = self.edge_convs2_ac0(self.edge_convs2_bn0(self.edge_convs2_conv0(x)))
        x = self.edge_convs2_ac1(self.edge_convs2_bn1(self.edge_convs2_conv1(x)))
        x = self.edge_convs2_ac2(self.edge_convs2_bn2(self.edge_convs2_conv2(x)))
        
        x = x.mean(dim=-1)
        # shortcut
        sc = self.edge_convs2_shot_bn(self.edge_convs2_shot_conv(fts))
        fts = self.edge_convs2_shot_ac(sc + x) * mask
        end_edge_3 = time.time()

        # fts = self.edge_convs[2](pts, fts) * mask     

        x = fts.sum(dim=-1) / counts
        
        output = self.fc(x)
        if self.for_inference:
            output = torch.softmax(output, dim=1)
        # print('output============:\n', output)

        print("edge_conv3_knn: ",end_knn-start_knn)
        print("edge_conv1: ",end_edge_1-start_edge_1)
        print("edge_conv2: ",end_edge_2-start_edge_2)
        print("edge_conv3: ",end_edge_3-start_edge_3)      
        print("all:",end_edge_3 - start_edge_1) 
        return output

