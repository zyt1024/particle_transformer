''' ParticleNet

Paper: "ParticleNet: Jet Tagging via Particle Clouds" - https://arxiv.org/abs/1902.08570

Adapted from the DGCNN implementation in https://github.com/WangYueFt/dgcnn/blob/master/pytorch/model.py.


针对于此数据集的ParticleNet:https://zenodo.org/record/2603256

'''
import numpy as np
import torch
import torch.nn as nn
import time
from weaver.utils.logger import _logger, _configLogger
import sys
stdout = sys.stdout
_configLogger('weaver', stdout=stdout, filename="infer_log.log")
import numpy as np
torch.set_printoptions(threshold=np.inf)   # 配置打印的时候不显示省略号

def knn(x, k): # x(N, 2, 128)  k=6
    # _logger.info("knn,x={x}".format(x=x[0,:,:]))
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
    
    idx = pairwise_distance.topk(k=k + 1, dim=-1)[1][:, :, 1:]  # (batch_size, num_points, k)
    
    # _logger.info("idx,idx={idx}".format(idx=idx[0,:,:]))

    return idx


# v1 is faster on GPU
def get_graph_feature_v1(x, k, idx):
    batch_size, num_dims, num_points = x.size()
    _logger.info("num_dims={num_dims}".format(num_dims=num_dims))
    _logger.info("num_points={num_points}".format(num_points=num_points))
    _logger.info("x.size()={x}".format(x=x.size()))
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
    batch_size, num_dims, num_points = x.size() # torch.Size([512, 7, 128]) torch.Size([512, 7, 128]) torch.Size([512, 7, 128])

    idx_base = torch.arange(0, batch_size, device=x.device).view(-1, 1, 1) * num_points
    idx = idx + idx_base
    idx = idx.view(-1)

    fts = x.transpose(0, 1).reshape(num_dims, -1)  # -> (num_dims, batch_size, num_points) -> (num_dims, batch_size*num_points)
    fts = fts[:, idx].view(num_dims, batch_size, num_points, k)  # neighbors: -> (num_dims, batch_size*num_points*k) -> ...
    fts = fts.transpose(1, 0).contiguous()  # (batch_size, num_dims, num_points, k)

    x = x.view(batch_size, num_dims, num_points, 1).repeat(1, 1, 1, k)
    fts = torch.cat((x, fts - x), dim=1)  # ->(batch_size, 2*num_dims, num_points, k) # torch.Size([512, 14, 128,16])

    return fts


class EdgeConvBlock(nn.Module):
    r"""EdgeConv layer.
    Introduced in "`Dynamic Graph CNN for Learning on Point Clouds
    <https://arxiv.org/pdf/1801.07829>`__".  Can be described as follows:
    .. math::
       x_i^{(l+1)} = \max_{j \in \mathcal{N}(i)} \mathrm{ReLU}(
       \Theta \cdot (x_j^{(l)} - x_i^{(l)}) + \Phi \cdot x_i^{(l)})
    where :math:`\mathcal{N}(i)` is the neighbor of :math:`i`.
    Parameters
    ----------
    in_feat : int
        Input feature size.
    out_feat : int
        Output feature size.
    batch_norm : bool
        Whether to include batch normalization on messages.
    """

    def __init__(self, k, in_feat, out_feats, batch_norm=True, activation=True, cpu_mode=False):
        super(EdgeConvBlock, self).__init__()
        self.k = k
        self.batch_norm = batch_norm
        self.activation = activation
        self.num_layers = len(out_feats)
        self.get_graph_feature = get_graph_feature_v2 if cpu_mode else get_graph_feature_v1

        self.convs = nn.ModuleList()
        for i in range(self.num_layers):
            # self.convs.append(nn.Conv2d(2 * in_feat if i == 0 else out_feats[i - 1], out_feats[i], kernel_size=1, bias=False if self.batch_norm else True))
            self.convs.append(nn.Conv2d(2 * in_feat if i == 0 else out_feats[i - 1], out_feats[i], kernel_size=1, bias=False))


        # if batch_norm:
        #     self.bns = nn.ModuleList()
        #     for i in range(self.num_layers):
        #         self.bns.append(nn.BatchNorm2d(out_feats[i]))

        self.bns = nn.ModuleList()
        for i in range(self.num_layers):
            self.bns.append(nn.BatchNorm2d(out_feats[i])) # ()

        # if activation:
        self.acts = nn.ModuleList()
        for i in range(self.num_layers):
            self.acts.append(nn.ReLU())

        # if in_feat == out_feats[-1]:
        #     self.sc = None # 如果输入通道和输出通道相同就不需要了,本次模型一直没设置
        # else:
        #     self.sc = nn.Conv1d(in_feat, out_feats[-1], kernel_size=1, bias=False)
        #     self.sc_bn = nn.BatchNorm1d(out_feats[-1])

        
        # shot up 
        self.sc = nn.Conv1d(in_feat, out_feats[-1], kernel_size=1, bias=False)
        self.sc_bn = nn.BatchNorm1d(out_feats[-1])
        # if activation:
        #     self.sc_act = nn.ReLU()

        self.sc_act = nn.ReLU()

    def forward(self, points, features):

        _logger.info("points:{points},shape:{shape}".format(points=points,shape=points.shape))
        topk_indices = knn(points, self.k)   # 计算距离，找出最近的K个点
        x = self.get_graph_feature(features, self.k, topk_indices) #  将最近的K个点和feature构造为输入数据

        _logger.info("top_k_indices:{val}top_k_indices.shape:{shape}".format(val=topk_indices,shape=topk_indices.shape))
        _logger.info("x.shape:{shape}".format(shape=x.shape))
        # for conv, bn, act in zip(self.convs, self.bns, self.acts):
        #     x = conv(x)  # (N, C', P, K)
        #     if bn:
        #         x = bn(x)
        #     if act:
        #         x = act(x)

        # 将for循环展开
        x = self.convs[0](x)
        x = self.bns[0](x)
        x = self.acts[0](x)

        x = self.convs[1](x)
        x = self.bns[1](x)
        x = self.acts[1](x)

        x = self.convs[2](x)
        x = self.bns[2](x)
        x = self.acts[2](x)

        fts = x.mean(dim=-1)  # (N, C, P)

        # shortcut
        # if self.sc:
        #     sc = self.sc(features)  # (N, C_out, P)
        #     sc = self.sc_bn(sc)
        # else:
        #     sc = features

        sc = self.sc(features)  # (N, C_out, P)
        sc = self.sc_bn(sc)
        
        return self.sc_act(sc + fts)  # (N, C_out, P) # 两个加在一起聚合


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

        self.edge_convs = nn.ModuleList()
        for idx, layer_param in enumerate(conv_params):
            k, channels = layer_param
            in_feat = input_dims if idx == 0 else conv_params[idx - 1][1][-1]
            self.edge_convs.append(EdgeConvBlock(k=k, in_feat=in_feat, out_feats=channels, cpu_mode=for_inference))

        """"该数据集不需要"""
        # self.use_fusion = use_fusion
        # if self.use_fusion:
        #     in_chn = sum(x[-1] for _, x in conv_params)
        #     out_chn = np.clip((in_chn // 128) * 128, 128, 1024)
        #     self.fusion_block = nn.Sequential(nn.Conv1d(in_chn, out_chn, kernel_size=1, bias=False), nn.BatchNorm1d(out_chn), nn.ReLU())

        self.for_segmentation = for_segmentation

        fcs = []
        for idx, layer_param in enumerate(fc_params):
            channels, drop_rate = layer_param
            if idx == 0:
                in_chn = conv_params[-1][1][-1] # 输入通道
                # in_chn = out_chn if self.use_fusion else conv_params[-1][1][-1]
            else:
                in_chn = fc_params[idx - 1][0]
            # if self.for_segmentation: # 为False
            #     fcs.append(nn.Sequential(nn.Conv1d(in_chn, channels, kernel_size=1, bias=False),
            #                              nn.BatchNorm1d(channels), nn.ReLU(), nn.Dropout(drop_rate)))
            # else:
            #     fcs.append(nn.Sequential(nn.Linear(in_chn, channels), nn.ReLU(), nn.Dropout(drop_rate)))
            fcs.append(nn.Sequential(nn.Linear(in_chn, channels), nn.ReLU(), nn.Dropout(drop_rate)))
        # if self.for_segmentation: # 为Flase
        #     fcs.append(nn.Conv1d(fc_params[-1][0], num_classes, kernel_size=1))
        # else:
        #     fcs.append(nn.Linear(fc_params[-1][0], num_classes))
        fcs.append(nn.Linear(fc_params[-1][0], num_classes))
        self.fc = nn.Sequential(*fcs)

        self.for_inference = for_inference

    def forward(self, points, features, mask=None):

        print("\n")
        _logger.info("points.shape: {point}".format(point=points.shape))
        _logger.info("features.shape: {features}".format(features=features.shape))
        _logger.info("mask.shape: {mask}".format(mask=mask.shape))

        # if mask is None:  # 针对于此数据集、可取消
        #     print("==================mask is None=========================")
        #     mask = (features.abs().sum(dim=1, keepdim=True) != 0)  # (N, 1, P)
        
        points *= mask
        features *= mask
        coord_shift = (mask == 0) * 1e9
        # if self.use_counts:
        #     counts = mask.float().sum(dim=-1)
        #     counts = torch.max(counts, torch.ones_like(counts))  # >=1

        counts = mask.float().sum(dim=-1)
        counts = torch.max(counts, torch.ones_like(counts))  # >=1

        # _logger.info("coord_shift: {coord_shift}".format(coord_shift=coord_shift))
        _logger.info("counts.shape: {counts}".format(counts=counts.shape))

        # if self.use_fts_bn: # 为TRUE
        #     fts = self.bn_fts(features) * mask
        # else:
        #     fts = features
        
        #self.bn_fts = nn.BatchNorm1d(input_dims)
        fts = self.bn_fts(features) * mask  # 先将feature做了一次归一化处理


        outputs = []
        # for idx, conv in enumerate(self.edge_convs):
        #     pts = (points if idx == 0 else fts) + coord_shift
        #     fts = conv(pts, fts) * mask
            # if self.use_fusion:
            #     outputs.append(fts)

        """ 将for 展开 """
        #第一个EdgeConvBlock
        pts = points + coord_shift

        _logger.info("第一个EdgeConvBlock:(pts={pts},fts={fts})".format(pts=pts.shape,fts=fts.shape))
        fts = self.edge_convs[0](pts, fts) * mask

        #第二个EdgeConvBlock
        pts = fts + coord_shift
        fts = self.edge_convs[1](pts, fts) * mask      

        #第三个EdgeConvBlock
        pts = fts + coord_shift
        fts = self.edge_convs[2](pts, fts) * mask     




        # if self.use_fusion:
        #     fts = self.fusion_block(torch.cat(outputs, dim=1)) * mask

#         assert(((fts.abs().sum(dim=1, keepdim=True) != 0).float() - mask.float()).abs().sum().item() == 0)
        
        # if self.for_segmentation:
        #     x = fts
        # else:
        #     if self.use_counts:
        #         x = fts.sum(dim=-1) / counts  # divide by the real counts
        #     else:
        #         x = fts.mean(dim=-1)

        x = fts.sum(dim=-1) / counts

        _logger.info("fts.sum(dim=-1).shape: {fts}".format(fts=fts.sum(-1).shape))

        output = self.fc(x)
        if self.for_inference:
            output = torch.softmax(output, dim=1)
        # print('output============:\n', output)
        return output

