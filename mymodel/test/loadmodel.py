import torch
import torch.nn as nn
# import cv2

import torch.nn.functional as F
import time
# from train import Net
from torchvision import datasets, transforms
from PIL import Image
import sys,os
# sys.path.append('需要作为模块引入的路径')
# 添加当前路径的前一级文件作为源文件夹
path = os.path.dirname(os.path.dirname(__file__)) 
print(path)
sys.path.append(path)

# from ParticleNet_zyt import ParticleNet
from TopLand_ParticleNet import ParticleNet




class ParticleNetWrapper(torch.nn.Module):
    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.mod = ParticleNet(**kwargs) # 初始化模型,并设置一些参数

    def forward(self, points, features, lorentz_vectors, mask): #前向传播
        return self.mod(points, features, mask)




def get_model(**kwargs):
    conv_params = [
        (16, (64, 64, 64)),
        (16, (128, 128, 128)),
        (16, (256, 256, 256)),
    ]
    fc_params = [(256, 0.1)]
    #**接受键值对的不定量参数，network_option
    # pf_features_dims = len(data_config.input_dicts['pf_features'])
    # num_classes = len(data_config.label_value)

    # print("input_dims = pf_features_dims = {len}\n".format(len = len(data_config.input_dicts['pf_features']) ))  # 7
    # print("num_classes = num_classes = {len}".format(len = len(data_config.label_value) ))                       # 2
    
    model = ParticleNetWrapper(
        input_dims=7,
        num_classes=2,
        conv_params=kwargs.get('conv_params', conv_params),
        fc_params=kwargs.get('fc_params', fc_params),
        use_fusion=kwargs.get('use_fusion', False),
        use_fts_bn=kwargs.get('use_fts_bn', True),  # 输入之前先将数据归一化处理
        use_counts=kwargs.get('use_counts', True),  # 是否使用counts
        for_inference=kwargs.get('for_inference', False)
    )

    # model_info = {
    #     'input_names': list(data_config.input_names),
    #     'input_shapes': {k: ((1,) + s[1:]) for k, s in data_config.input_shapes.items()},
    #     'output_names': ['softmax'],
    #     'dynamic_axes': {**{k: {0: 'N', 2: 'n_' + k.split('_')[0]} for k in data_config.input_names}, **{'softmax': {0: 'N'}}},
    # }

    # print("------------------modify-------")
    # return model, model_info
    return model

# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
#         self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
#         self.conv2_drop = nn.Dropout2d()
#         self.fc1 = nn.Linear(320, 50)
#         self.fc2 = nn.Linear(50, 10)

#     def forward(self, x):
#         x = self.conv1(x)
#         x = F.relu(F.max_pool2d(x, 2))
#         x = self.conv2(x)
#         x = F.relu(F.max_pool2d(self.conv2_drop(x), 2))
#         x = x.view(-1, 320)
#         x = F.relu(self.fc1(x))
#         x = F.dropout(x, training=self.training)
#         x = self.fc2(x)
#         return F.log_softmax(x, dim=1)

# import pandas as pd


network = get_model()
# print(network) network(input)就会执行前馈
# network.eval()
# 加载模型参数
# ##加载训练好的模型参数
# state_dict = torch.load('./model.pth')
# network.load_state_dict(state_dict)

# state_dict = torch.load('./net_best_epoch_state.pt').state_dict()
# network.load_state_dict(state_dict)

state_dict = torch.load('./net_best_epoch_state.pt')
network.load_state_dict(state_dict,strict=False)
import pandas as pd
# #这里是导出模型的权重参数,会员python代码的目录生成几个.csv文件
for name,param in network.named_parameters():
    print(f"name:{name}\t\t\t,shape:{param.shape}")
    data = pd.DataFrame(param.detach().numpy().reshape(1, -1))
    filename = f"{name}.csv"
    data.to_csv(f"./{filename}", index=False, header=False, sep=',')