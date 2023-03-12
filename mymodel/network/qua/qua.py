#-*- coding:utf-8 -*-
import torch
 
 
...
 
 
 
#量化仅可用cpu
networrk = ResNet().cpu()

state_dict = torch.load('./net_best_epoch_state.pt')
network.load_state_dict(state_dict,strict=False)
 
#Specify quantization configuration
#在这一步声明了对称量化或非对称量化，及量化bit数
#如下代码中采用了默认的非对称量化，及8bit
model.qconfig = torch.quantization.default_qconfig
model = torch.quantization.prepare(model)
 
#Convert to quantized model
model = torch.quantization.convert(model)
 
#Save model, 保存后模型的size显著减小，但性能损失相对较大
#故，建议考虑量化感知训练
torch.save(model.state_dict(), "path.pt")