import torch.nn as nn

class Parent(nn.Module):
    def __init__(self,
                input_dims,
                num_classes,
                conv_params=[(7, (32, 32, 32)), (7, (64, 64, 64))],
                fc_params=[(128, 0.1)],
                use_fusion=True,
                use_fts_bn=True,
                use_counts=False,
                for_inference=False,
                for_segmentation=False,
                **kwargs):
        super(Parent).__init__(**kwargs)
        self.use_fusion = use_fusion
        self.use_fts_bn = use_fts_bn
        self.use_counts =  use_counts
        self.for_inference = for_inference
        print("kwargs:\n",kwargs)
        print("use_fusion:\n",self.use_fusion)
        print("use_fts_bn:\n",self.use_fts_bn)
        print("use_counts:\n",self.use_counts)
        print("for_inference:\n",self.for_inference)

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
    model = Parent(
        input_dims=111,
        num_classes=1,
        conv_params=kwargs.get('conv_params', conv_params),
        fc_params=kwargs.get('fc_params', fc_params),
        use_fusion=kwargs.get('use_fusion', False),
        use_fts_bn=kwargs.get('use_fts_bn', True),
        use_counts=kwargs.get('use_counts', True),  # 是否使用counts
        for_inference=kwargs.get('for_inference', False) # get方法,如果这个键有值则使用默认的，若没有使用指定的False
    )
    return model
network_option={}
# network_option['for_inference'] = True
# network_option['use_amp'] = True
get_model(**network_option)



mask = [[1,2,0]]
aaa = (mask == 0) * 22
print(aaa)