from argparse import Namespace
from torch.utils.data import DataLoader
from weaver.utils.dataset import SimpleIterDataset
import os
import functools
import torch
import tqdm
import numpy as np
import awkward as ak
# 声明命令行参数
args = Namespace(backend=None, batch_size=1, copy_inputs=False, cross_validation=None, data_config='../data/TopLandscape/top_kin.yaml', data_fraction=1, data_test=['../datasets/TopLandscape/test_file.parquet'], data_train=[], data_val=[], demo=False, export_onnx=None, extra_selection=None, extra_test_selection=None, fetch_by_files=False, fetch_step=0.01, file_fraction=1, gpus='', in_memory=False, io_test=False, load_epoch=None, load_model_weights=None, local_rank=None, log='', lr_finder=None, lr_scheduler='flat+decay', model_prefix='./test/best_simple/net_best_epoch_state.pt', network_config='./network/example_Top_ParticleNet_zyt.py', network_option=[], no_remake_weights=False, num_epochs=20, num_workers=1, optimizer='ranger', optimizer_option=[], predict=True, predict_gpus='', predict_output='./test/output.root', print=False, profile=False, regression_mode=False, samples_per_epoch=None, samples_per_epoch_val=None, start_lr=0.005, steps_per_epoch=None, steps_per_epoch_val=None, tensorboard=None, tensorboard_custom_fn=None, train_val_split=0.8, use_amp=False, warmup_steps=0)

# 此args是调用test_load的args
file_dict = {'': ['../datasets/TopLandscape/test_file.parquet']} 


def test_load(args):
    def get_test_loader(name):
        
        filelist =  ['../datasets/TopLandscape/test_file.parquet']
        num_workers = min(args.num_workers, len(filelist))
        print("----------------------------",num_workers,"\n",args," ",type(args),filelist,"----------------------------------")
        test_data = SimpleIterDataset({name: filelist}, args.data_config, for_training=False,
                                    extra_selection=args.extra_test_selection,
                                    load_range_and_fraction=((0, 1), args.data_fraction),
                                    fetch_by_files=True, fetch_step=1,
                                    name='test_' + name)
        test_loader = DataLoader(test_data, num_workers=num_workers, batch_size=args.batch_size, drop_last=False,
                                    pin_memory=True)
        return test_loader   

    test_loaders = {name: functools.partial(get_test_loader, name) for name in file_dict}
    data_config = SimpleIterDataset({}, args.data_config, for_training=False).config
    return test_loaders, data_config


# for name in file_dict:
#     print(name,file_dict[name])
# print(data_config.__dict__)

"""
    if args.io_test:
        data_loader = train_loader if training_mode else list(test_loaders.values())[0]()
        iotest(args, data_loader)
        return
"""

# data_loader = list(test_loaders.values())[0]()

print(not args.model_prefix.endswith('.onnx'))

path = "/home/atzyt/Project/myparticle/particle_transformer/mymodel/readData/readData_torch/"
fullPath = path+"input_data.txt"
print(fullPath)
def _main(args):

    test_loaders, data_config = test_load(args)

    # test_loaders = test_load(args)

    # 遍历数据集
    for name, get_test_loader in test_loaders.items(): # 根据文件个数遍历

        test_loader = get_test_loader()
        print(test_loader.dataset.config.input_names) # 数据集的DataConfig

        data_config = test_loader.dataset.config
        with torch.no_grad():
            with tqdm.tqdm(test_loader) as tq:
                index = 0
                for X , y , Z in tq:
                    if(index == 0):
                        # inputs =  [X[k] for k in data_config.input_names]
                        # print(inputs)
                        for k in data_config.input_names :
                            with open("/home/atzyt/Project/myparticle/particle_transformer/mymodel/readData/readData_torch/input_data.txt","a+") as f:
                                print(k,file=f)
                                print(X[k],file=f)
                                f.close()
                        label = y[data_config.label_names[0]].long()
                        with open("/home/atzyt/Project/myparticle/particle_transformer/mymodel/readData/readData_torch/input_data.txt","a+") as f:
                            print(label,file=f)
                            f.close()
                        index = index + 1
                    else: 
                        break

    """
        参考: tools.py 中的 def train_classification()

        from weaver.utils.nn.tools import evaluate_classification as evaluate

        调用的evaluate是 evaluate_classification 函数

       input_names : ('pf_points', 'pf_features', 'pf_vectors', 'pf_mask')
    """


"""
_flatten_label 打平label
"""
def _flatten_label(label, mask=None):
    if label.ndim > 1:
        label = label.view(-1)
        if mask is not None:
            label = label[mask.view(-1)]
    # print('label', label.shape, label)
    return label

_main(args)


"""


{'options': {'treename': None, 'selection': None, 'test_time_selection': None, 'preprocess': {'method': 'manual', 'data_fraction': 0.5, 'params': None}, 'new_variables': {'part_mask': 'ak.ones_like(part_deta)', 'part_pt': 'np.hypot(part_px, part_py)', 'part_pt_log': 'np.log(part_pt)', 'part_e_log': 'np.log(part_energy)', 'part_logptrel': 'np.log(part_pt/jet_pt)', 'part_logerel': 'np.log(part_energy/jet_energy)', 'part_deltaR': 'np.hypot(part_deta, part_dphi)', 'jet_isTop': 'label', 'jet_isQCD': '1-label'}, 'inputs': {'pf_points': {'length': 128, 'pad_mode': 'wrap', 'vars': [['part_deta', None], ['part_dphi', None]]}, 'pf_features': {'length': 128, 'pad_mode': 'wrap', 'vars': [['part_pt_log', 1.7, 0.7], ['part_e_log', 2.0, 0.7], ['part_logptrel', -4.7, 0.7], ['part_logerel', -4.7, 0.7], ['part_deltaR', 0.2, 4.0], ['part_deta', None], ['part_dphi', None]]}, 'pf_vectors': {'length': 128, 'pad_mode': 'wrap', 'vars': [['part_px', None], ['part_py', None], ['part_pz', None], ['part_energy', None]]}, 'pf_mask': {'length': 128, 'pad_mode': 'constant', 'vars': [['part_mask', None]]}}, 'labels': {'type': 'simple', 'value': ['jet_isTop', 'jet_isQCD']}, 'observers': ['jet_pt', 'jet_eta'], 'monitor_variables': [], 'weights': None}, 'selection': None, 'test_time_selection': None, 'var_funcs': {'part_mask': 'ak.ones_like(part_deta)', 'part_pt': 'np.hypot(part_px, part_py)', 'part_pt_log': 'np.log(part_pt)', 'part_e_log': 'np.log(part_energy)', 'part_logptrel': 'np.log(part_pt/jet_pt)', 'part_logerel': 'np.log(part_energy/jet_energy)', 'part_deltaR': 'np.hypot(part_deta, part_dphi)', 'jet_isTop': 'label', 'jet_isQCD': '1-label', '_label_': 'np.argmax(np.stack([ak.to_numpy(jet_isTop),ak.to_numpy(jet_isQCD)], axis=1), axis=1)', '_labelcheck_': 'np.sum(np.stack([ak.to_numpy(jet_isTop),ak.to_numpy(jet_isQCD)], axis=1), axis=1)'}, 'preprocess': {'method': 'manual', 'data_fraction': 0.5, 'params': None}, '_auto_standardization': False, '_missing_standardization_info': False, 'preprocess_params': {'part_deta': {'length': 128, 'pad_mode': 'wrap', 'center': None, 'scale': 1, 'min': -5, 'max': 5, 'pad_value': 0}, 'part_dphi': {'length': 128, 'pad_mode': 'wrap', 'center': None, 'scale': 1, 'min': -5, 'max': 5, 'pad_value': 0}, 'part_pt_log': {'length': 128, 'pad_mode': 'wrap', 'center': 1.7, 'scale': 0.7, 'min': -5, 'max': 5, 'pad_value': 0}, 'part_e_log': {'length': 128, 'pad_mode': 'wrap', 'center': 2.0, 'scale': 0.7, 'min': -5, 'max': 5, 'pad_value': 0}, 'part_logptrel': {'length': 128, 'pad_mode': 'wrap', 'center': -4.7, 'scale': 0.7, 'min': -5, 'max': 5, 'pad_value': 0}, 'part_logerel': {'length': 128, 'pad_mode': 'wrap', 'center': -4.7, 'scale': 0.7, 'min': -5, 'max': 5, 'pad_value': 0}, 'part_deltaR': {'length': 128, 'pad_mode': 'wrap', 'center': 0.2, 'scale': 4.0, 'min': -5, 'max': 5, 'pad_value': 0}, 'part_px': {'length': 128, 'pad_mode': 'wrap', 'center': None, 'scale': 1, 'min': -5, 'max': 5, 'pad_value': 0}, 'part_py': {'length': 128, 'pad_mode': 'wrap', 'center': None, 'scale': 1, 'min': -5, 'max': 5, 'pad_value': 0}, 'part_pz': {'length': 128, 'pad_mode': 'wrap', 'center': None, 'scale': 1, 'min': -5, 'max': 5, 'pad_value': 0}, 'part_energy': {'length': 128, 'pad_mode': 'wrap', 'center': None, 'scale': 1, 'min': -5, 'max': 5, 'pad_value': 0}, 'part_mask': {'length': 128, 'pad_mode': 'constant', 'center': None, 'scale': 1, 'min': -5, 'max': 5, 'pad_value': 0}}, 'input_names': ('pf_points', 'pf_features', 'pf_vectors', 'pf_mask'), 'input_dicts': {'pf_points': ['part_deta', 'part_dphi'], 'pf_features': ['part_pt_log', 'part_e_log', 'part_logptrel', 'part_logerel', 'part_deltaR', 'part_deta', 'part_dphi'], 'pf_vectors': ['part_px', 'part_py', 'part_pz', 'part_energy'], 'pf_mask': ['part_mask']}, 'input_shapes': {'pf_points': (-1, 2, 128), 'pf_features': (-1, 7, 128), 'pf_vectors': (-1, 4, 128), 'pf_mask': (-1, 1, 128)}, 'label_type': 'simple', 'label_value': ['jet_isTop', 'jet_isQCD'], 'label_names': ('_label_',), 'basewgt_name': '_basewgt_', 'weight_name': None, 'observer_names': ('jet_pt', 'jet_eta'), 'monitor_variables': (), 'z_variables': ('jet_pt', 'jet_eta'), 'keep_branches': {'part_logptrel', '_labelcheck_', 'part_dphi', '_label_', 'part_py', 'jet_isQCD', 'part_px', 'part_pt_log', 'part_deta', 'jet_pt', 'part_pz', 'part_e_log', 'part_logerel', 'part_deltaR', 'jet_isTop', 'part_mask', 'jet_eta', 'part_energy', 'part_pt'}, 'drop_branches': {'label', 'jet_energy'}, 'load_branches': {'jet_pt', 'jet_energy', 'part_px', 'part_pz', 'jet_eta', 'label', 'part_energy', 'part_dphi', 'part_deta', 'part_py'}}

"""

        # print(test_loader.__dict__)
        # dataset = test_loaders.dataset
        # print(dataset.__dict__)
        # for name,label in test_loaders:
        # run prediction
    #     test_metric, scores, labels, observers = evaluate(
    # model, test_loader, dev, epoch=None, for_training=False, tb_helper=tb)