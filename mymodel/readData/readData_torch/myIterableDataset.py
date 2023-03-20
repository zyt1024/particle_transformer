from torch.utils.data import IterableDataset

class MyIterableDataset(IterableDataset):

    def __init__(self, file_path):
        self.file_path = file_path
    
    def __iter__(self):
       with open(self.file_path, 'r') as file_obj:
            for line in file_obj: # 更多操作在这里完成
                line_data = line.strip('\n').split(',')
                yield line_data


if __name__ == '__main__':
    dataset = MyIterableDataset('test_csv.csv') # 调用
    for data in dataset:
        print(data)



_sampler_options = {
    'up_sample': True,
    'weight_scale': 1,
    'max_resample': 10,
}

_sampler_options.update(training=False, shuffle=False, reweight=False)
dict2={'zyt':1}
_sampler_options.update(dict2)
print(_sampler_options)

dic1 = {}
dic1.update(backend=None, batch_size=1, copy_inputs=False, cross_validation=None, data_config='../data/TopLandscape/top_kin.yaml', data_fraction=1, data_test=['../datasets/TopLandscape/test_file.parquet'], data_train=[], data_val=[], demo=False, export_onnx=None, extra_selection=None, extra_test_selection=None, fetch_by_files=False, fetch_step=0.01, file_fraction=1, gpus='', in_memory=False, io_test=False, load_epoch=None, load_model_weights=None, local_rank=None, log='', lr_finder=None, lr_scheduler='flat+decay', model_prefix='./test/best_simple/net_best_epoch_state.pt', network_config='./network/example_Top_ParticleNet_zyt.py', network_option=[], no_remake_weights=False, num_epochs=20, num_workers=1, optimizer='ranger', optimizer_option=[], predict=True, predict_gpus='', predict_output='./test/output.root', print=False, profile=False, regression_mode=False, samples_per_epoch=None, samples_per_epoch_val=None, start_lr=0.005, steps_per_epoch=None, steps_per_epoch_val=None, tensorboard=None, tensorboard_custom_fn=None, train_val_split=0.8, use_amp=False, warmup_steps=0)
print(dic1)
# print(dic1.data_config)

from argparse import Namespace
aa = Namespace(backend=None, batch_size=1, copy_inputs=False, cross_validation=None, data_config='../data/TopLandscape/top_kin.yaml', data_fraction=1, data_test=['../datasets/TopLandscape/test_file.parquet'], data_train=[], data_val=[], demo=False, export_onnx=None, extra_selection=None, extra_test_selection=None, fetch_by_files=False, fetch_step=0.01, file_fraction=1, gpus='', in_memory=False, io_test=False, load_epoch=None, load_model_weights=None, local_rank=None, log='', lr_finder=None, lr_scheduler='flat+decay', model_prefix='./test/best_simple/net_best_epoch_state.pt', network_config='./network/example_Top_ParticleNet_zyt.py', network_option=[], no_remake_weights=False, num_epochs=20, num_workers=1, optimizer='ranger', optimizer_option=[], predict=True, predict_gpus='', predict_output='./test/output.root', print=False, profile=False, regression_mode=False, samples_per_epoch=None, samples_per_epoch_val=None, start_lr=0.005, steps_per_epoch=None, steps_per_epoch_val=None, tensorboard=None, tensorboard_custom_fn=None, train_val_split=0.8, use_amp=False, warmup_steps=0)

print(aa)
print(aa.data_config)