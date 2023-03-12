
from torch.utils.data import DataLoader
from weaver.utils.logger import _logger, _configLogger
from weaver.utils.dataset import SimpleIterDataset
from weaver.utils.import_tools import import_module
import glob
import functools
def test_load():
    """
    Loads the test data.
    :param args:
    :return: test_loaders, data_config
    """
    # keyword-based --data-test: 'a:/path/to/a b:/path/to/b'
    # split --data-test: 'a%10:/path/to/a/*'
    file_dict = {}
    split_dict = {}
    # args = {}
    data_config = "../../data/TopLandscape/top_kin.yaml"
    data_test=["../../datasets/TopLandscape/test_file.parquet"]
    extra_test_selection=None
    data_fraction = 0.1
    batch_size = 1
    # args.add(data_config)
    # args.add(data_test)
    # args.extra_test_selection=[]
    # args.data_fraction=[]
    # args.batch_size= 1
    for f in data_test:
        if ':' in f:
            name, fp = f.split(':')
            if '%' in name:
                name, split = name.split('%')
                split_dict[name] = int(split)
        else:
            name, fp = '', f
        files = glob.glob(fp)
        if name in file_dict:
            file_dict[name] += files
        else:
            file_dict[name] = files
    print(file_dict)
    # sort files
    for name, files in file_dict.items():
        file_dict[name] = sorted(files)

    # apply splitting
    for name, split in split_dict.items():
        files = file_dict.pop(name)
        for i in range((len(files) + split - 1) // split):
            file_dict[f'{name}_{i}'] = files[i * split:(i + 1) * split]

    def get_test_loader(name):
        filelist = file_dict[name]
        _logger.info('Running on test file group %s with %d files:\n...%s', name, len(filelist), '\n...'.join(filelist))
        num_workers = min(1, len(filelist))
        test_data = SimpleIterDataset({name: filelist}, "../../data/TopLandscape/top_kin.yaml", for_training=False,
                                      extra_selection=extra_test_selection,
                                      load_range_and_fraction=((0, 1), data_fraction),
                                      fetch_by_files=True, fetch_step=1,
                                      name='test_' + name)
        test_loader = DataLoader(test_data, num_workers=num_workers, batch_size=batch_size, drop_last=False,
                                 pin_memory=True)
        return test_loader

    test_loaders = {name: functools.partial(get_test_loader, name) for name in file_dict}
    data_config = SimpleIterDataset({}, data_config, for_training=False).config
    return test_loaders, data_config


test_loaders, data_config = test_load()
print(test_loaders)
print(data_config)

# data_loader = list(test_loaders.values())[0]()

for name, get_test_loader in test_loaders.items():
    print(name)
    print(get_test_loader)