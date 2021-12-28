import argparse
import json
import os
from solver import Solver
from data_loader import get_loader
from torch.backends import cudnn
import random

def main(config):
    cudnn.benchmark = True

    if config.mode != 'testing':
        # Create directories if not exist
        if not os.path.exists(config.model_path):
           os.makedirs(config.model_path)
        ls_loader = get_loader(image_path=config.ls_dir,
                               image_size=config.image_size,
                               batch_size=config.batch_size,
                               shuffle=True,
                               mode='training',
                               model_type=config.model_type,
                               multi_style=config.multi_style,
                               warmup=config.warmup,
                               c_order=config.c_order,
                               num_workers=config.num_workers)
        print('ls_loader loaded')
        ls_val_loader = get_loader(image_path=config.ls_val_dir,
                                   image_size=config.image_size,
                                   batch_size=config.batch_size,
                                   shuffle=False,
                                   mode='validation',
                                   model_type=config.model_type,
                                   multi_style=False,
                                   warmup=config.warmup,
                                   c_order=config.c_order,
                                   num_workers=config.num_workers)
        print('ls_val_loader loaded')
        solver = Solver(config, ls_loader=ls_loader, ls_val_loader=ls_val_loader)
        solver.train()

    else:
        test_loader = []
        for test_n in range(len(config.test_dir)):
            dataloader = get_loader(image_path=config.test_dir[test_n],
                                    image_size=config.image_size,
                                    batch_size=config.batch_size,
                                    shuffle=False,
                                    mode='testing',
                                    model_type=config.model_type,
                                    multi_style=False,
                                    warmup=True,
                                    c_order=config.c_order,
                                    num_workers=config.num_workers)
            test_loader.append(dataloader)
        solver = Solver(config, test_loader=test_loader)
        print('testing set loaded')
        solver.test()

    
def Config(d=None, **kwargs):
    def funcdict(d=None, **kwargs):
        if d is not None:
            funcdict.__dict__.update(d)
        funcdict.__dict__.update(kwargs)
        return funcdict.__dict__
    funcdict(d, **kwargs)
    return funcdict

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--config_path', type=str, default='config.json')
    args = parser.parse_args()

    with open(args.config_path, 'r')  as f:
        setting = json.load(f)
    
    config = Config(setting)
    print(setting)

    main(config)
