from os import path as osp
import random
import numpy as np

import torch
from torch.utils import data as data
from torchvision.transforms.functional import normalize

from basicsr.data.data_util import paths_from_lmdb
from basicsr.utils import FileClient, imfrombytes, img2tensor, rgb2ycbcr, scandir
from basicsr.utils.registry import DATASET_REGISTRY

def get_patch(x, patch_size=96):
    ih, iw = x.shape[:2]
    ix = random.randrange(0, iw - patch_size + 1)
    iy = random.randrange(0, ih - patch_size + 1)
    return x[iy: iy + patch_size, ix: ix + patch_size, :]

@DATASET_REGISTRY.register()
class NoisyImageDataset(data.Dataset):
    """Read HQ (High Quality) images and synthesize noisy HQ images.

    Args:
        opt (dict): Config for train datasets. It contains the following keys:
            dataroot_hq (str): Data root path for hq.
            meta_info_file (str): Path for meta information file.
            io_backend (dict): IO backend type and other kwarg.
    """

    def __init__(self, opt):
        super(NoisyImageDataset, self).__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None
        self.hq_folder = opt['dataroot_hq']

        self.patch_size = opt['patch_size']
        self.sigma = opt['sigma']

        if self.io_backend_opt['type'] == 'lmdb':
            self.io_backend_opt['db_paths'] = [self.lq_folder]
            self.io_backend_opt['client_keys'] = ['lq']
            self.paths = paths_from_lmdb(self.lq_folder)
        elif 'meta_info_file' in self.opt:
            with open(self.opt['meta_info_file'], 'r') as fin:
                self.paths = [osp.join(self.lq_folder, line.rstrip().split(' ')[0]) for line in fin]
        else:
            self.paths = sorted(list(scandir(self.lq_folder, full_path=True)))

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        # load hq image
        hq_path = self.paths[index]
        img_bytes = self.file_client.get(hq_path, 'lq')
        img_hq = imfrombytes(img_bytes, float32=True)
        img_hq = get_patch(img_hq)

        # BGR to RGB, HWC to CHW, numpy to tensor
        img_hq = img2tensor(img_hq, bgr2rgb=True, float32=True)
        # normalize
        if self.mean is not None or self.std is not None:
            normalize(img_hq, self.mean, self.std, inplace=True)

        img_lq = img_hq.clone()
        noise_level = (
            torch.from_numpy([np.random.uniform(0, self.args.sigma)]) / 255.0
        )
        noise = torch.randn(img_hq.size()).mul_(noise_level).float()
        img_lq.add_(noise)

        noise_level = noise_level.unsqueeze(1).unsqueeze(1)

        return {'lq': img_lq, 'hq': img_hq, 'sigma': noise_level, 'lq_path': hq_path}

    def __len__(self):
        return len(self.paths)
