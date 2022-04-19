from __future__ import division, print_function, absolute_import
import re
import glob
import os.path as osp
import warnings
import os
import numpy as np

from ..dataset import ImageDataset


class LPW(ImageDataset):
    """LPW.

    Reference:
        Labeled Pedestrian in the Wild 2018.

    URL: `<https://liuyu.us/dataset/lpw/index.html>`_
    
    Dataset statistics:
        - identities: 2700
        - images: 500K
    """
    dataset_dir = 'LPW'

    def __init__(self, root='', **kwargs):
        self.root = osp.abspath(osp.expanduser(root))
        self.dataset_dir = osp.join(self.root, self.dataset_dir)

        # allow alternative directory structure
        self.data_dir = self.dataset_dir
        data_dir = osp.join(self.data_dir)
        if osp.isdir(data_dir):
            self.data_dir = data_dir
        else:
            warnings.warn(
                'The current data structure is deprecated. Please '
                'put data folders such as "bounding_box_train" under '
                '"Market-1501-v15.09.15".'
            )

        self.train_dir_1 = osp.join(self.data_dir, 'scen1')
        self.train_dir_2 = osp.join(self.data_dir, 'scen2')
        self.train_dir_3 = osp.join(self.data_dir, 'scen3')


        required_files = [
            self.train_dir_1, self.train_dir_2, self.train_dir_3
        ]

        self.check_before_run(required_files)

        train_1, max_pid = self.process_dir(self.train_dir_1, relabel=True, pid_offset=0)
        train_2, max_pid = self.process_dir(self.train_dir_2, relabel=True, pid_offset=max_pid, camid_offset=3)
        train_3, _ = self.process_dir(self.train_dir_3, relabel=True, pid_offset=max_pid, camid_offset=7)
        train = train_1 + train_2 + train_3

        super(LPW, self).__init__(train=train, query=train, gallery=train, **kwargs)

    def process_dir(self, dir_path, relabel=False, pid_offset=0, camid_offset=0):
        
        img_paths = set()
        for dir_, _, files in os.walk(dir_path):
            for file_name in files:
                rel_dir = os.path.relpath(dir_, dir_path)
                rel_file = os.path.join(rel_dir, file_name)
                img_paths.add(rel_file)
        img_paths = list(img_paths)
        img_paths = [dir_path + '/' + s for s in img_paths]
        
        # here, img_paths contains all images in scen1 or scen2 or scen3

        # get pid's 1,3,15,... we later fix them to 0,1,2,..
        pid_org = []
        for img_path in img_paths:
            pid_org.append(int(re.findall('\d+', img_path)[3]))
        pid_org = np.asarray(pid_org)
        pid_org_unq = np.unique(pid_org)
        pid_fix_unq = np.arange(0, pid_org_unq.shape[0])
        pid_mapping = dict(zip(pid_org_unq, pid_fix_unq))

        data = []
        max_pid = pid_fix_unq[-1]+1+pid_offset     # track max pid so we can concatenate next scene pid accordingly
        pid_container = set()
        for img_path in img_paths:
            #pid = int(re.findall('\d+', img_path)[3]) + pid_offset
            pid = pid_mapping[int(re.findall('\d+', img_path)[3])] + pid_offset
            camid = int(re.findall('\d+', img_path)[2]) + camid_offset
            pid_container.add(pid)
            #data.append({'impth':img_path, 'pid':pid, 'camid':camid})
            data.append((img_path, pid, camid))
        return data, max_pid
