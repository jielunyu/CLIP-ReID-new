
# # encoding: utf-8

# import glob
# import os
# import os.path as osp
# import random

# from datasets.bases import BaseImageDataset


# class new_BikeReID(BaseImageDataset):
#     dataset_dir = "new_BikeReID/new_BikeReID"

#     def __init__(self, root='', verbose=True, pid_begin=0, sample_seed=42, **kwargs):
#         super(new_BikeReID, self).__init__()

#         self.data_dir = osp.join(root, self.dataset_dir)
#         self.train_dir = osp.join(self.data_dir, 'train')
#         self.query_dir = osp.join(self.data_dir, 'test', 'query')
#         self.gallery_dir = osp.join(self.data_dir, 'test', 'gallery')

#         self._check_before_run()
#         self.pid_begin = pid_begin
#         self.sample_seed = sample_seed

#         train = self._process_dir(self.train_dir, relabel=True, sample_four_per_pid=True)
#         query = self._process_dir(self.query_dir, relabel=False, sample_four_per_pid=False)
#         gallery = self._process_dir(self.gallery_dir, relabel=False, sample_four_per_pid=False)

#         if verbose:
#             print("=> new_BikeReID loaded")
#             print(f"=> train mode: random 2 images per pid (seed={self.sample_seed})")
#             self.print_dataset_statistics(train, query, gallery)

#         self.train = train
#         self.query = query
#         self.gallery = gallery

#         self.num_train_pids, self.num_train_imgs, self.num_train_cams, self.num_train_vids = self.get_imagedata_info(self.train)
#         self.num_query_pids, self.num_query_imgs, self.num_query_cams, self.num_query_vids = self.get_imagedata_info(self.query)
#         self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams, self.num_gallery_vids = self.get_imagedata_info(self.gallery)

#     def _check_before_run(self):
#         if not osp.exists(self.data_dir):
#             raise RuntimeError("'{}' is not available".format(self.data_dir))
#         if not osp.exists(self.train_dir):
#             raise RuntimeError("'{}' is not available".format(self.train_dir))
#         if not osp.exists(self.query_dir):
#             raise RuntimeError("'{}' is not available".format(self.query_dir))
#         if not osp.exists(self.gallery_dir):
#             raise RuntimeError("'{}' is not available".format(self.gallery_dir))

#     def _process_dir(self, dir_path, relabel=False, sample_four_per_pid=False):
#         rng = random.Random(self.sample_seed)

#         img_paths = []
#         subdirs = sorted([d for d in os.listdir(dir_path) if osp.isdir(osp.join(dir_path, d))])

#         for subdir in subdirs:
#             cur_imgs = sorted(glob.glob(osp.join(dir_path, subdir, '*.jpg')))
#             if len(cur_imgs) == 0:
#                 continue

#             if sample_four_per_pid:
#                 if len(cur_imgs) <= 8:
#                     img_paths.extend(cur_imgs)
#                 else:
#                     img_paths.extend(rng.sample(cur_imgs, 8))
#             else:
#                 img_paths.extend(cur_imgs)

#         pid_container = set()
#         for img_path in sorted(img_paths):
#             pid = int(osp.basename(osp.dirname(img_path)))
#             if pid == -1:
#                 continue
#             pid_container.add(pid)

#         pid2label = {pid: label for label, pid in enumerate(sorted(pid_container))}

#         dataset = []
#         for img_path in sorted(img_paths):
#             pid = int(osp.basename(osp.dirname(img_path)))
#             if pid == -1:
#                 continue

#             file_name = osp.basename(img_path)
#             parts = file_name.split('_')
#             if len(parts) < 2:
#                 raise ValueError("文件名格式不正确，无法解析 camid: {}".format(file_name))

#             camid = int(parts[1])
#             camid -= 1

#             if relabel:
#                 pid = pid2label[pid]

#             dataset.append((img_path, self.pid_begin + pid, camid, 1))

#         return dataset
# encoding: utf-8

"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

import glob
import re
import os
import os.path as osp

from datasets.bases import BaseImageDataset
# from .bases import BaseImageDataset
from collections import defaultdict
import pickle
class new_BikeReID(BaseImageDataset):
    """
    Dataset statistics:
    # identities: 1501 (+1 for background)
    # images: 7816 (train) + 830 (query) + 830 (gallery)
    """

    dataset_dir = "new_BikeReID/new_BikeReID"
    def __init__(self, root='', verbose=True, pid_begin=0, **kwargs):
        super(new_BikeReID, self).__init__()
        self.data_dir = osp.join(root, self.dataset_dir)
        self.train_dir = osp.join(self.data_dir, 'train')
        self.query_dir = osp.join(self.data_dir, 'test', 'query')
        self.gallery_dir = osp.join(self.data_dir, 'test', 'gallery')

        self._check_before_run()    # 检查: 所有目录是否存在
        self.pid_begin = pid_begin
        
        train = self._process_dir(self.train_dir, relabel=True)
        query = self._process_dir(self.query_dir, relabel=False)
        gallery = self._process_dir(self.gallery_dir, relabel=False)

        if verbose:
            print("=> new_BikeReID loaded")
            self.print_dataset_statistics(train, query, gallery)

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids, self.num_train_imgs, self.num_train_cams, self.num_train_vids = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams, self.num_query_vids = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams, self.num_gallery_vids = self.get_imagedata_info(self.gallery)

    def _check_before_run(self):
        """
        Check if all files are available before going deeper
        """
        if not osp.exists(self.data_dir):
            raise RuntimeError("'{}' is not available".format(self.data_dir))
        if not osp.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))
        if not osp.exists(self.query_dir):
            raise RuntimeError("'{}' is not available".format(self.query_dir))
        if not osp.exists(self.gallery_dir):
            raise RuntimeError("'{}' is not available".format(self.gallery_dir))

    def _process_dir(self, dir, relabel=False):
        """
        获取dir_path子目录下的所有图片
        dir_path
            |-- folder
                |-- *.jpg
                |-- ..
            |-- ..
        """
        img_paths = []
        for subdir in os.listdir(dir):
            # print(type(glob.glob(osp.join(dir_path, dir, '*.jpg'))))
            for i in glob.glob(osp.join(dir, subdir, '*.jpg')):
                img_paths.append(i)
        # img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        pattern = re.compile(r'([-\d]+)_c(\d)')

        pid_container = set()
        for img_path in sorted(img_paths):
            pid = int(img_path.split('/')[-2])
            if pid == -1: continue  # junk images are just ignored
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}
        dataset = []
        for img_path in sorted(img_paths):
            # pid, camid = map(int, pattern.search(img_path).groups())
            pid = int(img_path.split('/')[-2])
            camid = int(img_path.split('/')[-1].split('_')[1])
            if pid == -1: continue  # junk images are just ignored
            assert 0 <= pid <= 10000  # pid == 0 means background
            assert 1 <= camid <= 2
            camid -= 1  # index starts from 0
            if relabel: pid = pid2label[pid]

            dataset.append((img_path, self.pid_begin + pid, camid, 1))
        return dataset


# if __name__ == '__main__':
#     dataset_path = '/mnt/TransReID-main/data/'
#     BikeReID(dataset_path)