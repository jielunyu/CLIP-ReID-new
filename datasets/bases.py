import os
import json
import os.path as osp
import random
import torch
from PIL import Image, ImageFile
from torch.utils.data import Dataset

ImageFile.LOAD_TRUNCATED_IMAGES = True

def read_image(img_path):
    """Keep reading image until succeed."""
    if not osp.exists(img_path):
        raise IOError("{} does not exist".format(img_path))
    while True:
        try:
            return Image.open(img_path).convert('RGB')
        except IOError:
            print("IOError incurred when reading '{}'. " "Will retry.".format(img_path))

class BaseDataset(object):

    def get_imagedata_info(self, data):
        pids = []
        cams = []
        tracks = []
        for _, pid, camid, trackid in data:
            pids.append(pid)
            cams.append(camid)
            tracks.append(trackid)
        num_pids = len(set(pids))
        num_cams = len(set(cams))
        num_imgs = len(data)
        num_views = len(set(tracks))
        return (num_pids, num_imgs, num_cams, num_views)

    def print_dataset_statistics(self):
        raise NotImplementedError

class BaseImageDataset(BaseDataset):

    def print_dataset_statistics(self, train, query, gallery):
        train_info = self.get_imagedata_info(train)
        query_info = self.get_imagedata_info(query)
        gallery_info = self.get_imagedata_info(gallery)
        print("Dataset statistics:")
        print("----------------------------------------")
        print("subset   | # ids | # images | # cameras")
        print("----------------------------------------")
        print("train    | {:5d} | {:8d} | {:9d}".format(train_info[0], train_info[1], train_info[2]))
        print("query    | {:5d} | {:8d} | {:9d}".format(query_info[0], query_info[1], query_info[2]))
        print("gallery  | {:5d} | {:8d} | {:9d}".format(gallery_info[0], gallery_info[1], gallery_info[2]))
        print("----------------------------------------")
# ============================================================
# Original Dataset
# Used by Stage2 and evaluation.
# Keep its interface unchanged.
# ============================================================

class ImageDataset(Dataset):

    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_path, pid, camid, trackid = self.dataset[index]
        img = read_image(img_path)
        if self.transform is not None:
            img = self.transform(img)
        return (img, pid, camid, trackid, osp.basename(img_path))
# ============================================================
# Stage1-only Dataset with explicit dataset-domain labels
# ============================================================

class ImageDatasetWithDomain(Dataset):

    def __init__(self, dataset, domain_ids, transform=None):
        if len(dataset) != len(domain_ids):
            raise ValueError("dataset and domain_ids must have " "the same length.")
        self.dataset = dataset
        self.domain_ids = [int(x) for x in domain_ids]
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_path, pid, camid, trackid = self.dataset[index]
        domain_id = self.domain_ids[index]
        img = read_image(img_path)
        if self.transform is not None:
            img = self.transform(img)
        return (img, pid, camid, trackid, domain_id, osp.basename(img_path))
# ============================================================
# Original State-Aware & Subgroup Dataset
# ============================================================

class ImageDatasetWithSubgroup(Dataset):

    def __init__(self, dataset, subgroup_json_path, transform=None):
        self.dataset = dataset
        self.transform = transform
        if not osp.exists(subgroup_json_path):
            raise FileNotFoundError("Cannot find subgroup file: {}".format(subgroup_json_path))
        with open(subgroup_json_path, "r", encoding="utf-8") as f:
            self.subgroup_map = json.load(f)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_path, pid, camid, trackid = self.dataset[index]
        img = read_image(img_path)
        if self.transform is not None:
            img = self.transform(img)
        abs_path = os.path.abspath(img_path)
        if abs_path in self.subgroup_map:
            sg_info = self.subgroup_map[abs_path]
            state = sg_info["state"]
            subgroup_local = sg_info["subgroup_local"]
            subgroup_global = sg_info["subgroup_global"]
        else:
            state = 0
            subgroup_local = 0
            subgroup_global = 0
        return (img, pid, camid, trackid, state, subgroup_local, subgroup_global)
