from PIL import Image, ImageFile

from torch.utils.data import Dataset
import os.path as osp
import random
import torch
ImageFile.LOAD_TRUNCATED_IMAGES = True


def read_image(img_path):
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process."""
    got_img = False
    if not osp.exists(img_path):
        raise IOError("{} does not exist".format(img_path))
    while not got_img:
        try:
            img = Image.open(img_path).convert('RGB')
            got_img = True
        except IOError:
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
            pass
    return img


class BaseDataset(object):
    """
    Base class of reid dataset
    """

    def get_imagedata_info(self, data):
        pids, cams, tracks = [], [], []
        for _, pid, camid, trackid in data:
            pids += [pid]
            cams += [camid]
            tracks += [trackid]
        pids = set(pids)
        cams = set(cams)
        tracks = set(tracks)
        num_pids = len(pids)
        num_cams = len(cams)
        num_imgs = len(data)
        num_views = len(tracks)
        return num_pids, num_imgs, num_cams, num_views

    def print_dataset_statistics(self):
        raise NotImplementedError


class BaseImageDataset(BaseDataset):
    """
    Base class of image reid dataset
    """

    def print_dataset_statistics(self, train, query, gallery):
        num_train_pids, num_train_imgs, num_train_cams, num_train_views = self.get_imagedata_info(train)
        num_query_pids, num_query_imgs, num_query_cams, num_train_views = self.get_imagedata_info(query)
        num_gallery_pids, num_gallery_imgs, num_gallery_cams, num_train_views = self.get_imagedata_info(gallery)

        print("Dataset statistics:")
        print("  ----------------------------------------")
        print("  subset   | # ids | # images | # cameras")
        print("  ----------------------------------------")
        print("  train    | {:5d} | {:8d} | {:9d}".format(num_train_pids, num_train_imgs, num_train_cams))
        print("  query    | {:5d} | {:8d} | {:9d}".format(num_query_pids, num_query_imgs, num_query_cams))
        print("  gallery  | {:5d} | {:8d} | {:9d}".format(num_gallery_pids, num_gallery_imgs, num_gallery_cams))
        print("  ----------------------------------------")


# =========================================================================
# 原始的 Dataset (用于 Query 和 Gallery，不干扰测试期逻辑)
# =========================================================================
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

        return img, pid, camid, trackid, img_path.split('/')[-1]


# =========================================================================
# 新增的 State-Aware & Subgroup Dataset (专用于 Stage 2 训练)
# =========================================================================
class ImageDatasetWithSubgroup(Dataset):
    def __init__(self, dataset, subgroup_json_path, transform=None):
        """
        :param dataset: 原始的训练数据集 list: [(img_path, pid, camid, trackid), ...]
        :param subgroup_json_path: 离线跑出来的 subgroup_map.json 的路径
        :param transform: 数据增强
        """
        self.dataset = dataset
        self.transform = transform

        if not os.path.exists(subgroup_json_path):
            raise FileNotFoundError(f"未能找到 subgroup 标签文件: {subgroup_json_path}")
            
        with open(subgroup_json_path, "r", encoding="utf-8") as f:
            self.subgroup_map = json.load(f)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        # 1. 解析基础信息
        img_path, pid, camid, trackid = self.dataset[index]
        img = read_image(img_path)

        if self.transform is not None:
            img = self.transform(img)

        # 2. 匹配 State 和 Subgroup 标签
        abs_path = os.path.abspath(img_path)
        
        # 安全获取，防止由于路径斜杠问题（Windows/Linux差异）导致的 KeyError
        if abs_path in self.subgroup_map:
            sg_info = self.subgroup_map[abs_path]
            state = sg_info["state"]
            subgroup_local = sg_info["subgroup_local"]
            subgroup_global = sg_info["subgroup_global"]
        else:
            # 防御性回退：如果某张图片没在 json 里，给默认状态，避免训练崩溃
            state, subgroup_local, subgroup_global = 0, 0, 0

        # 3. 按照 Processor 所需的格式一并返回
        return img, pid, camid, trackid, state, subgroup_local, subgroup_global