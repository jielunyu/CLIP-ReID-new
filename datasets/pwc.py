# encoding: utf-8

import glob
import re
import os.path as osp

from .bases import BaseImageDataset

class PWC(BaseImageDataset):
    """
    PWC ReID Dataset
    
    Dataset statistics:
    # identities: 58
    # images: depends on your extraction (query + gallery)
    # cameras: 3 (cam1, cam2, cam3)
    # status: 0 (walk), 1 (bike)
    """
    dataset_dir = 'PWC_val' # 你的总文件夹名称

    def __init__(self, root='', verbose=True, pid_begin=0, **kwargs):
        super(PWC, self).__init__()
        
        # 定义数据集的根目录
        self.dataset_dir = osp.join(root, self.dataset_dir)
        
        # 定义子文件夹路径
        self.train_dir = osp.join(self.dataset_dir, 'test') # 预留，当前可能不存在
        self.query_dir = osp.join(self.dataset_dir, 'query')
        self.gallery_dir = osp.join(self.dataset_dir, 'gallery')

        self._check_before_run()
        self.pid_begin = pid_begin

        # 如果存在 train 文件夹则处理，不存在则置为空列表 (适应你目前只做测试的需求)
        if osp.exists(self.train_dir):
            train = self._process_dir(self.train_dir, relabel=True)
        else:
            train = []

        query = self._process_dir(self.query_dir, relabel=False)
        gallery = self._process_dir(self.gallery_dir, relabel=False)

        if verbose:
            print("=> PWC Dataset loaded")
            self.print_dataset_statistics(train, query, gallery)

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids, self.num_train_imgs, self.num_train_cams, self.num_train_vids = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams, self.num_query_vids = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams, self.num_gallery_vids = self.get_imagedata_info(self.gallery)

    def _check_before_run(self):
        """检查测试必须的 query 和 gallery 文件夹是否存在"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.query_dir):
            raise RuntimeError("'{}' is not available".format(self.query_dir))
        if not osp.exists(self.gallery_dir):
            raise RuntimeError("'{}' is not available".format(self.gallery_dir))

    def _process_dir(self, dir_path, relabel=False):
        # 获取所有图片，支持 jpg, png 等格式
        img_paths = glob.glob(osp.join(dir_path, '*.*'))
        
        # ================= 核心修改区域 =================
        # 我们的命名格式: 001-cam1-01-0.jpg
        # 正则解析: 提取 身份ID(组1) - 摄像机ID(组2) - 序列号(略) - 状态标签(组3)
        pattern = re.compile(r'(\d+)-cam(\d+)-\d+-(\d+)')
        # ================================================

        pid_container = set()
        for img_path in sorted(img_paths):
            match = pattern.search(img_path)
            if not match: 
                continue # 忽略命名不规范的脏图片
            pid = int(match.group(1))
            pid_container.add(pid)
            
        # 重新映射 ID (针对训练集，将 001-058 映射为 0-57)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}
        
        dataset = []
        for img_path in sorted(img_paths):
            match = pattern.search(img_path)
            if not match:
                continue
                
            pid = int(match.group(1))
            camid = int(match.group(2))
            status = int(match.group(3)) # 0: 走路, 1: 骑行

            # 在 ReID 框架中，相机 ID 通常要求从 0 开始算
            camid -= 1 

            if relabel: 
                pid = pid2label[pid]

            # 将你的 status (走路/骑行) 作为第四个参数传入
            # 框架原本这里存的是视频序列的 trackid，我们存 status 非常巧妙
            # 以后你可以修改评测代码，专门测“用走路的图去检索骑车的图”
            dataset.append((img_path, self.pid_begin + pid, camid, status))
            
        return dataset