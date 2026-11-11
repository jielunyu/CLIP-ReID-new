# # datasets/cu_market_bike_train.py
# # encoding: utf-8

# from .bases import BaseImageDataset
# from .market1501 import Market1501
# from .dukemtmcreid import DukeMTMCreID
# from .new_BikeReID import new_BikeReID


# class CU_Market_Bike_Train(BaseImageDataset):
#     """
#     统一协议：
#       train   = DukeMTMC-reID(train) + new_BikeReID(train)
#       query   = Market1501(query) + new_BikeReID(query)
#       gallery = Market1501(gallery) + new_BikeReID(gallery)
#     """

#     def __init__(self, root='', verbose=True, **kwargs):
#         super(CU_Market_Bike_Train, self).__init__()

#         market = Market1501(root=root, verbose=False, pid_begin=0)
#         duke = DukeMTMCreID(root=root, verbose=False, pid_begin=0)
#         bike = new_BikeReID(root=root, verbose=False, pid_begin=0)

#         # train = Duke train + Bike train
#         self.train = self._merge_train(
#             duke_train=duke.train,
#             bike_train=bike.train,
#             duke_pid_count=duke.num_train_pids,
#             duke_cam_count=duke.num_train_cams
#         )

#         # query/gallery = Market query/gallery + Bike query/gallery
#         self.query, self.gallery = self._merge_eval(
#             market_query=market.query,
#             market_gallery=market.gallery,
#             bike_query=bike.query,
#             bike_gallery=bike.gallery
#         )

#         if verbose:
#             print("=> CU_Market_Bike_Train loaded")
#             print("   train   = DukeMTMC-reID(train) + new_BikeReID(train)")
#             print("   query   = Market1501(query) + new_BikeReID(query)")
#             print("   gallery = Market1501(gallery) + new_BikeReID(gallery)")
#             self.print_dataset_statistics(self.train, self.query, self.gallery)

#         self.num_train_pids, self.num_train_imgs, self.num_train_cams, self.num_train_vids = \
#             self.get_imagedata_info(self.train)
#         self.num_query_pids, self.num_query_imgs, self.num_query_cams, self.num_query_vids = \
#             self.get_imagedata_info(self.query)
#         self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams, self.num_gallery_vids = \
#             self.get_imagedata_info(self.gallery)

#     def _merge_train(self, duke_train, bike_train, duke_pid_count, duke_cam_count):
#         """
#         训练集：
#         - Duke pid 不动
#         - new_Bike pid 偏移到 Duke 后面
#         - new_Bike camid 再偏移，避免和 Duke 相机号冲突
#         """
#         merged_train = list(duke_train)

#         bike_train_shifted = self._shift_records(
#             bike_train,
#             pid_offset=duke_pid_count,
#             cam_offset=duke_cam_count
#         )

#         merged_train.extend(bike_train_shifted)

#         return merged_train

#     def _merge_eval(self, market_query, market_gallery, bike_query, bike_gallery):
#         """
#         测试集：
#         - Market query/gallery 保持不变
#         - new_Bike query/gallery 使用同样的 pid 偏移和 camid 偏移
#         - 注意：Bike 的 query 和 gallery 必须使用相同的 pid_offset，
#           否则同一个 Bike 身份在 query/gallery 中会被错开，导致评估错误
#         """
#         merged_query = list(market_query)
#         merged_gallery = list(market_gallery)

#         market_all = list(market_query) + list(market_gallery)

#         bike_pid_offset = self._next_pid_start(market_all)
#         bike_cam_offset = self._next_cam_start(market_all)

#         bike_query_shifted = self._shift_records(
#             bike_query,
#             pid_offset=bike_pid_offset,
#             cam_offset=bike_cam_offset
#         )

#         bike_gallery_shifted = self._shift_records(
#             bike_gallery,
#             pid_offset=bike_pid_offset,
#             cam_offset=bike_cam_offset
#         )

#         merged_query.extend(bike_query_shifted)
#         merged_gallery.extend(bike_gallery_shifted)

#         return merged_query, merged_gallery

#     @staticmethod
#     def _shift_records(records, pid_offset=0, cam_offset=0):
#         shifted = []

#         for img_path, pid, camid, trackid in records:
#             shifted.append(
#                 (
#                     img_path,
#                     pid + pid_offset,
#                     camid + cam_offset,
#                     trackid
#                 )
#             )

#         return shifted

#     @staticmethod
#     def _next_pid_start(records):
#         if len(records) == 0:
#             return 0

#         max_pid = max(pid for _, pid, _, _ in records)
#         return max_pid + 1

#     @staticmethod
#     def _next_cam_start(records):
#         if len(records) == 0:
#             return 0

#         max_cam = max(camid for _, _, camid, _ in records)
#         return max_cam + 1

# # datasets/cu_market_bike_train.py
# # encoding: utf-8

# from .bases import BaseImageDataset
# from .market1501 import Market1501
# from .dukemtmcreid import DukeMTMCreID
# from .new_BikeReID import new_BikeReID


# class CU_Market_Bike_Train(BaseImageDataset):
#     """
#     统一协议：
#       train   = Market1501(train) + new_BikeReID(train)
#       query   = DukeMTMC-reID(query) + new_BikeReID(query)
#       gallery = DukeMTMC-reID(gallery) + new_BikeReID(gallery)
#     """

#     def __init__(self, root='', verbose=True, **kwargs):
#         super(CU_Market_Bike_Train, self).__init__()

#         market = Market1501(root=root, verbose=False, pid_begin=0)
#         duke = DukeMTMCreID(root=root, verbose=False, pid_begin=0)
#         bike = new_BikeReID(root=root, verbose=False, pid_begin=0)

#         self.train = self._merge_train(
#             market_train=market.train,
#             bike_train=bike.train,
#             market_pid_count=market.num_train_pids,
#             market_cam_count=market.num_train_cams
#         )

#         self.query, self.gallery = self._merge_eval(
#             duke_query=duke.query,
#             duke_gallery=duke.gallery,
#             bike_query=bike.query,
#             bike_gallery=bike.gallery
#         )

#         if verbose:
#             print("=> CU_Market_Bike_Train loaded")
#             self.print_dataset_statistics(self.train, self.query, self.gallery)

#         self.num_train_pids, self.num_train_imgs, self.num_train_cams, self.num_train_vids = \
#             self.get_imagedata_info(self.train)
#         self.num_query_pids, self.num_query_imgs, self.num_query_cams, self.num_query_vids = \
#             self.get_imagedata_info(self.query)
#         self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams, self.num_gallery_vids = \
#             self.get_imagedata_info(self.gallery)

#     def _merge_train(self, market_train, bike_train, market_pid_count, market_cam_count):
#         """
#         训练集：
#         - Market pid 不动
#         - new_Bike pid 偏移到 Market 后面
#         - new_Bike camid 再偏移，避免和 Market 相机号冲突
#         """
#         merged_train = list(market_train)
#         bike_train_shifted = self._shift_records(
#             bike_train,
#             pid_offset=market_pid_count,
#             cam_offset=market_cam_count
#         )
#         merged_train.extend(bike_train_shifted)
#         return merged_train

#     def _merge_eval(self, duke_query, duke_gallery, bike_query, bike_gallery):
#         """
#         测试集：
#         - Duke query/gallery 保持不变
#         - new_Bike query/gallery 统一做同样的 pid 偏移和 camid 偏移
#         """
#         merged_query = list(duke_query)
#         merged_gallery = list(duke_gallery)

#         duke_all = list(duke_query) + list(duke_gallery)
#         bike_pid_offset = self._next_pid_start(duke_all)
#         bike_cam_offset = self._next_cam_start(duke_all)

#         bike_query_shifted = self._shift_records(
#             bike_query,
#             pid_offset=bike_pid_offset,
#             cam_offset=bike_cam_offset
#         )
#         bike_gallery_shifted = self._shift_records(
#             bike_gallery,
#             pid_offset=bike_pid_offset,
#             cam_offset=bike_cam_offset
#         )

#         merged_query.extend(bike_query_shifted)
#         merged_gallery.extend(bike_gallery_shifted)

#         return merged_query, merged_gallery

#     @staticmethod
#     def _shift_records(records, pid_offset=0, cam_offset=0):
#         shifted = []
#         for img_path, pid, camid, trackid in records:
#             shifted.append((img_path, pid + pid_offset, camid + cam_offset, trackid))
#         return shifted

#     @staticmethod
#     def _next_pid_start(records):
#         if len(records) == 0:
#             return 0
#         max_pid = max(pid for _, pid, _, _ in records)
#         return max_pid + 1

#     @staticmethod
#     def _next_cam_start(records):
#         if len(records) == 0:
#             return 0
#         max_cam = max(camid for _, _, camid, _ in records)
#         return max_cam + 1
    
    
#     # datasets/cu_market_bike_train.py
# # encoding: utf-8

# import random

# from .bases import BaseImageDataset
# from .market1501 import Market1501
# from .dukemtmcreid import DukeMTMCreID
# from .new_BikeReID import new_BikeReID


# class CU_Market_Bike_Train(BaseImageDataset):
#     """
#     统一协议：
#       train   = Market1501(train) + new_BikeReID(train)
#       query   = DukeMTMC-reID(query) + new_BikeReID(query)
#       gallery = DukeMTMC-reID(gallery) + new_BikeReID(gallery)
#     """

#     def __init__(self, root='', verbose=True, train_shuffle=True, train_shuffle_seed=42, **kwargs):
#         super(CU_Market_Bike_Train, self).__init__()

#         self.train_shuffle = train_shuffle
#         self.train_shuffle_seed = train_shuffle_seed

#         market = Market1501(root=root, verbose=False, pid_begin=0)
#         duke = DukeMTMCreID(root=root, verbose=False, pid_begin=0)
#         bike = new_BikeReID(root=root, verbose=False, pid_begin=0)

#         self.train = self._merge_train(
#             market_train=market.train,
#             bike_train=bike.train,
#             market_pid_count=market.num_train_pids,
#             market_cam_count=market.num_train_cams
#         )

#         self.query, self.gallery = self._merge_eval(
#             duke_query=duke.query,
#             duke_gallery=duke.gallery,
#             bike_query=bike.query,
#             bike_gallery=bike.gallery
#         )

#         if verbose:
#             print("=> CU_Market_Bike_Train loaded")
#             print(f"=> train shuffle: {self.train_shuffle}, seed: {self.train_shuffle_seed}")
#             self.print_dataset_statistics(self.train, self.query, self.gallery)

#         self.num_train_pids, self.num_train_imgs, self.num_train_cams, self.num_train_vids = \
#             self.get_imagedata_info(self.train)
#         self.num_query_pids, self.num_query_imgs, self.num_query_cams, self.num_query_vids = \
#             self.get_imagedata_info(self.query)
#         self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams, self.num_gallery_vids = \
#             self.get_imagedata_info(self.gallery)

#     def _merge_train(self, market_train, bike_train, market_pid_count, market_cam_count):
#         """
#         训练集：
#         - Market pid 不动
#         - new_Bike pid 偏移到 Market 后面
#         - new_Bike camid 再偏移，避免和 Market 相机号冲突
#         - 合并后随机打乱，提升 walking / cycling 混合概率
#         """
#         merged_train = list(market_train)

#         bike_train_shifted = self._shift_records(
#             bike_train,
#             pid_offset=market_pid_count,
#             cam_offset=market_cam_count
#         )
#         merged_train.extend(bike_train_shifted)

#         if self.train_shuffle:
#             rng = random.Random(self.train_shuffle_seed)
#             rng.shuffle(merged_train)

#         return merged_train

#     def _merge_eval(self, duke_query, duke_gallery, bike_query, bike_gallery):
#         """
#         测试集：
#         - Duke query/gallery 保持不变
#         - new_Bike query/gallery 统一做同样的 pid 偏移和 camid 偏移
#         """
#         merged_query = list(duke_query)
#         merged_gallery = list(duke_gallery)

#         duke_all = list(duke_query) + list(duke_gallery)
#         bike_pid_offset = self._next_pid_start(duke_all)
#         bike_cam_offset = self._next_cam_start(duke_all)

#         bike_query_shifted = self._shift_records(
#             bike_query,
#             pid_offset=bike_pid_offset,
#             cam_offset=bike_cam_offset
#         )
#         bike_gallery_shifted = self._shift_records(
#             bike_gallery,
#             pid_offset=bike_pid_offset,
#             cam_offset=bike_cam_offset
#         )

#         merged_query.extend(bike_query_shifted)
#         merged_gallery.extend(bike_gallery_shifted)

#         return merged_query, merged_gallery

#     @staticmethod
#     def _shift_records(records, pid_offset=0, cam_offset=0):
#         shifted = []
#         for img_path, pid, camid, trackid in records:
#             shifted.append((img_path, pid + pid_offset, camid + cam_offset, trackid))
#         return shifted

#     @staticmethod
#     def _next_pid_start(records):
#         if len(records) == 0:
#             return 0
#         max_pid = max(pid for _, pid, _, _ in records)
#         return max_pid + 1

#     @staticmethod
#     def _next_cam_start(records):
#         if len(records) == 0:
#             return 0
#         max_cam = max(camid for _, _, camid, _ in records)
#         return max_cam + 1

# datasets/cu_market_bike_train.py
# encoding: utf-8
import random
from .bases import BaseImageDataset
from .market1501 import Market1501
from .dukemtmcreid import DukeMTMCreID
from .new_BikeReID import new_BikeReID


class CU_Market_Bike_Train(BaseImageDataset):
    """
    统一协议：
      train   = Market1501(train) + new_BikeReID(train, random 800 pids)
      query   = DukeMTMC-reID(query) + new_BikeReID(query)
      gallery = DukeMTMC-reID(gallery) + new_BikeReID(gallery)
    """

    def __init__(self, root='', verbose=True, train_shuffle=True, train_shuffle_seed=42, **kwargs):
        super(CU_Market_Bike_Train, self).__init__()

        self.train_shuffle = train_shuffle
        self.train_shuffle_seed = train_shuffle_seed

        # 加载基础数据集
        market = Market1501(root=root, verbose=True, pid_begin=0)
        duke = DukeMTMCreID(root=root, verbose=True, pid_begin=0)
        bike = new_BikeReID(root=root, verbose=True, pid_begin=0)

        # 随机选取 new_Bike 的 800 个 PID
        bike_train = self._sample_bike_train(bike.train, num_pid=800)

        # 合并训练集
        self.train = self._merge_train(
            market_train=market.train,
            bike_train=bike_train
        )

        # 合并测试集
        self.query, self.gallery = self._merge_eval(
            duke_query=duke.query,
            duke_gallery=duke.gallery,
            bike_query=bike.query,
            bike_gallery=bike.gallery
        )

        # 自动生成 pid -> 连续 label 映射
        self.train_pids = sorted(list(set(pid for _, pid, _, _ in self.train)))
        self.pid2label = {pid: idx for idx, pid in enumerate(self.train_pids)}

        # 将 train dataset 的 pid 替换成 label
        self.train = [(img_path, self.pid2label[pid], camid, trackid)
                      for img_path, pid, camid, trackid in self.train]

        if verbose:
            print("=> CU_Market_Bike_Train loaded")
            print(f"=> train shuffle: {self.train_shuffle}, seed: {self.train_shuffle_seed}")
            self.print_dataset_statistics(self.train, self.query, self.gallery)
            print(f"=> num_train_pids: {len(self.train_pids)}")

        # 数据集信息
        self.num_train_pids, self.num_train_imgs, self.num_train_cams, self.num_train_vids = \
            self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams, self.num_query_vids = \
            self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams, self.num_gallery_vids = \
            self.get_imagedata_info(self.gallery)

    def _sample_bike_train(self, bike_train, num_pid=800):
        """从 new_BikeReID 的 train 中随机选择指定数量的 PID"""
        all_pids = list(set(pid for _, pid, _, _ in bike_train))
        if len(all_pids) <= num_pid:
            return bike_train

        rng = random.Random(self.train_shuffle_seed)
        selected_pids = rng.sample(all_pids, num_pid)
        sampled_train = [record for record in bike_train if record[1] in selected_pids]
        return sampled_train

    def _merge_train(self, market_train, bike_train):
        """训练集合并，安全偏移 PID/CAMID"""
        merged_train = list(market_train)

        # 安全偏移基准：训练集已有最大 pid/camid
        max_pid = max((pid for _, pid, _, _ in merged_train), default=-1)
        max_cam = max((camid for _, _, camid, _ in merged_train), default=-1)

        bike_train_shifted = self._shift_records(
            bike_train,
            pid_offset=max_pid + 1,
            cam_offset=max_cam + 1
        )
        merged_train.extend(bike_train_shifted)

        if self.train_shuffle:
            rng = random.Random(self.train_shuffle_seed)
            rng.shuffle(merged_train)

        return merged_train

    def _merge_eval(self, duke_query, duke_gallery, bike_query, bike_gallery):
        """测试集合并，安全偏移 PID/CAMID"""
        merged_query = list(duke_query)
        merged_gallery = list(duke_gallery)

        all_eval = merged_query + merged_gallery

        bike_pid_offset = self._next_pid_start(all_eval)
        bike_cam_offset = self._next_cam_start(all_eval)

        bike_query_shifted = self._shift_records(
            bike_query,
            pid_offset=bike_pid_offset,
            cam_offset=bike_cam_offset
        )
        bike_gallery_shifted = self._shift_records(
            bike_gallery,
            pid_offset=bike_pid_offset,
            cam_offset=bike_cam_offset
        )

        merged_query.extend(bike_query_shifted)
        merged_gallery.extend(bike_gallery_shifted)

        return merged_query, merged_gallery

    @staticmethod
    def _shift_records(records, pid_offset=0, cam_offset=0):
        """偏移记录，同时检查安全性"""
        shifted = []
        for img_path, pid, camid, trackid in records:
            new_pid = pid + pid_offset
            new_cam = camid + cam_offset
            assert new_pid >= 0, f"pid 越界: {new_pid}"
            assert new_cam >= 0, f"camid 越界: {new_cam}"
            shifted.append((img_path, new_pid, new_cam, trackid))
        return shifted

    @staticmethod
    def _next_pid_start(records):
        """返回下一个安全 PID"""
        if len(records) == 0:
            return 0
        max_pid = max(pid for _, pid, _, _ in records)
        return max_pid + 1

    @staticmethod
    def _next_cam_start(records):
        """返回下一个安全 CAMID"""
        if len(records) == 0:
            return 0
        max_cam = max(camid for _, _, camid, _ in records)
        return max_cam + 1