# datasets/cu_market_bike_train.py
# encoding: utf-8
import random
from .bases import BaseImageDataset
from .market1501 import Market1501
from .dukemtmcreid import DukeMTMCreID
from .new_BikeReID import new_BikeReID

# 注意：这里根据你项目里的真实文件名调整
# 如果你的文件叫 Bike_ReID.py，类名叫 Bike_ReID，就用下面这一行
from .BikeReID import BikeReID
# 如果你的文件叫 BikeReID.py，类名叫 BikeReID，则改成：
# from .BikeReID import BikeReID


class CU_Market_Bike_Train(BaseImageDataset):
    """
    统一协议：
      train   = Market1501(train) + Bike_ReID(train)
      query   = DukeMTMC-reID(query) + new_BikeReID(query)
      gallery = DukeMTMC-reID(gallery) + new_BikeReID(gallery)

    说明：
      1. 训练集使用 Market + Bike_ReID。
      2. 验证集保持原逻辑不变，仍然是 Duke + new_BikeReID。
      3. 训练集 PID 会自动重新映射成连续 label，避免 CLIP-ReID 分类越界。
      4. 合并训练集时会对 Bike_ReBikeReIDID 的 pid/camid 做安全偏移，避免和 Market 冲突。
    """

    def __init__(self, root='', verbose=True, train_shuffle=True, train_shuffle_seed=42, **kwargs):
        super(CU_Market_Bike_Train, self).__init__()

        self.train_shuffle = train_shuffle
        self.train_shuffle_seed = train_shuffle_seed

        # ---------------------------------------------------------
        # 1. 加载基础数据集
        # ---------------------------------------------------------
        market = Market1501(root=root, verbose=True, pid_begin=0)
        duke = DukeMTMCreID(root=root, verbose=True, pid_begin=0)

        # 验证集仍然使用 new_BikeReID
        new_bike = new_BikeReID(root=root, verbose=True, pid_begin=0)

        # 训练集改为 BikeReID
        bike_train_dataset = BikeReID(root=root, verbose=True, pid_begin=0)
        # 如果你的类名是 BikeReID，则改成：
        # bike_train_dataset = BikeReID(root=root, verbose=True, pid_begin=0)

        # ---------------------------------------------------------
        # 2. 合并训练集：Market train + Bike_ReID train
        # ---------------------------------------------------------
        self.train = self._merge_train(
            market_train=market.train,
            bike_train=bike_train_dataset.train
        )

        # ---------------------------------------------------------
        # 3. 合并验证集：Duke query/gallery + new_BikeReID query/gallery
        # ---------------------------------------------------------
        self.query, self.gallery = self._merge_eval(
            duke_query=duke.query,
            duke_gallery=duke.gallery,
            bike_query=new_bike.query,
            bike_gallery=new_bike.gallery
        )

        # ---------------------------------------------------------
        # 4. 自动生成 pid -> 连续 label 映射
        # ---------------------------------------------------------
        self.train_pids = sorted(list(set(pid for _, pid, _, _ in self.train)))
        self.pid2label = {pid: idx for idx, pid in enumerate(self.train_pids)}

        # 将 train dataset 的 pid 替换成连续 label
        self.train = [
            (img_path, self.pid2label[pid], camid, trackid)
            for img_path, pid, camid, trackid in self.train
        ]

        if verbose:
            print("=> CU_Market_Bike_Train loaded")
            print("=> train = Market1501(train) + BikeReID(train)")
            print("=> query/gallery = DukeMTMC-reID + new_BikeReID")
            print(f"=> train shuffle: {self.train_shuffle}, seed: {self.train_shuffle_seed}")
            self.print_dataset_statistics(self.train, self.query, self.gallery)
            print(f"=> num_train_pids: {len(self.train_pids)}")

        # ---------------------------------------------------------
        # 5. 数据集统计信息
        # ---------------------------------------------------------
        self.num_train_pids, self.num_train_imgs, self.num_train_cams, self.num_train_vids = \
            self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams, self.num_query_vids = \
            self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams, self.num_gallery_vids = \
            self.get_imagedata_info(self.gallery)

    def _merge_train(self, market_train, bike_train):
        """
        训练集合并：
          Market1501(train) + Bike_ReID(train)

        对 Bike_ReID 的 pid 和 camid 做安全偏移，避免和 Market 冲突。
        """
        merged_train = list(market_train)

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
        """
        验证集合并，保持原逻辑：
          query   = DukeMTMC-reID(query) + new_BikeReID(query)
          gallery = DukeMTMC-reID(gallery) + new_BikeReID(gallery)
        """
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
        """
        对数据记录做 pid/camid 偏移。

        输入格式：
            (img_path, pid, camid, trackid)

        输出格式：
            (img_path, new_pid, new_camid, trackid)
        """
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
        """返回下一个安全 PID 起点。"""
        if len(records) == 0:
            return 0
        max_pid = max(pid for _, pid, _, _ in records)
        return max_pid + 1

    @staticmethod
    def _next_cam_start(records):
        """返回下一个安全 CAMID 起点。"""
        if len(records) == 0:
            return 0
        max_cam = max(camid for _, _, camid, _ in records)
        return max_cam + 1