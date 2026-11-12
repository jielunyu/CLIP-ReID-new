# datasets/cu_market_bike_train.py
# encoding: utf-8

import random
from .bases import BaseImageDataset
from .market1501 import Market1501
from .dukemtmcreid import DukeMTMCreID
from .new_BikeReID import new_BikeReID
from .BikeReID import BikeReID

class CU_Market_Bike_Train(BaseImageDataset):
    """
    Unified protocol:

        train:
            Market1501(train) + BikeReID(train)

        query:
            DukeMTMC(query) + new_BikeReID(query)

        gallery:
            DukeMTMC(gallery) + new_BikeReID(gallery)

    Domain definition:

        0 = Market1501
        1 = CS-S / BikeReID

    Important:
        domain_id is stored separately from the original
        four-field ReID record:

        (img_path, pid, camid, trackid)

    Therefore Stage2 keeps its original data interface.
    """
    DOMAIN_MARKET = 0
    DOMAIN_BIKE = 1

    def __init__(self, root='', verbose=True, train_shuffle=True, train_shuffle_seed=42, **kwargs):
        super(CU_Market_Bike_Train, self).__init__()
        self.train_shuffle = train_shuffle
        self.train_shuffle_seed = train_shuffle_seed
        # -----------------------------------------------------
        # 1. Load original datasets
        # -----------------------------------------------------
        market = Market1501(root=root, verbose=verbose, pid_begin=0)
        duke = DukeMTMCreID(root=root, verbose=verbose, pid_begin=0)
        new_bike = new_BikeReID(root=root, verbose=verbose, pid_begin=0)
        # BikeReID uses data_path rather than root.
        bike_train_dataset = BikeReID(data_path=root, verbose=verbose, pid_begin=0)
        # -----------------------------------------------------
        # 2. Merge training data with explicit domain IDs
        # -----------------------------------------------------
        self.train, self.train_domain_ids = self._merge_train(
            market_train=market.train, bike_train=bike_train_dataset.train
        )
        self.num_train_domains = len(set(self.train_domain_ids))
        if len(self.train) != len(self.train_domain_ids):
            raise RuntimeError("train_domain_ids are not aligned with train records.")
        # -----------------------------------------------------
        # 3. Merge validation data
        # -----------------------------------------------------
        self.query, self.gallery = self._merge_eval(
            duke_query=duke.query, duke_gallery=duke.gallery, bike_query=new_bike.query, bike_gallery=new_bike.gallery
        )
        # -----------------------------------------------------
        # 4. Relabel training identities
        # -----------------------------------------------------
        self.train_pids = sorted(set(pid for _, pid, _, _ in self.train))
        self.pid2label = {pid: idx for idx, pid in enumerate(self.train_pids)}
        self.train = [(img_path, self.pid2label[pid], camid, trackid) for img_path, pid, camid, trackid in self.train]
        # IMPORTANT:
        # PID relabeling does not change the record order.
        # train_domain_ids therefore remains aligned.
        # -----------------------------------------------------
        # 5. Dataset statistics
        # -----------------------------------------------------
        (self.num_train_pids, self.num_train_imgs, self.num_train_cams, self.num_train_vids) = self.get_imagedata_info(
            self.train
        )
        (self.num_query_pids, self.num_query_imgs, self.num_query_cams, self.num_query_vids) = self.get_imagedata_info(
            self.query
        )
        (self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams, self.num_gallery_vids) = (
            self.get_imagedata_info(self.gallery)
        )
        if verbose:
            print("=> CU_Market_Bike_Train loaded")
            print("=> train = Market1501(train) + BikeReID(train)")
            print("=> query/gallery = DukeMTMC + new_BikeReID")
            print("=> train shuffle: {}, seed: {}".format(self.train_shuffle, self.train_shuffle_seed))
            self.print_dataset_statistics(self.train, self.query, self.gallery)
            print("=> num_train_pids: {}".format(self.num_train_pids))
            print("=> num_train_domains: {}".format(self.num_train_domains))
            domain_counts = {d: self.train_domain_ids.count(d) for d in sorted(set(self.train_domain_ids))}
            print("=> domain image counts: {}".format(domain_counts))
    # =========================================================
    # Merge training data
    # =========================================================

    def _merge_train(self, market_train, bike_train):
        """
        Return:

            records:
                [(img_path, pid, camid, trackid), ...]

            domain_ids:
                [0, 0, ..., 1, 1, ...]

        Domain IDs:

            0 = Market
            1 = BikeReID / CS-S

        The original records remain unchanged except
        for PID and camera ID offsets.
        """
        merged = [(record, self.DOMAIN_MARKET) for record in market_train]
        max_pid = max((pid for _, pid, _, _ in market_train), default=-1)
        max_cam = max((camid for _, _, camid, _ in market_train), default=-1)
        bike_train_shifted = self._shift_records(bike_train, pid_offset=max_pid + 1, cam_offset=max_cam + 1)
        merged.extend([(record, self.DOMAIN_BIKE) for record in bike_train_shifted])
        # Shuffle records and domain IDs together.
        if self.train_shuffle:
            rng = random.Random(self.train_shuffle_seed)
            rng.shuffle(merged)
        records = [record for record, _ in merged]
        domain_ids = [domain_id for _, domain_id in merged]
        return records, domain_ids
    # =========================================================
    # Merge evaluation data
    # =========================================================

    def _merge_eval(self, duke_query, duke_gallery, bike_query, bike_gallery):
        merged_query = list(duke_query)
        merged_gallery = list(duke_gallery)
        all_eval = merged_query + merged_gallery
        bike_pid_offset = self._next_pid_start(all_eval)
        bike_cam_offset = self._next_cam_start(all_eval)
        bike_query_shifted = self._shift_records(bike_query, pid_offset=bike_pid_offset, cam_offset=bike_cam_offset)
        bike_gallery_shifted = self._shift_records(bike_gallery, pid_offset=bike_pid_offset, cam_offset=bike_cam_offset)
        merged_query.extend(bike_query_shifted)
        merged_gallery.extend(bike_gallery_shifted)
        return merged_query, merged_gallery
    # =========================================================
    # Record shifting
    # =========================================================

    @staticmethod
    def _shift_records(records, pid_offset=0, cam_offset=0):
        shifted = []
        for img_path, pid, camid, trackid in records:
            new_pid = pid + pid_offset
            new_cam = camid + cam_offset
            assert new_pid >= 0
            assert new_cam >= 0
            shifted.append((img_path, new_pid, new_cam, trackid))
        return shifted

    @staticmethod
    def _next_pid_start(records):
        if len(records) == 0:
            return 0
        return max(pid for _, pid, _, _ in records) + 1

    @staticmethod
    def _next_cam_start(records):
        if len(records) == 0:
            return 0
        return max(camid for _, _, camid, _ in records) + 1
