import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader
from timm.data.random_erasing import RandomErasing
from .bases import ImageDataset, ImageDatasetWithSubgroup, ImageDatasetWithDomain
from .sampler import RandomIdentitySampler
from .sampler_ddp import RandomIdentitySampler_DDP
import torch.distributed as dist
from .dukemtmcreid import DukeMTMCreID
from .market1501 import Market1501
from .msmt17 import MSMT17
from .occ_duke import OCC_DukeMTMCreID
from .vehicleid import VehicleID
from .veri import VeRi
from .new_BikeReID import new_BikeReID
from .cu_market_bike_train import CU_Market_Bike_Train
from .pwc import PWC
from .BikeReID import BikeReID

__factory = {
    'market1501': Market1501,
    'dukemtmc': DukeMTMCreID,
    'msmt17': MSMT17,
    'occ_duke': OCC_DukeMTMCreID,
    'veri': VeRi,
    'VehicleID': VehicleID,
    'new_BikeReID': new_BikeReID,
    'BikeReID': BikeReID,
    'cu_market_bike_train': CU_Market_Bike_Train,
    'pwc': PWC,
}

# ============================================================
# Original Stage2 training collate function
# ============================================================

def train_collate_fn(batch):
    imgs, pids, camids, viewids, _ = zip(*batch)
    pids = torch.tensor(pids, dtype=torch.int64)
    viewids = torch.tensor(viewids, dtype=torch.int64)
    camids = torch.tensor(camids, dtype=torch.int64)
    return (torch.stack(imgs, dim=0), pids, camids, viewids)
# ============================================================
# Stage1 POPD collate function
# ============================================================

def popd_collate_fn(batch):
    imgs, pids, camids, viewids, domainids, _ = zip(*batch)
    return (
        torch.stack(imgs, dim=0),
        torch.tensor(pids, dtype=torch.int64),
        torch.tensor(camids, dtype=torch.int64),
        torch.tensor(viewids, dtype=torch.int64),
        torch.tensor(domainids, dtype=torch.int64),
    )
# ============================================================
# Evaluation collate function
# ============================================================

def val_collate_fn(batch):
    imgs, pids, camids, viewids, img_paths = zip(*batch)
    viewids = torch.tensor(viewids, dtype=torch.int64)
    camids_batch = torch.tensor(camids, dtype=torch.int64)
    return (torch.stack(imgs, dim=0), pids, camids, camids_batch, viewids, img_paths)
# ============================================================
# Main dataloader
# ============================================================

def make_dataloader(cfg):
    # --------------------------------------------------------
    # 1. Transforms
    # --------------------------------------------------------
    train_transforms = T.Compose(
        [
            T.Resize(cfg.INPUT.SIZE_TRAIN, interpolation=3),
            T.RandomHorizontalFlip(p=cfg.INPUT.PROB),
            T.Pad(cfg.INPUT.PADDING),
            T.RandomCrop(cfg.INPUT.SIZE_TRAIN),
            T.ToTensor(),
            T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD),
            RandomErasing(probability=cfg.INPUT.RE_PROB, mode='pixel', max_count=1, device='cpu'),
        ]
    )
    val_transforms = T.Compose(
        [T.Resize(cfg.INPUT.SIZE_TEST), T.ToTensor(), T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)]
    )
    num_workers = cfg.DATALOADER.NUM_WORKERS
    # --------------------------------------------------------
    # 2. Dataset initialization
    # --------------------------------------------------------
    dataset = __factory[cfg.DATASETS.NAMES](root=cfg.DATASETS.ROOT_DIR)
    num_classes = dataset.num_train_pids
    cam_num = dataset.num_train_cams
    view_num = dataset.num_train_vids
    # --------------------------------------------------------
    # 3. Explicit domain labels for POPD Stage1
    # --------------------------------------------------------
    domain_ids = getattr(dataset, 'train_domain_ids', None)
    if domain_ids is None:
        raise RuntimeError(
            "POPD Stage1 requires explicit " "dataset-level train_domain_ids. " "Please use CU_Market_Bike_Train."
        )
    if len(domain_ids) != len(dataset.train):
        raise RuntimeError("train_domain_ids must align " "one-to-one with dataset.train.")
    expected_domains = int(cfg.DATASETS.NUM_DOMAINS)
    observed_domains = sorted(set(int(x) for x in domain_ids))
    if observed_domains != list(range(expected_domains)):
        raise RuntimeError("Expected domain IDs 0..{}; " "observed {}.".format(expected_domains - 1, observed_domains))
    print("POPD dataset domains: {}".format(observed_domains))
    # --------------------------------------------------------
    # 4. Stage2 original training dataset
    # --------------------------------------------------------
    train_set = ImageDataset(dataset.train, train_transforms)
    # --------------------------------------------------------
    # 5. Stage1 POPD dataset
    #
    # The original four-field ReID record is preserved.
    # domain_id is appended separately.
    #
    # We intentionally retain deterministic transforms
    # because Stage1 caches image features only once.
    # --------------------------------------------------------
    train_set_stage1 = ImageDatasetWithDomain(dataset.train, domain_ids, val_transforms)
    # --------------------------------------------------------
    # 6. Stage2 original sampler
    # --------------------------------------------------------
    if 'triplet' in cfg.DATALOADER.SAMPLER:
        if cfg.MODEL.DIST_TRAIN:
            print('DIST_TRAIN START')
            mini_batch_size = cfg.SOLVER.STAGE2.IMS_PER_BATCH // dist.get_world_size()
            data_sampler = RandomIdentitySampler_DDP(
                dataset.train, cfg.SOLVER.STAGE2.IMS_PER_BATCH, cfg.DATALOADER.NUM_INSTANCE
            )
            batch_sampler = torch.utils.data.sampler.BatchSampler(data_sampler, mini_batch_size, True)
            train_loader_stage2 = DataLoader(
                train_set,
                num_workers=num_workers,
                batch_sampler=batch_sampler,
                collate_fn=train_collate_fn,
                pin_memory=True,
            )
        else:
            train_loader_stage2 = DataLoader(
                train_set,
                batch_size=cfg.SOLVER.STAGE2.IMS_PER_BATCH,
                sampler=RandomIdentitySampler(
                    dataset.train, cfg.SOLVER.STAGE2.IMS_PER_BATCH, cfg.DATALOADER.NUM_INSTANCE
                ),
                num_workers=num_workers,
                collate_fn=train_collate_fn,
            )
    elif cfg.DATALOADER.SAMPLER == 'softmax':
        print('using softmax sampler')
        train_loader_stage2 = DataLoader(
            train_set,
            batch_size=cfg.SOLVER.STAGE2.IMS_PER_BATCH,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=train_collate_fn,
        )
    else:
        raise ValueError("Unsupported sampler: {}".format(cfg.DATALOADER.SAMPLER))
    # --------------------------------------------------------
    # 7. Validation dataset
    # --------------------------------------------------------
    val_set = ImageDataset(dataset.query + dataset.gallery, val_transforms)
    val_loader = DataLoader(
        val_set, batch_size=cfg.TEST.IMS_PER_BATCH, shuffle=False, num_workers=num_workers, collate_fn=val_collate_fn
    )
    # --------------------------------------------------------
    # 8. Stage1 loader
    #
    # No fixed balance sampler.
    # The original dynamic no-replacement batching algorithm
    # is retained in processor_clipreid_stage1.py.
    # --------------------------------------------------------
    train_loader_stage1 = DataLoader(
        train_set_stage1,
        batch_size=cfg.SOLVER.STAGE1.IMS_PER_BATCH,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=popd_collate_fn,
    )
    return (train_loader_stage2, train_loader_stage1, val_loader, len(dataset.query), num_classes, cam_num, view_num)
