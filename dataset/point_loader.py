'''Dataloader for 3D points.'''

from glob import glob
import multiprocessing as mp
from os.path import join, exists
import numpy as np
import torch
import SharedArray as SA
import dataset.augmentation as t
from dataset.voxelizer import Voxelizer


def sa_create(name, var):
    '''Create share memory.'''

    shared_mem = SA.create(name, var.shape, dtype=var.dtype)
    shared_mem[...] = var[...]
    shared_mem.flags.writeable = False
    return shared_mem


def collation_fn(batch):
    '''
    :param batch:
    :return:   coords_batch: N x 4 (x,y,z,batch)

    '''
    coords, feats, labels = list(zip(*batch))

    for i, coord in enumerate(coords):
        coord[:, 0] *= i

    return torch.cat(coords), torch.cat(feats), torch.cat(labels)


def collation_fn_eval_all(batch):
    '''
    :param batch:
    :return:   coords_batch: N x 4 (x,y,z,batch)

    '''
    coords, feats, labels, inds_recons = list(zip(*batch))
    inds_recons = list(inds_recons)

    accmulate_points_num = 0
    for i, coord in enumerate(coords):
        coord[:, 0] *= i
        inds_recons[i] = accmulate_points_num + inds_recons[i]
        accmulate_points_num += coords[i].shape[0]

    return torch.cat(coords), torch.cat(feats), torch.cat(labels), torch.cat(inds_recons)


class Point3DLoader(torch.utils.data.Dataset):
    '''Dataloader for 3D points and labels.'''

    # Augmentation arguments
    SCALE_AUGMENTATION_BOUND = (0.9, 1.1)
    ROTATION_AUGMENTATION_BOUND = ((-np.pi / 64, np.pi / 64), (-np.pi / 64, np.pi / 64), (-np.pi,
                                                                                          np.pi))
    TRANSLATION_AUGMENTATION_RATIO_BOUND = ((-0.2, 0.2), (-0.2, 0.2), (0, 0))
    ELASTIC_DISTORT_PARAMS = ((0.2, 0.4), (0.8, 1.6))

    ROTATION_AXIS = 'z'
    LOCFEAT_IDX = 2

    def __init__(self, datapath_prefix='data', voxel_size=0.05,
                 split='train', aug=False, memcache_init=False, identifier=1233, loop=1,
                 data_aug_color_trans_ratio=0.1,
                 data_aug_color_jitter_std=0.05,
                 data_aug_hue_max=0.5,
                 data_aug_saturation_max=0.2,
                 eval_all=False, input_color=False
                 ):
        """Initialize the 3D point cloud dataset by discovering .pth files, configuring
        the voxelizer, setting up data-augmentation transforms, and optionally
        pre-loading all scenes into POSIX shared memory for faster training.

        Called once when the DataLoader is constructed in the training or evaluation
        script. Every subsequent ``__getitem__`` call relies on the state built here.

        Steps:
            1. Glob all ``*.pth`` files under ``<datapath_prefix>/<split>/``.
            2. Instantiate a ``Voxelizer`` with the given ``voxel_size`` and
               class-level augmentation bounds.
            3. If ``aug`` is True, compose elastic-distortion, flip, and chromatic
               augmentation transforms for training.
            4. If ``memcache_init`` is True and shared-memory segments do not yet
               exist, iterate over every ``.pth`` file with a multi-worker
               DataLoader and write (locs, feats, labels) into ``/dev/shm``.

        Args:
            datapath_prefix (str): Root directory containing the ``<split>/``
                sub-folder with ``.pth`` point-cloud files.
            voxel_size (float): Edge length (in metres) of each voxel used for
                sparse-voxel down-sampling.
            split (str | None): Dataset split name (``'train'``, ``'val'``,
                ``'test'``), used as a sub-directory and shared-memory key.
            aug (bool): Whether to enable geometric and chromatic augmentation.
            memcache_init (bool): Whether to cache all point clouds in POSIX
                shared memory (``/dev/shm``) for zero-copy reads.
            identifier (int): Unique integer appended to shared-memory segment
                names so that multiple runs do not collide.
            loop (int): Multiplier applied to ``__len__`` so the dataloader
                iterates over the dataset ``loop`` times per epoch.
            data_aug_color_trans_ratio (float): Maximum ratio for chromatic
                translation augmentation.
            data_aug_color_jitter_std (float): Standard deviation for chromatic
                jitter augmentation.
            data_aug_hue_max (float): Maximum hue shift for HSV augmentation.
            data_aug_saturation_max (float): Maximum saturation shift for HSV
                augmentation.
            eval_all (bool): If True, ``__getitem__`` returns the *original*
                (un-voxelized) labels together with reconstruction indices so
                that predictions can be mapped back to every input point.
            input_color (bool): If True, normalized point colours are returned
                as features; otherwise all-ones features are used.

        Returns:
            None
        """
        super().__init__()
        self.split = split
        if split is None:
            split = ''
        self.identifier = identifier
        self.data_paths = sorted(glob(join(datapath_prefix, split, '*.pth')))
        if len(self.data_paths) == 0:
            raise Exception('0 file is loaded in the point loader.')

        self.input_color = input_color
        self.voxel_size = voxel_size
        self.aug = aug
        self.loop = loop
        self.eval_all = eval_all
        dataset_name = datapath_prefix.split('/')[-1]
        self.dataset_name = dataset_name
        self.use_shm = memcache_init

        self.voxelizer = Voxelizer(
            voxel_size=voxel_size,
            clip_bound=None,
            use_augmentation=True,
            scale_augmentation_bound=self.SCALE_AUGMENTATION_BOUND,
            rotation_augmentation_bound=self.ROTATION_AUGMENTATION_BOUND,
            translation_augmentation_ratio_bound=self.TRANSLATION_AUGMENTATION_RATIO_BOUND)

        if aug:
            prevoxel_transform_train = [
                t.ElasticDistortion(self.ELASTIC_DISTORT_PARAMS)]
            self.prevoxel_transforms = t.Compose(prevoxel_transform_train)
            input_transforms = [
                t.RandomHorizontalFlip(self.ROTATION_AXIS, is_temporal=False),
                t.ChromaticAutoContrast(),
                t.ChromaticTranslation(data_aug_color_trans_ratio),
                t.ChromaticJitter(data_aug_color_jitter_std),
                t.HueSaturationTranslation(
                    data_aug_hue_max, data_aug_saturation_max),
            ]
            self.input_transforms = t.Compose(input_transforms)

        if memcache_init and (not exists("/dev/shm/%s_%s_%06d_locs_%08d" % (dataset_name, split, identifier, 0))):
            print('[*] Starting shared memory init ...')
            print('No. CPUs: ', mp.cpu_count())
            for i, (locs, feats, labels) in enumerate(torch.utils.data.DataLoader(
                    self.data_paths, collate_fn=lambda x: torch.load(x[0]),
                    num_workers=min(16, mp.cpu_count()), shuffle=False)):
                labels[labels == -100] = 255
                labels = labels.astype(np.uint8)
                # no color in the input point cloud, e.g nuscenes
                if np.isscalar(feats) and feats == 0:
                    feats = np.zeros_like(locs)
                # Scale color to 0-255
                feats = (feats + 1.) * 127.5
                sa_create("shm://%s_%s_%06d_locs_%08d" %
                          (dataset_name, split, identifier, i), locs)
                sa_create("shm://%s_%s_%06d_feats_%08d" %
                          (dataset_name, split, identifier, i), feats)
                sa_create("shm://%s_%s_%06d_labels_%08d" %
                          (dataset_name, split, identifier, i), labels)
            print('[*] %s (%s) loading 3D points done (%d)! ' %
                  (datapath_prefix, split, len(self.data_paths)))

    def __getitem__(self, index_long):
        """Load a single point-cloud scene, apply augmentation and voxelization,
        and return sparse-voxel tensors ready for the 3-D backbone.

        Called by the PyTorch DataLoader on every iteration. The ``index_long``
        may exceed the number of files when ``self.loop > 1``; a modulo maps it
        back to a valid file index.

        Steps:
            1. Load (locs, feats, labels) from shared memory or disk (``.pth``).
            2. Remap ignore labels (``-100 -> 255``) and rescale colours to
               ``[0, 255]``.
            3. Optionally apply elastic-distortion pre-voxel transform.
            4. Voxelize with ``self.voxelizer`` to obtain sparse coordinates and
               the reconstruction index array.
            5. Optionally apply chromatic / flip augmentations.
            6. Prepend a batch-index column of ones to ``coords``.
            7. Return colour features or all-ones depending on ``input_color``.

        Args:
            index_long (int): Sample index supplied by the DataLoader sampler.
                Wrapped via modulo to handle ``loop > 1``.

        Returns:
            tuple: When ``eval_all`` is False:
                - coords (torch.IntTensor): Voxel coordinates, shape
                  ``(N_vox, 4)`` where columns are ``(batch_idx, x, y, z)``.
                - feats (torch.FloatTensor): Per-voxel colour features, shape
                  ``(N_vox, 3)``.
                - labels (torch.LongTensor): Semantic labels, shape
                  ``(N_vox,)``.
            tuple: When ``eval_all`` is True the above plus:
                - inds_reconstruct (torch.LongTensor): Mapping from each
                  original point to its voxel, shape ``(N_pts,)``.
        """
        index = index_long % len(self.data_paths)
        if self.use_shm:
            locs_in = SA.attach("shm://%s_%s_%06d_locs_%08d" %
                                (self.dataset_name, self.split, self.identifier, index)).copy()
            feats_in = SA.attach("shm://%s_%s_%06d_feats_%08d" %
                                 (self.dataset_name, self.split, self.identifier, index)).copy()
            labels_in = SA.attach("shm://%s_%s_%06d_labels_%08d" %
                                  (self.dataset_name, self.split, self.identifier, index)).copy()
        else:
            locs_in, feats_in, labels_in = torch.load(self.data_paths[index])
            labels_in[labels_in == -100] = 255
            labels_in = labels_in.astype(np.uint8)
            # no color in the input point cloud, e.g nuscenes
            if np.isscalar(feats_in) and feats_in == 0:
                feats_in = np.zeros_like(locs_in)
            feats_in = (feats_in + 1.) * 127.5

        locs = self.prevoxel_transforms(locs_in) if self.aug else locs_in
        locs, feats, labels, inds_reconstruct = self.voxelizer.voxelize(
            locs, feats_in, labels_in)
        if self.eval_all:
            labels = labels_in
        if self.aug:
            locs, feats, labels = self.input_transforms(locs, feats, labels)
        coords = torch.from_numpy(locs).int()
        coords = torch.cat(
            (torch.ones(coords.shape[0], 1, dtype=torch.int), coords), dim=1)
        if self.input_color:
            feats = torch.from_numpy(feats).float() / 127.5 - 1.
        else:
            feats = torch.ones(coords.shape[0], 3)
        labels = torch.from_numpy(labels).long()

        if self.eval_all:
            return coords, feats, labels, torch.from_numpy(inds_reconstruct).long()
        return coords, feats, labels

    def __len__(self):
        """Return the effective dataset length, accounting for the loop multiplier.

        Called by the PyTorch DataLoader to determine how many samples constitute
        one epoch. Multiplying by ``self.loop`` lets the trainer see every scene
        multiple times per epoch without external wrappers.

        Steps:
            1. Compute ``len(self.data_paths) * self.loop``.

        Args:
            None

        Returns:
            int: Total number of indexable samples (files times loop factor).
        """
        return len(self.data_paths) * self.loop
