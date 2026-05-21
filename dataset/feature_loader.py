'''Dataloader for fused point features.'''

import copy
import time
import logging
from glob import glob
from os.path import join
import torch
import numpy as np
import SharedArray as SA

logger = logging.getLogger("main-logger")

from dataset.point_loader import Point3DLoader

class FusedFeatureLoader(Point3DLoader):
    '''Dataloader for fused point features.'''

    def __init__(self,
                 datapath_prefix,
                 datapath_prefix_feat,
                 voxel_size=0.05,
                 split='train', aug=False, memcache_init=False,
                 identifier=7791, loop=1, eval_all=False,
                 input_color = False,
                 ):
        """Initialize the fused-feature dataset by extending Point3DLoader with
        per-scene feature paths and precomputed occurrence counts.

        Called once when building the training or evaluation DataLoader for the
        distillation pipeline, where each 3-D point cloud is paired with fused
        2-D-to-3-D feature vectors stored as ``.pt`` files.

        Steps:
            1. Call ``Point3DLoader.__init__`` to set up ``.pth`` paths,
               voxelizer, augmentations, and optional shared-memory caching.
            2. Store ``datapath_prefix_feat`` for locating feature files.
            3. For non-nuScenes datasets, glob ``<scene_name>_*.pt`` files for
               every scene to count how many feature variants (occurrences)
               exist (typically 5 for ScanNet/Matterport training, 1 for eval).
            4. Filter out scenes that have zero matching feature files (can
               happen with Matterport) so they are never sampled.

        Args:
            datapath_prefix (str): Root directory for raw 3-D point-cloud
                ``.pth`` files, forwarded to ``Point3DLoader``.
            datapath_prefix_feat (str): Directory containing fused feature
                ``.pt`` files named ``<scene_name>_<idx>.pt``.
            voxel_size (float): Voxel edge length in metres, forwarded to
                ``Point3DLoader``.
            split (str): Dataset split (``'train'``, ``'val'``, ``'test'``).
            aug (bool): Enable geometric and chromatic augmentation.
            memcache_init (bool): Cache raw points in POSIX shared memory.
            identifier (int): Unique ID for shared-memory segment names.
            loop (int): Epoch-length multiplier forwarded to ``Point3DLoader``.
            eval_all (bool): If True, return full-resolution labels and
                reconstruction indices for evaluation.
            input_color (bool): If True, use real point colours as input
                features; otherwise use all-ones.

        Returns:
            None
        """
        super().__init__(datapath_prefix=datapath_prefix, voxel_size=voxel_size,
                                           split=split, aug=aug, memcache_init=memcache_init,
                                           identifier=identifier, loop=loop,
                                           eval_all=eval_all, input_color=input_color)
        self.aug = aug
        self.input_color = input_color # decide whether we use point color values as input

        # prepare for 3D features
        self.datapath_feat = datapath_prefix_feat

        # Precompute the occurances for each scene
        # for training sets, ScanNet and Matterport has 5 each, nuscene 1
        # for evaluation/test sets, all has just one
        if 'nuscenes' in self.dataset_name: # only one file for each scene
            self.list_occur = None
        else:
            self.list_occur = []
            for data_path in self.data_paths:
                if 'scannet' in self.dataset_name:
                    scene_name = data_path[:-15].split('/')[-1]
                else:
                    scene_name = data_path[:-4].split('/')[-1]
                    scene_name = data_path[:-4].split('/')[-1]
                file_dirs = glob(join(self.datapath_feat, scene_name + '_*.pt'))
                self.list_occur.append(len(file_dirs))
            # some scenes in matterport have no features at all
            ind = np.where(np.array(self.list_occur) != 0)[0]
            if np.any(np.array(self.list_occur)==0):
                data_paths, list_occur = [], []
                for i in ind:
                    data_paths.append(self.data_paths[i])
                    list_occur.append(self.list_occur[i])
                self.data_paths = data_paths
                self.list_occur = list_occur

        if len(self.data_paths) == 0:
            raise Exception('0 file is loaded in the feature loader.')

    def __getitem__(self, index_long):
        """Load a 3-D point cloud together with its fused 2-D-to-3-D feature
        vectors, apply voxelization, and return aligned sparse-voxel tensors
        plus the per-voxel feature chunk and visibility mask.

        Called by the PyTorch DataLoader during distillation training or
        evaluation. A random feature-occurrence variant is selected for
        training; occurrence 0 is used when only one variant exists.

        Steps:
            1. Load (locs, feats, labels) from shared memory or disk.
            2. Derive the scene name and load the corresponding ``.pt`` feature
               file (randomly picking among available occurrences for training).
            3. Parse the feature dict: extract ``feat`` and ``mask_full``, and
               optionally ``mask`` for legacy formats. For val/test, expand
               compressed features back to full point-cloud size.
            4. Squeeze any trailing singleton dimension from ``feat_3d``.
            5. Apply optional elastic-distortion pre-voxel transform.
            6. Voxelize points. For training, build index mappings so that
               ``feat_3d`` contains only the features corresponding to the
               surviving voxels (handling both merged-mask and legacy formats).
               For eval, voxelize only the visible (masked) subset.
            7. Apply chromatic/flip augmentation if enabled.
            8. Prepend batch-index column to coordinates and convert to tensors.

        Args:
            index_long (int): Sample index from the DataLoader sampler; wrapped
                via modulo for ``loop > 1``.

        Returns:
            tuple: When ``eval_all`` is False:
                - coords (torch.IntTensor): ``(N_vox, 4)`` with columns
                  ``(batch_idx, x, y, z)``.
                - feats (torch.FloatTensor): ``(N_vox, 3)`` colour or ones.
                - labels (torch.LongTensor): ``(N_vox,)`` semantic labels.
                - feat_3d (torch.Tensor): ``(N_vox_visible, C)`` fused CLIP /
                  OpenSeg features for the visible voxels.
                - mask (torch.BoolTensor): ``(N_vox,)`` True where a voxel has
                  a valid fused feature.
            tuple: When ``eval_all`` is True the above plus:
                - inds_reconstruct (torch.LongTensor): ``(N_pts_visible,)``
                  mapping from original visible points to voxel indices.
        """
        index = index_long % len(self.data_paths)
        t_start = time.time()

        if self.use_shm:
            locs_in = SA.attach("shm://%s_%s_%06d_locs_%08d" % (
                self.dataset_name, self.split, self.identifier, index)).copy()
            feats_in = SA.attach("shm://%s_%s_%06d_feats_%08d" % (
                self.dataset_name, self.split, self.identifier, index)).copy()
            labels_in = SA.attach("shm://%s_%s_%06d_labels_%08d" % (
                self.dataset_name, self.split, self.identifier, index)).copy()
        else:
            locs_in, feats_in, labels_in = torch.load(self.data_paths[index])
            labels_in[labels_in == -100] = 255
            labels_in = labels_in.astype(np.uint8)
            if np.isscalar(feats_in) and feats_in == 0:
                # no color in the input point cloud, e.g nuscenes lidar
                feats_in = np.zeros_like(locs_in)
            else:
                feats_in = (feats_in + 1.) * 127.5

        t_load_3d = time.time()
        print(f"[TIMING] Load 3D .pth: {t_load_3d - t_start:.2f}s | "
              f"points={locs_in.shape[0]:,} | path={self.data_paths[index]}")

        # load 3D features
        if self.dataset_name == 'scannet_3d':
            scene_name = self.data_paths[index][:-15].split('/')[-1]
        else:
            scene_name = self.data_paths[index][:-4].split('/')[-1]

        if 'nuscenes' not in self.dataset_name:
            n_occur = self.list_occur[index]
            if n_occur > 1:
                nn_occur = np.random.randint(n_occur)
            elif n_occur == 1:
                nn_occur = 0
            else:
                raise NotImplementedError

            processed_data = torch.load(join(
                self.datapath_feat, scene_name+'_%d.pt'%(nn_occur)))
        else:
            # no repeated file
            processed_data = torch.load(join(self.datapath_feat, scene_name+'.pt'))

        t_load_feat = time.time()
        print(f"[TIMING] Load fused .pt: {t_load_feat - t_load_3d:.2f}s | "
              f"feat={processed_data['feat'].shape} mask={processed_data['mask_full'].shape}")

        flag_mask_merge = False
        if len(processed_data.keys())==2:
            flag_mask_merge = True
            feat_3d, mask_chunk = processed_data['feat'], processed_data['mask_full']
            if isinstance(mask_chunk, np.ndarray): # if the mask itself is a numpy array
                mask_chunk = torch.from_numpy(mask_chunk)
            mask = copy.deepcopy(mask_chunk)
            if self.split != 'train': # val or test set
                feat_3d_new = torch.zeros((locs_in.shape[0], feat_3d.shape[1]), dtype=feat_3d.dtype)
                feat_3d_new[mask] = feat_3d
                feat_3d = feat_3d_new
                mask_chunk = torch.ones_like(mask_chunk) # every point needs to be evaluted
                t_expand = time.time()
                print(f"[TIMING] Expand features: {t_expand - t_load_feat:.2f}s | "
                      f"feat_3d={feat_3d.shape} ({feat_3d.nelement()*feat_3d.element_size()/1e9:.2f} GB)")
        elif len(processed_data.keys())>2: # legacy, for old processed features
            feat_3d, mask_visible, mask_chunk = processed_data['feat'], processed_data['mask'], processed_data['mask_full']
            mask = torch.zeros(feat_3d.shape[0], dtype=torch.bool)
            mask[mask_visible] = True # mask out points without feature assigned

        if len(feat_3d.shape)>2:
            feat_3d = feat_3d[..., 0]

        locs = self.prevoxel_transforms(locs_in) if self.aug else locs_in

        t_pre_vox = time.time()
        print(f"[TIMING] Pre-voxelization prep: {t_pre_vox - t_load_feat:.2f}s (total so far: {t_pre_vox - t_start:.2f}s)")

        # calculate the corresponding point features after voxelization
        if self.split == 'train' and flag_mask_merge:
            locs, feats, labels, inds_reconstruct, vox_ind = self.voxelizer.voxelize(
                locs_in, feats_in, labels_in, return_ind=True)
            vox_ind = torch.from_numpy(vox_ind)
            mask = mask_chunk[vox_ind] # voxelized visible mask for entire point cloud
            mask_ind = mask_chunk.nonzero(as_tuple=False)[:, 0]
            index1 = - torch.ones(mask_chunk.shape[0], dtype=int)
            index1[mask_ind] = mask_ind

            index1 = index1[vox_ind]
            chunk_ind = index1[index1!=-1]

            index2 = torch.zeros(mask_chunk.shape[0])
            index2[mask_ind] = 1
            index3 = torch.cumsum(index2, dim=0, dtype=int)
            # get the indices of corresponding masked point features after voxelization
            indices = index3[chunk_ind] - 1

            # get the corresponding features after voxelization
            feat_3d = feat_3d[indices]
        elif self.split == 'train' and not flag_mask_merge: # legacy, for old processed features
            feat_3d = feat_3d[mask] # get features for visible points
            locs, feats, labels, inds_reconstruct, vox_ind = self.voxelizer.voxelize(
                locs_in, feats_in, labels_in, return_ind=True)
            mask_chunk[mask_chunk.clone()] = mask
            vox_ind = torch.from_numpy(vox_ind)
            mask = mask_chunk[vox_ind] # voxelized visible mask for entire point clouds
            mask_ind = mask_chunk.nonzero(as_tuple=False)[:, 0]
            index1 = - torch.ones(mask_chunk.shape[0], dtype=int)
            index1[mask_ind] = mask_ind

            index1 = index1[vox_ind]
            chunk_ind = index1[index1!=-1]

            index2 = torch.zeros(mask_chunk.shape[0])
            index2[mask_ind] = 1
            index3 = torch.cumsum(index2, dim=0, dtype=int)
            # get the indices of corresponding masked point features after voxelization
            indices = index3[chunk_ind] - 1

            # get the corresponding features after voxelization
            feat_3d = feat_3d[indices]
        else:
            locs, feats, labels, inds_reconstruct, vox_ind = self.voxelizer.voxelize(
                locs[mask_chunk], feats_in[mask_chunk], labels_in[mask_chunk], return_ind=True)
            vox_ind = torch.from_numpy(vox_ind)
            feat_3d = feat_3d[vox_ind]
            mask = mask[vox_ind]

        if self.eval_all: # during evaluation, no voxelization for GT labels
            labels = labels_in
        t_post_vox = time.time()
        print(f"[TIMING] Voxelization done: {t_post_vox - t_pre_vox:.2f}s | "
              f"voxels={locs.shape[0]:,} from {locs_in.shape[0]:,} points | "
              f"feat_3d={feat_3d.shape}")
        if self.aug:
            locs, feats, labels = self.input_transforms(locs, feats, labels)
        coords = torch.from_numpy(locs).int()
        coords = torch.cat((torch.ones(coords.shape[0], 1, dtype=torch.int), coords), dim=1)
        if self.input_color:
            feats = torch.from_numpy(feats).float() / 127.5 - 1.
        else:
            # hack: directly use color=(1, 1, 1) for all points
            feats = torch.ones(coords.shape[0], 3)
        labels = torch.from_numpy(labels).long()

        t_end = time.time()
        print(f"[TIMING] Total __getitem__: {t_end - t_start:.2f}s | "
              f"coords={coords.shape} feat_3d={feat_3d.shape} labels={labels.shape}")

        if self.eval_all:
            return coords, feats, labels, feat_3d, mask, torch.from_numpy(inds_reconstruct).long()
        return coords, feats, labels, feat_3d, mask

def collation_fn(batch):
    '''
    :param batch:
    :return:    coords: N x 4 (batch,x,y,z)
                feats:  N x 3
                labels: N
                colors: B x C x H x W x V
                labels_2d:  B x H x W x V
                links:  N x 4 x V (B,H,W,mask)

    '''
    coords, feats, labels, feat_3d, mask_chunk = list(zip(*batch))

    for i in range(len(coords)):
        coords[i][:, 0] *= i

    return torch.cat(coords), torch.cat(feats), torch.cat(labels), \
        torch.cat(feat_3d), torch.cat(mask_chunk)


def collation_fn_eval_all(batch):
    '''
    :param batch:
    :return:    coords: N x 4 (x,y,z,batch)
                feats:  N x 3
                labels: N
                colors: B x C x H x W x V
                labels_2d:  B x H x W x V
                links:  N x 4 x V (B,H,W,mask)
                inds_recons:ON

    '''
    coords, feats, labels, feat_3d, mask, inds_recons = list(zip(*batch))
    inds_recons = list(inds_recons)

    accmulate_points_num = 0
    for i in range(len(coords)):
        coords[i][:, 0] *= i
        inds_recons[i] = accmulate_points_num + inds_recons[i]
        accmulate_points_num += coords[i].shape[0]

    return torch.cat(coords), torch.cat(feats), torch.cat(labels), \
        torch.cat(feat_3d), torch.cat(mask), torch.cat(inds_recons)