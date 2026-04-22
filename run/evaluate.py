import os
import random
import time
import numpy as np
import logging
import argparse
import urllib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from PIL import Image as PILImage

import torch
import torch.backends.cudnn as cudnn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.multiprocessing as mp
import torch.distributed as dist
from util import metric
from torch.utils import model_zoo

from MinkowskiEngine import SparseTensor
from util import config
from util.util import export_pointcloud, get_palette, \
    convert_labels_with_palette, extract_text_feature, visualize_labels
from tqdm import tqdm
from run.distill import get_model

from dataset.label_constants import *


def get_parser():
    '''Parse the config file.'''

    parser = argparse.ArgumentParser(description='OpenScene evaluation')
    parser.add_argument('--config', type=str,
                    default='config/scannet/eval_openseg.yaml',
                    help='config file')
    parser.add_argument('opts',
                    default=None,
                    help='see config/scannet/test_ours_openseg.yaml for all options',
                    nargs=argparse.REMAINDER)
    args = parser.parse_args()
    assert args.config is not None
    cfg = config.load_cfg_from_cfg_file(args.config)
    if args.opts is not None:
        cfg = config.merge_cfg_from_list(cfg, args.opts)
    return cfg


def get_logger():
    '''Define logger.'''

    logger_name = "main-logger"
    logger_in = logging.getLogger(logger_name)
    logger_in.setLevel(logging.DEBUG)
    handler = logging.StreamHandler()
    fmt = "[%(asctime)s %(filename)s line %(lineno)d] %(message)s"
    handler.setFormatter(logging.Formatter(fmt))
    logger_in.addHandler(handler)
    return logger_in

def is_url(url):
    scheme = urllib.parse.urlparse(url).scheme
    return scheme in ('http', 'https')

def main_process():
    return not args.multiprocessing_distributed or (
            args.multiprocessing_distributed and args.rank % args.ngpus_per_node == 0)

def precompute_text_related_properties(labelset_name):
    '''pre-compute text features, labelset, palette, and mapper.'''

    if 'scannet' in labelset_name:
        labelset = list(SCANNET_LABELS_20)
        labelset[-1] = 'other' # change 'other furniture' to 'other'
        palette = get_palette(colormap='scannet')
    elif labelset_name == 'matterport_3d' or labelset_name == 'matterport':
        labelset = list(MATTERPORT_LABELS_21)
        palette = get_palette(colormap='matterport')
    elif 'matterport_3d_40' in labelset_name or labelset_name == 'matterport40':
        labelset = list(MATTERPORT_LABELS_40)
        palette = get_palette(colormap='matterport_160')
    elif 'matterport_3d_80' in labelset_name or labelset_name == 'matterport80':
        labelset = list(MATTERPORT_LABELS_80)
        palette = get_palette(colormap='matterport_160')
    elif 'matterport_3d_160' in labelset_name or labelset_name == 'matterport160':
        labelset = list(MATTERPORT_LABELS_160)
        palette = get_palette(colormap='matterport_160')
    elif 'nuscenes' in labelset_name:
        labelset = list(NUSCENES_LABELS_16)
        palette = get_palette(colormap='nuscenes16')
    else: # an arbitrary dataset, just use a large labelset
        labelset = list(MATTERPORT_LABELS_160)
        palette = get_palette(colormap='matterport_160')

    mapper = None
    if hasattr(args, 'map_nuscenes_details'):
        labelset = list(NUSCENES_LABELS_DETAILS)
        mapper = torch.tensor(MAPPING_NUSCENES_DETAILS, dtype=int)
        # Append mapping for 'unlabeled' index (used by mark_no_feature_to_unknown)
        # Maps to class 16 (one past the last valid 0-15 class)
        mapper = torch.cat([mapper, torch.tensor([len(NUSCENES_LABELS_16)], dtype=int)])

    text_features = extract_text_feature(labelset, args)
    # labelset.append('unknown')
    labelset.append('unlabeled')
    return text_features, labelset, mapper, palette


def visualize_camera_projections(pcl, point_colors, scene_name, data_root_2d,
                                   split, save_folder, suffix, title):
    """
    Project labeled 3D points onto 6 camera images and save a 2x3 mosaic PNG.

    Physical ring layout:
        Row 0: front_left | front | front_right
        Row 1: back_left  | back  | back_right

    Args:
        pcl          : (N,3) float array - world-frame 3D coordinates
        point_colors : (N,3) uint8 array - RGB label colours
        scene_name   : str, e.g. 'scene-0061'
        data_root_2d : str, e.g. 'data/nuscenes_2d'
        split        : str, 'train' or 'val'
        save_folder  : str, output directory
        suffix       : str, 'gt' / 'fusion' / 'distill' / 'ensemble'
        title        : str, figure subtitle
    """
    import matplotlib.pyplot as plt

    cam_layout = [
        ['front_left', 'front',  'front_right'],
        ['back_left',  'back',   'back_right'],
    ]

    scene_2d_dir = os.path.join(data_root_2d, split, scene_name)
    if not os.path.isdir(scene_2d_dir):
        return

    # Ensure plain numpy float32
    if hasattr(pcl, 'numpy'):
        pcl = pcl.numpy()
    pcl = np.asarray(pcl, dtype=np.float32)
    point_colors = np.asarray(point_colors, dtype=np.float32)

    pts_hom = np.concatenate(
        [pcl, np.ones((len(pcl), 1), dtype=np.float32)], axis=1
    ).T  # (4, N)

    fig, axes = plt.subplots(2, 3, figsize=(21, 9))
    fig.suptitle(f'{scene_name}  -  {title}', fontsize=13, fontweight='bold')

    for row_idx, cam_row in enumerate(cam_layout):
        for col_idx, cam in enumerate(cam_row):
            ax = axes[row_idx][col_idx]

            img_path = os.path.join(scene_2d_dir, 'color', f'{cam}.jpg')
            k_path   = os.path.join(scene_2d_dir, 'K',     f'{cam}.npy')
            p_path   = os.path.join(scene_2d_dir, 'pose',  f'{cam}.npy')

            if not all(os.path.exists(p) for p in [img_path, k_path, p_path]):
                ax.text(0.5, 0.5, f'{cam}\n(missing)', ha='center', va='center',
                        transform=ax.transAxes, fontsize=9)
                ax.axis('off')
                continue

            img  = plt.imread(img_path)
            K    = np.load(k_path).astype(np.float64)
            c2w  = np.load(p_path).astype(np.float64)
            H, W = img.shape[:2]

            # World -> camera frame (pose is cam-to-world, invert to get world-to-cam)
            w2c     = np.linalg.inv(c2w)                        # (4,4)
            pts_cam = (w2c @ pts_hom.astype(np.float64))[:3]    # (3,N)

            in_front = pts_cam[2] > 0.1
            pts_f    = pts_cam[:, in_front]
            col_f    = point_colors[in_front]

            if pts_f.shape[1] == 0:
                ax.imshow(img)
                ax.set_title(cam, fontsize=9)
                ax.axis('off')
                continue

            # Project to pixel coordinates
            pts_px = K @ pts_f        # (3,N)
            u = pts_px[0] / pts_px[2]
            v = pts_px[1] / pts_px[2]

            in_img = (u >= 0) & (u < W) & (v >= 0) & (v < H)
            u_px   = u[in_img].astype(int)
            v_px   = v[in_img].astype(int)
            colors = col_f[in_img] / 255.0

            ax.imshow(img)
            ax.scatter(u_px, v_px, c=colors, s=1, alpha=0.6, linewidths=0)
            ax.set_title(cam, fontsize=9)
            ax.axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out_path = os.path.join(save_folder, f'{scene_name}_cameras_{suffix}.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def main():
    '''Main function.'''

    args = get_parser()

    # os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(x) for x in args.train_gpu)
    cudnn.benchmark = True
    if args.manual_seed is not None:
        random.seed(args.manual_seed)
        np.random.seed(args.manual_seed)
        torch.manual_seed(args.manual_seed)
        torch.cuda.manual_seed(args.manual_seed)
        torch.cuda.manual_seed_all(args.manual_seed)

    print(
        'torch.__version__:%s\ntorch.version.cuda:%s\ntorch.backends.cudnn.version:%s\ntorch.backends.cudnn.enabled:%s' % (
            torch.__version__, torch.version.cuda, torch.backends.cudnn.version(), torch.backends.cudnn.enabled))

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed    
    args.ngpus_per_node = len(args.test_gpu)
    if len(args.test_gpu) == 1:
        args.sync_bn = False
        args.distributed = False
        args.multiprocessing_distributed = False
        args.use_apex = False

    
    # By default we do not use shared memory for evaluation
    if not hasattr(args, 'use_shm'):
        args.use_shm = False
    if args.use_shm:
        if args.multiprocessing_distributed:
            args.world_size = args.ngpus_per_node * args.world_size
            mp.spawn(main_worker, nprocs=args.ngpus_per_node, args=(args.ngpus_per_node, args))
    else:
        main_worker(args.test_gpu, args.ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, argss):
    global args
    args = argss
    if args.distributed:
        if args.multiprocessing_distributed:
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url, world_size=args.world_size,
                                rank=args.rank)

    model = get_model(args)
    if main_process():
        global logger
        logger = get_logger()
        logger.info(args)

    if args.distributed:
        torch.cuda.set_device(gpu)
        args.test_batch_size = int(args.test_batch_size / ngpus_per_node)
        args.test_workers = int(args.test_workers / ngpus_per_node)
        model = torch.nn.parallel.DistributedDataParallel(model.cuda(), device_ids=[gpu])
    else:
        model = model.cuda()

    if args.feature_type == 'fusion':
        pass # do not need to load weight
    elif is_url(args.model_path): # load from url
        checkpoint = model_zoo.load_url(args.model_path, progress=True)
        model.load_state_dict(checkpoint['state_dict'], strict=True)
    
    elif args.model_path is not None and os.path.isfile(args.model_path):
        # load from directory
        if main_process():
            logger.info("=> loading checkpoint '{}'".format(args.model_path))
        checkpoint = torch.load(args.model_path, map_location=lambda storage, loc: storage.cuda())
        try:
            model.load_state_dict(checkpoint['state_dict'], strict=True)
        except Exception as ex:
            # The model was trained in a parallel manner, so need to be loaded differently
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in checkpoint['state_dict'].items():
                if k.startswith('module.'):
                    # remove module
                    k = k[7:]
                else:
                    # add module
                    k = 'module.' + k

                new_state_dict[k]=v
            model.load_state_dict(new_state_dict, strict=True)
            logger.info('Loaded a parallel model')

        if main_process():
            logger.info("=> loaded checkpoint '{}' (epoch {})".format(args.model_path, checkpoint['epoch']))    
    else:
        raise RuntimeError("=> no checkpoint found at '{}'".format(args.model_path))

    # ####################### Data Loader ####################### #
    if not hasattr(args, 'input_color'):
        # by default we do not use the point color as input
        args.input_color = False
    
    from dataset.feature_loader import FusedFeatureLoader, collation_fn_eval_all
    val_data = FusedFeatureLoader(datapath_prefix=args.data_root,
                                datapath_prefix_feat=args.data_root_2d_fused_feature,
                                voxel_size=args.voxel_size, 
                                split=args.split, aug=False,
                                memcache_init=args.use_shm, eval_all=True, identifier=6797,
                                input_color=args.input_color)
    val_sampler = None
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=args.test_batch_size,
                                                shuffle=False, num_workers=args.test_workers, pin_memory=True,
                                                drop_last=False, collate_fn=collation_fn_eval_all,
                                                sampler=val_sampler)

    # ####################### Test ####################### #
    labelset_name = args.data_root.split('/')[-1]
    if hasattr(args, 'labelset'):
        # if the labelset is specified
        labelset_name = args.labelset

    evaluate(model, val_loader, labelset_name)

def plot_camera_views(pcl, colors, scene_name, split, data_root_3d, save_path, labelset, palette, present_labels):
    """Project 3D labeled points onto 6 nuScenes cameras and save TWO plots:
      {stem}_front.png  — front_left | front | front_right
      {stem}_back.png   — back_left  | back  | back_right
    Each plot is a 1x3 grid of full-size camera images with a legend row at the bottom.

    Args:
        pcl:            (N, 3) numpy array, world-frame XYZ coordinates
        colors:         (N, 3) float [0,1] array, per-point RGB label colours
        scene_name:     e.g. 'scene-0061'
        split:          'train' or 'val'
        data_root_3d:   e.g. 'data/nuscenes_3d'  (used to derive 'data/nuscenes_2d')
        save_path:      base PNG path; '_front.png' / '_back.png' are derived from its stem
        labelset:       list of class name strings (16 + 'unlabeled')
        palette:        flat 1D uint8 array [R0,G0,B0, R1,G1,B1, ...] of class colours
        present_labels: label indices present in this scene (for legend)
    """
    img_root  = data_root_3d.replace('nuscenes_3d', 'nuscenes_2d')
    scene_dir = os.path.join(img_root, split, scene_name)
    IMG_W, IMG_H = 800, 450
    stem = os.path.splitext(save_path)[0]

    if hasattr(pcl, 'numpy'):
        pcl = pcl.numpy()
    pcl = pcl.astype(np.float64)
    pts_h = np.hstack([pcl, np.ones((len(pcl), 1), dtype=np.float64)])  # (N, 4)

    # Build legend patches once — shared by both plots
    patches = []
    for lbl_idx in sorted(set(present_labels)):
        lbl_idx = int(lbl_idx)
        if lbl_idx < len(labelset) and lbl_idx * 3 + 2 < len(palette):
            color = np.array(palette[lbl_idx*3:(lbl_idx+1)*3], dtype=float) / 255.0
            patches.append(mpatches.Patch(facecolor=color, label=labelset[lbl_idx]))

    cam_rows = {
        'front': ['front_left', 'front', 'front_right'],
        'back':  ['back_left',  'back',  'back_right'],
    }

    for row_name, cam_list in cam_rows.items():
        out_path = f'{stem}_{row_name}.png'
        # 2-row gridspec: images (tall) + legend (short)
        fig = plt.figure(figsize=(18, 7))
        gs  = gridspec.GridSpec(2, 3, height_ratios=[5, 1], hspace=0.05, wspace=0.02)

        for col, cam in enumerate(cam_list):
            ax = fig.add_subplot(gs[0, col])

            img_path  = os.path.join(scene_dir, 'color', cam + '.jpg')
            K_path    = os.path.join(scene_dir, 'K',     cam + '.npy')
            pose_path = os.path.join(scene_dir, 'pose',  cam + '.npy')

            if not os.path.exists(img_path):
                ax.text(0.5, 0.5, f'{cam}\n(missing)', ha='center', va='center',
                        transform=ax.transAxes, fontsize=9)
                ax.axis('off')
                continue

            img = PILImage.open(img_path)
            K   = np.load(K_path).astype(np.float64)
            c2w = np.load(pose_path).astype(np.float64)

            w2c     = np.linalg.inv(c2w)
            cam_pts = (w2c @ pts_h.T).T   # (N, 4)
            depth   = cam_pts[:, 2]

            ax.imshow(img)
            in_front = depth > 0.1
            if in_front.sum() > 0:
                uv = (K @ cam_pts[in_front, :3].T).T  # (M, 3)
                u  = uv[:, 0] / uv[:, 2]
                v  = uv[:, 1] / uv[:, 2]
                in_bounds = (u >= 0) & (u < IMG_W) & (v >= 0) & (v < IMG_H)
                if in_bounds.sum() > 0:
                    c_pts = colors[in_front][in_bounds]  # float [0,1]
                    ax.scatter(u[in_bounds], v[in_bounds],
                               c=c_pts, s=0.5, linewidths=0, alpha=0.85)

            ax.set_title(cam.replace('_', ' '), fontsize=11, pad=3)
            ax.axis('off')

        # Legend row spanning all three columns
        ax_leg = fig.add_subplot(gs[1, :])
        ax_leg.axis('off')
        if patches:
            ax_leg.legend(handles=patches, loc='center',
                          ncol=min(len(patches), 9),
                          fontsize=9, frameon=False, columnspacing=1.0)

        label_str = os.path.basename(out_path).replace('.png', '')
        fig.suptitle(f'{scene_name}  |  {label_str}', fontsize=12, y=0.99)
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close(fig)


def evaluate(model, val_data_loader, labelset_name='scannet_3d'):
    '''Evaluate our OpenScene model.'''

    torch.backends.cudnn.enabled = False

    if not os.path.exists(args.save_folder):
        os.makedirs(args.save_folder, exist_ok=True)

    if args.save_feature_as_numpy: # save point features to folder
        out_root = os.path.commonprefix([args.save_folder, args.model_path])
        saved_feature_folder = os.path.join(out_root, 'saved_feature')
        os.makedirs(saved_feature_folder, exist_ok=True)

    # short hands
    save_folder = args.save_folder
    feature_type = args.feature_type
    eval_iou = True
    if hasattr(args, 'eval_iou'):
        eval_iou = args.eval_iou
    mark_no_feature_to_unknown = False
    if hasattr(args, 'mark_no_feature_to_unknown') and args.mark_no_feature_to_unknown and feature_type == 'fusion':
        # some points do not have 2D features from 2D feature fusion. Directly assign 'unknown' label to those points during inference
        mark_no_feature_to_unknown = True
    vis_input = False
    if hasattr(args, 'vis_input') and args.vis_input:
        vis_input = True
    vis_pred = False
    if hasattr(args, 'vis_pred') and args.vis_pred:
        vis_pred = True
    vis_gt = False
    if hasattr(args, 'vis_gt') and args.vis_gt:
        vis_gt = True

    text_features, labelset, mapper, palette = \
        precompute_text_related_properties(labelset_name)

    with torch.no_grad():
        model.eval()
        store = 0.0
        for rep_i in range(args.test_repeats):
            preds, gts = [], []
            val_data_loader.dataset.offset = rep_i
            if main_process():
                logger.info(
                    "\nEvaluation {} out of {} runs...\n".format(rep_i+1, args.test_repeats))

            # repeat the evaluation process
            # to account for the randomness in MinkowskiNet voxelization
            if rep_i>0:
                seed = np.random.randint(10000)
                random.seed(seed)
                np.random.seed(seed)
                torch.manual_seed(seed)
                torch.cuda.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)

            if mark_no_feature_to_unknown:
                masks = []

            for i, (coords, feat, label, feat_3d, mask, inds_reverse) in enumerate(tqdm(val_data_loader)):
                t_iter_start = time.time()
                print(f"\n[EVAL] Scene {i}: coords={coords.shape} feat_3d={feat_3d.shape} "
                      f"label={label.shape} inds_reverse={inds_reverse.shape}")

                sinput = SparseTensor(feat.cuda(non_blocking=True), coords.cuda(non_blocking=True))
                coords = coords[inds_reverse, :]
                pcl = coords[:, 1:].cpu().numpy()
                t_sparse = time.time()
                print(f"[EVAL] SparseTensor + coord expand: {t_sparse - t_iter_start:.2f}s")

                if feature_type == 'distill':
                    predictions = model(sinput)
                    predictions = predictions[inds_reverse, :]
                    pred = predictions.half() @ text_features.t()
                    logits_pred = torch.max(pred, 1)[1].cpu()
                elif feature_type == 'fusion':
                    t0 = time.time()
                    predictions = feat_3d.cuda(non_blocking=True)[inds_reverse, :]
                    torch.cuda.synchronize()
                    t1 = time.time()
                    print(f"[EVAL] feat_3d to GPU + index: {t1 - t0:.2f}s | predictions={predictions.shape}")

                    pred = predictions.half() @ text_features.t()
                    torch.cuda.synchronize()
                    t2 = time.time()
                    print(f"[EVAL] matmul (N,768)@(768,43): {t2 - t1:.2f}s")

                    logits_pred = torch.max(pred, 1)[1].detach().cpu()
                    if mark_no_feature_to_unknown:
                    # some points do not have 2D features from 2D feature fusion.
                    # Directly assign 'unknown' label to those points during inference.
                        logits_pred[~mask[inds_reverse]] = len(labelset)-1
                    t3 = time.time()
                    print(f"[EVAL] argmax + mask: {t3 - t2:.2f}s | total scene: {t3 - t_iter_start:.2f}s")

                elif feature_type == 'ensemble':
                    feat_fuse = feat_3d.cuda(non_blocking=True)[inds_reverse, :]
                    # pred_fusion = feat_fuse.half() @ text_features.t()
                    pred_fusion = (feat_fuse/(feat_fuse.norm(dim=-1, keepdim=True)+1e-5)).half() @ text_features.t()

                    predictions = model(sinput)
                    predictions = predictions[inds_reverse, :]
                    # pred_distill = predictions.half() @ text_features.t()
                    pred_distill = (predictions/(predictions.norm(dim=-1, keepdim=True)+1e-5)).half() @ text_features.t()

                    # logits_distill = torch.max(pred_distill, 1)[1].detach().cpu()
                    # mask_ensem = pred_distill<pred_fusion # confidence-based ensemble
                    # pred = pred_distill
                    # pred[mask_ensem] = pred_fusion[mask_ensem]
                    # logits_pred = torch.max(pred, 1)[1].detach().cpu()

                    feat_ensemble = predictions.clone().half()
                    mask_ =  pred_distill.max(dim=-1)[0] < pred_fusion.max(dim=-1)[0]
                    feat_ensemble[mask_] = feat_fuse[mask_]
                    pred = feat_ensemble @ text_features.t()
                    logits_pred = torch.max(pred, 1)[1].detach().cpu()

                    predictions = feat_ensemble # if we need to save the features
                else:
                    raise NotImplementedError

                if args.save_feature_as_numpy:
                    scene_name = val_data_loader.dataset.data_paths[i].split('/')[-1].split('.pth')[0]
                    np.save(os.path.join(saved_feature_folder, '{}_openscene_feat_{}.npy'.format(scene_name, feature_type)), predictions.cpu().numpy())
                
                # Visualize the input, predictions and GT

                # special case for nuScenes, evaluation points are only a subset of input
                if 'nuscenes' in labelset_name:
                    label_mask = (label!=255)
                    label = label[label_mask]
                    logits_pred = logits_pred[label_mask]
                    pred = pred[label_mask]
                    if vis_pred or vis_gt:
                        pcl = torch.load(val_data_loader.dataset.data_paths[i])[0][label_mask]
                    # extract scene/split info once per scene for camera view plots
                    _pth_path = val_data_loader.dataset.data_paths[i]
                    _scene_name_str = os.path.basename(_pth_path).replace('.pth', '')
                    _split_str = os.path.basename(os.path.dirname(_pth_path))
                    _labelset_16 = list(NUSCENES_LABELS_16)
                    _labelset_16[4] = 'construct. vehicle'
                    _labelset_16[10] = 'road'
                    _labelset_16.append('unlabeled')

                if vis_input:
                    input_color = torch.load(val_data_loader.dataset.data_paths[i])[1]
                    export_pointcloud(os.path.join(save_folder, '{}_input.ply'.format(i)), pcl, colors=(input_color+1)/2)

                if vis_pred:
                    if mapper is not None:
                        pred_label_color = convert_labels_with_palette(mapper[logits_pred].numpy(), palette)
                        export_pointcloud(os.path.join(save_folder, '{}_{}.ply'.format(i, feature_type)), pcl, colors=pred_label_color)
                        if 'nuscenes' in labelset_name:
                            plot_camera_views(
                                pcl, pred_label_color, _scene_name_str,
                                _split_str, args.data_root,
                                os.path.join(save_folder, f'{_scene_name_str}_cameras_{feature_type}.png'),
                                _labelset_16, palette,
                                list(np.unique(mapper[logits_pred].numpy())))
                    else:
                        pred_label_color = convert_labels_with_palette(logits_pred.numpy(), palette)
                        export_pointcloud(os.path.join(save_folder, '{}_{}.ply'.format(i, feature_type)), pcl, colors=pred_label_color)
                        visualize_labels(list(np.unique(logits_pred.numpy())),
                                    labelset,
                                    palette,
                                    os.path.join(save_folder, '{}_labels_{}.jpg'.format(i, feature_type)), ncol=5)

                # Visualize GT labels
                if vis_gt:
                    # for points not evaluating
                    label[label==255] = len(labelset)-1
                    gt_label_color = convert_labels_with_palette(label.cpu().numpy(), palette)
                    export_pointcloud(os.path.join(save_folder, '{}_gt.ply'.format(i)), pcl, colors=gt_label_color)
                    visualize_labels(list(np.unique(label.cpu().numpy())),
                                labelset,
                                palette,
                                os.path.join(save_folder, '{}_labels_gt.jpg'.format(i)), ncol=5)

                    if 'nuscenes' in labelset_name:
                        all_digits = np.unique(np.concatenate([np.unique(mapper[logits_pred].numpy()), np.unique(label)]))
                        labelset_16 = list(NUSCENES_LABELS_16)
                        labelset_16[4] = 'construct. vehicle'
                        labelset_16[10] = 'road'
                        labelset_16.append('unlabeled')
                        # Filter to valid indices only
                        all_digits = all_digits[all_digits < len(labelset_16)]
                        visualize_labels(list(all_digits), labelset_16,
                            palette, os.path.join(save_folder, '{}_label.jpg'.format(i)), ncol=max(all_digits.shape[0], 1))
                        # Camera projection for GT (skip if already generated for this scene)
                        _gt_cam = os.path.join(save_folder, '{}_cameras_gt.png'.format(_scene_name_str))
                        if not os.path.exists(_gt_cam.replace('.png', '_front.png')):
                            plot_camera_views(
                                pcl, gt_label_color, _scene_name_str,
                                _split_str, args.data_root,
                                _gt_cam,
                                _labelset_16, palette,
                                list(np.unique(label.cpu().numpy())))

                if eval_iou:
                    if mark_no_feature_to_unknown:
                        if "nuscenes" in labelset_name: # special case
                            masks.append(mask[inds_reverse][label_mask])
                        else:
                            masks.append(mask[inds_reverse])

                    if args.test_repeats==1:
                        # save directly the logits
                        preds.append(logits_pred)
                    else:
                        # only save the dot-product results, for repeat prediction
                        preds.append(pred.cpu())

                    gts.append(label.cpu())

            if eval_iou:
                gt = torch.cat(gts)
                pred = torch.cat(preds)

                pred_logit = pred
                if args.test_repeats>1:
                    pred_logit = pred.float().max(1)[1]

                if mapper is not None:
                    pred_logit = mapper[pred_logit]

                if mark_no_feature_to_unknown:
                    mask = torch.cat(masks)
                    pred_logit[~mask] = 256

                if args.test_repeats==1:
                    current_iou = metric.evaluate(pred_logit.numpy(),
                                                gt.numpy(),
                                                dataset=labelset_name,
                                                stdout=True)
                if args.test_repeats > 1:
                    store = pred + store
                    store_logit = store.float().max(1)[1]
                    if mapper is not None:
                        store_logit = mapper[store_logit]

                    if mark_no_feature_to_unknown:
                        store_logit[~mask] = 256
                    accumu_iou = metric.evaluate(store_logit.numpy(),
                                                gt.numpy(),
                                                stdout=True,
                                                dataset=labelset_name)

if __name__ == '__main__':
    main()
