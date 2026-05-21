"""Panoptic Quality evaluation (PQ/SQ/RQ + mIoU).

Adapted from MaskPLS eval_np.py (SemanticKITTI convention).
Our convention: 16 classes (0-indexed: 0-15), ignore = none by default.
Things: classes 0-9, Stuff: classes 10-15.
"""
import numpy as np


class PanopticEval:
    """Panoptic Quality (PQ) evaluator with auxiliary semantic mean-IoU computation.

    Accumulates per-class true-positive, false-positive, and false-negative counts
    for panoptic segments (IoU > 0.5 matching) as well as a pixel-level confusion
    matrix for semantic IoU. Used after inference to compute PQ, SQ, RQ, and mIoU
    metrics following the Kirillov et al. CVPR 2019 protocol.

    Typical usage:
        1. Instantiate once before the evaluation loop.
        2. Call ``addBatch`` for each predicted/GT sample.
        3. Call ``getPQ`` and ``getSemIoU`` to retrieve final metrics.

    Attributes:
        n_classes (int): Total number of semantic classes (default 16).
        include (np.ndarray): Array of class indices that are evaluated.
        offset (int): Large multiplier (2^32) for encoding GT/pred instance pairs.
        min_points (int): Minimum segment size to count as FP/FN.
    """

    def __init__(self, n_classes=16, ignore=None, min_points=50):
        """Initialize confusion matrices and per-class TP/FP/FN accumulators.

        Called once before the evaluation loop begins. Sets up the class inclusion
        list (excluding ignored classes) and allocates zeroed accumulator arrays.

        Steps:
            1. Store number of classes, epsilon, and offset constant.
            2. Compute ``include`` array (all classes minus ignored ones).
            3. Call ``self.reset()`` to zero all accumulator arrays.

        Args:
            n_classes (int): Number of semantic classes (0-indexed). Default 16.
            ignore (list[int] | None): Class indices to exclude from evaluation.
            min_points (int): Minimum number of points for a segment to be
                counted as a false positive or false negative.
        """
        self.n_classes = n_classes
        self.ignore = np.array(ignore if ignore is not None else [], dtype=np.int64)
        self.include = np.array(
            [n for n in range(n_classes) if n not in self.ignore], dtype=np.int64
        )
        self.offset = 2 ** 32
        self.min_points = min_points
        self.eps = 1e-15
        self.reset()

    def reset(self):
        """Zero all accumulator arrays to prepare for a fresh evaluation pass.

        Called during ``__init__`` and can be called manually to reuse the same
        evaluator object across multiple evaluation runs without re-instantiation.

        Steps:
            1. Zero the (n_classes x n_classes) semantic confusion matrix.
            2. Zero per-class panoptic TP counts and IoU sums.
            3. Zero per-class panoptic FP and FN counts.

        Returns:
            None
        """
        self.px_iou_conf_matrix = np.zeros(
            (self.n_classes, self.n_classes), dtype=np.int64
        )
        self.pan_tp = np.zeros(self.n_classes, dtype=np.int64)
        self.pan_iou = np.zeros(self.n_classes, dtype=np.double)
        self.pan_fp = np.zeros(self.n_classes, dtype=np.int64)
        self.pan_fn = np.zeros(self.n_classes, dtype=np.int64)

    def addBatch(self, x_sem, x_inst, y_sem, y_inst):
        """Add a batch. x=pred, y=gt. All arrays are 1-D numpy int."""
        self._addSemIoU(x_sem, y_sem)
        self._addPanoptic(x_sem, x_inst, y_sem, y_inst)

    def _addSemIoU(self, x_sem, y_sem):
        """Accumulate the semantic confusion matrix for one sample.

        Updates the (n_classes x n_classes) confusion matrix by incrementing
        counts at positions ``[predicted_class, ground_truth_class]`` for every
        point in the sample. Used internally by ``addBatch``.

        Steps:
            1. Stack predicted and GT semantic arrays into a (2, N) index array.
            2. Use ``np.add.at`` to scatter-add 1 at each (pred, gt) pair.

        Args:
            x_sem (np.ndarray): Predicted semantic labels, shape ``(N,)``, dtype int.
            y_sem (np.ndarray): Ground-truth semantic labels, shape ``(N,)``, dtype int.

        Returns:
            None
        """
        idxs = np.stack([x_sem, y_sem], axis=0)
        np.add.at(self.px_iou_conf_matrix, tuple(idxs), 1)

    def _addPanoptic(self, x_sem, x_inst, y_sem, y_inst):
        """Compute panoptic TP/FP/FN for one sample using IoU > 0.5 matching.

        For each included class, finds all predicted and ground-truth segments,
        computes pairwise intersection via an offset-encoding trick, calculates
        IoU, and matches segments with IoU > 0.5 as true positives. Unmatched
        GT segments (>= min_points) become FN; unmatched pred segments become FP.

        Steps:
            1. Shift instance IDs by +1 so that 0 means "no segment".
            2. Remove points belonging to ignored classes.
            3. For each included class, isolate predicted and GT instance masks.
            4. Encode (pred_inst, gt_inst) pairs via offset multiplication.
            5. Compute intersection, union, and IoU for all overlapping pairs.
            6. Match pairs with IoU > 0.5 as TP; accumulate IoU for SQ.
            7. Count unmatched GT (>= min_points) as FN, unmatched pred as FP.

        Args:
            x_sem (np.ndarray): Predicted semantic labels, shape ``(N,)``, dtype int.
            x_inst (np.ndarray): Predicted instance IDs, shape ``(N,)``, dtype int.
            y_sem (np.ndarray): Ground-truth semantic labels, shape ``(N,)``, dtype int.
            y_inst (np.ndarray): Ground-truth instance IDs, shape ``(N,)``, dtype int.

        Returns:
            None
        """
        x_inst = x_inst + 1
        y_inst = y_inst + 1
        for cl in self.ignore:
            mask = y_sem != cl
            x_sem, y_sem = x_sem[mask], y_sem[mask]
            x_inst, y_inst = x_inst[mask], y_inst[mask]

        for cl in self.include:
            x_in_cl = x_inst * (x_sem == cl).astype(np.int64)
            y_in_cl = y_inst * (y_sem == cl).astype(np.int64)

            unique_pred, counts_pred = np.unique(x_in_cl[x_in_cl > 0], return_counts=True)
            id2idx_pred = {id: idx for idx, id in enumerate(unique_pred)}
            matched_pred = np.zeros(len(unique_pred), dtype=bool)

            unique_gt, counts_gt = np.unique(y_in_cl[y_in_cl > 0], return_counts=True)
            id2idx_gt = {id: idx for idx, id in enumerate(unique_gt)}
            matched_gt = np.zeros(len(unique_gt), dtype=bool)

            valid = (x_in_cl > 0) & (y_in_cl > 0)
            offset_combo = x_in_cl[valid] + self.offset * y_in_cl[valid]
            unique_combo, counts_combo = np.unique(offset_combo, return_counts=True)

            gt_labels = unique_combo // self.offset
            pred_labels = unique_combo % self.offset
            gt_areas = np.array([counts_gt[id2idx_gt[id]] for id in gt_labels])
            pred_areas = np.array([counts_pred[id2idx_pred[id]] for id in pred_labels])
            intersections = counts_combo
            unions = gt_areas + pred_areas - intersections
            ious = intersections.astype(np.float64) / unions.astype(np.float64)

            tp_mask = ious > 0.5
            self.pan_tp[cl] += tp_mask.sum()
            self.pan_iou[cl] += ious[tp_mask].sum()

            matched_gt[[id2idx_gt[id] for id in gt_labels[tp_mask]]] = True
            matched_pred[[id2idx_pred[id] for id in pred_labels[tp_mask]]] = True

            self.pan_fn[cl] += ((counts_gt >= self.min_points) & ~matched_gt).sum()
            self.pan_fp[cl] += ((counts_pred >= self.min_points) & ~matched_pred).sum()

    def getSemIoU(self):
        """Compute mean IoU and per-class IoU from the accumulated confusion matrix.

        Called after all batches have been added to retrieve the final semantic
        segmentation metric. Zeros out columns corresponding to ignored classes
        before computing intersection-over-union.

        Steps:
            1. Copy confusion matrix and zero ignored-class columns.
            2. Extract TP (diagonal), FP (row sum - TP), FN (col sum - TP).
            3. Compute IoU = TP / (TP + FP + FN) per class.
            4. Average over included classes for mIoU.

        Returns:
            tuple: (iou_mean, iou) where
                - iou_mean (float): Mean IoU across included classes.
                - iou (np.ndarray): Per-class IoU array, shape ``(n_classes,)``.
        """
        conf = self.px_iou_conf_matrix.copy().astype(np.double)
        conf[:, self.ignore] = 0
        tp = conf.diagonal()
        fp = conf.sum(axis=1) - tp
        fn = conf.sum(axis=0) - tp
        union = np.maximum(tp + fp + fn, self.eps)
        iou = tp / union
        iou_mean = iou[self.include].mean()
        return iou_mean, iou

    def getPQ(self):
        """Compute Panoptic Quality, Segmentation Quality, and Recognition Quality.

        Called after all batches have been added. PQ = SQ * RQ where
        SQ = mean IoU of matched segments and RQ = TP / (TP + 0.5*FP + 0.5*FN).
        Returns both per-class and mean values (averaged over included classes).

        Steps:
            1. Compute SQ = accumulated IoU / TP per class.
            2. Compute RQ = TP / (TP + 0.5*FP + 0.5*FN) per class.
            3. Compute PQ = SQ * RQ per class.
            4. Average PQ, SQ, RQ over included classes.

        Returns:
            tuple: (pq_mean, sq_mean, rq_mean, pq, sq, rq) where
                - pq_mean (float): Mean PQ across included classes.
                - sq_mean (float): Mean SQ across included classes.
                - rq_mean (float): Mean RQ across included classes.
                - pq (np.ndarray): Per-class PQ, shape ``(n_classes,)``.
                - sq (np.ndarray): Per-class SQ, shape ``(n_classes,)``.
                - rq (np.ndarray): Per-class RQ, shape ``(n_classes,)``.
        """
        sq = self.pan_iou / np.maximum(self.pan_tp.astype(np.double), self.eps)
        rq = self.pan_tp.astype(np.double) / np.maximum(
            self.pan_tp + 0.5 * self.pan_fp + 0.5 * self.pan_fn, self.eps
        )
        pq = sq * rq
        return (
            pq[self.include].mean(),
            sq[self.include].mean(),
            rq[self.include].mean(),
            pq, sq, rq,
        )
