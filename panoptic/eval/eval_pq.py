"""Panoptic Quality evaluation (PQ/SQ/RQ + mIoU).

Adapted from MaskPLS eval_np.py (SemanticKITTI convention).
Our convention: 16 classes (0-indexed: 0-15), ignore = none by default.
Things: classes 0-9, Stuff: classes 10-15.
"""
import numpy as np


class PanopticEval:
    def __init__(self, n_classes=16, ignore=None, min_points=50):
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
        idxs = np.stack([x_sem, y_sem], axis=0)
        np.add.at(self.px_iou_conf_matrix, tuple(idxs), 1)

    def _addPanoptic(self, x_sem, x_inst, y_sem, y_inst):
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
