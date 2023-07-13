from ._base_metric import _BaseMetric
from .. import _timing
import numpy as np
from scipy.optimize import linear_sum_assignment

class DE(_BaseMetric):
    """Class which simply counts the number of tracker and gt detections and ids."""
    def __init__(self, config=None):
        super().__init__()
        self.loss_fields = ['FDE','ADE']
        self.fields = self.loss_fields
        self.summary_fields = self.fields

    @_timing.time
    def eval_sequence(self, data):
        """Returns counts for one sequence"""
        # Get results
        res = {}
        for field in self.fields:
            res[field] = 0

        m=data['num_gt_ids']
        n=data['num_tracker_ids']
        dist_sum = np.zeros((m,n))

        min_time_gt = len(data['gt_ids'])
        ids_at_first=data['gt_ids'][0]
        for ind, i in enumerate(data['gt_ids']):
            if len(i)!=len(ids_at_first):
                min_time_gt = ind
                break
        
        min_time_tracker = len(data['tracker_ids'])
        ids_at_first=data['tracker_ids'][0]
        for ind, i in enumerate(data['tracker_ids']):
            if len(i)!=len(ids_at_first):
                min_time_tracker = ind
                break
        
        if min_time_gt!=min_time_tracker:
            print(min_time_tracker, min_time_gt)
        min_time = min(min_time_gt, min_time_tracker)

        for t1, (gt_ids_t, tracker_ids_t) in enumerate(zip(data['gt_ids'][:min_time], data['tracker_ids'][:min_time])):
            dist = - data['similarity_scores'][t1]
            dist_sum += dist
        trk_dist = dist_sum / min_time
        match_rows, match_cols = linear_sum_assignment(trk_dist)

        if min_time_gt>min_time:
            for t2, (gt_ids_t, tracker_ids_t) in enumerate(zip(data['gt_ids'][min_time:min_time_gt], data['tracker_ids'][min_time:min_time_gt])):
                dist = - data['similarity_scores'][t1+t2]
                dist_sum += dist
            trk_dist = dist_sum / min_time_gt
            ade = np.mean(trk_dist[match_rows,match_cols])
            fde = np.mean(np.linalg.norm(data['gt_dets_3d'][min_time_gt-1][match_rows]-data['tracker_dets_3d'][-1][match_cols]), axis=1)
        else:
            ade = np.mean(trk_dist[match_rows,match_cols])
            fde = np.mean(np.linalg.norm(data['gt_dets_3d'][min_time-1][match_rows]-data['tracker_dets_3d'][min_time-1][match_cols], axis=1))
        
        res['ADE']=ade
        res['FDE']=fde

        return res

    def combine_sequences(self, all_res):
        """Combines metrics across all sequences"""
        res = {}
        for field in self.loss_fields:
            res[field] = self._combine_average(all_res, field)
        return res

    def combine_classes_class_averaged(self, all_res, ignore_empty_classes=None):
        """Combines metrics across all classes by averaging over the class values"""
        res = {}
        for field in self.loss_fields:
            res[field] = self._combine_sum(all_res, field)
        return res

    def combine_classes_det_averaged(self, all_res):
        """Combines metrics across all classes by averaging over the detection values"""
        res = {}
        for field in self.loss_fields:
            res[field] = self._combine_sum(all_res, field)
        return res