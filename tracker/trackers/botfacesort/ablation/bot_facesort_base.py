from collections import deque
import numpy as np

from tracker.appearance.adaface_backend import (
    AdafaceBackendTorch,
    AdafaceBackendONNX,
)
from tracker.motion.cmc.sof import SparseOptFlow
from tracker.motion.kalman_filters.botsort_kf import KalmanFilter
from ..basetrack import BaseTrack, TrackState
from tracker.utils.matching import (
    embedding_distance,
    iou_distance,
    linear_assignment,
)
from tracker.utils.ops import xywh2xyxy, xyxy2xywh
from ..utils import (
    joint_stracks,
    sub_stracks,
    remove_duplicate_stracks,
)


class STrack(BaseTrack):
    shared_kalman = KalmanFilter()

    def __init__(self, det, face_feat=None, feat_history=1000):
        # wait activate
        self.xywh = xyxy2xywh(det[:4])  # (x1, y1, x2, y2) --> (xc, yc, w, h)
        self.score = det[4]
        self.cls = det[5]
        self.det_ind = det[6]
        self.kalman_filter = None
        self.mean, self.covariance = None, None
        self.is_activated = False

        # We don't update the class since the label is only face
        self.cls_hist = []  # (cls id, freq)
        self.update_cls(self.cls, self.score)

        self.tracklet_len = 0

        self.smooth_feat = None
        self.curr_feat = face_feat

        self.body_features = deque([], maxlen=feat_history)
        self.face_features = deque([], maxlen=feat_history)
        self.alpha = 0.9

    def update_features(self, feat, norm=True):
        if norm:
            feat /= np.linalg.norm(feat)
        self.curr_feat = feat
        if self.smooth_feat is None:
            self.smooth_feat = feat
        else:
            self.smooth_feat = self.alpha * self.smooth_feat + (1 - self.alpha) * feat
        self.face_features.append(feat)
        if norm:
            self.smooth_feat /= np.linalg.norm(self.smooth_feat)

    def update_cls(self, cls, score):
        if len(self.cls_hist) > 0:
            max_freq = 0
            found = False
            for c in self.cls_hist:
                if cls == c[0]:
                    c[1] += score
                    found = True

                if c[1] > max_freq:
                    max_freq = c[1]
                    self.cls = c[0]
            if not found:
                self.cls_hist.append([cls, score])
                self.cls = cls
        else:
            self.cls_hist.append([cls, score])
            self.cls = cls

    # def predict(self):
    #     mean_state = self.mean.copy()
    #     if self.state != TrackState.Tracked:
    #         mean_state[6] = 0
    #         mean_state[7] = 0
    #     self.mean, self.covariance = self.kalman_filter.predict(
    #         mean_state, self.covariance
    #     )

    @staticmethod
    def multi_predict(stracks):
        if len(stracks) > 0:
            multi_mean = np.asarray([st.mean.copy() for st in stracks])
            multi_covariance = np.asarray([st.covariance for st in stracks])
            for i, st in enumerate(stracks):
                if st.state != TrackState.Tracked:
                    multi_mean[i][6] = 0
                    multi_mean[i][7] = 0
            multi_mean, multi_covariance = STrack.shared_kalman.multi_predict(
                multi_mean, multi_covariance
            )
            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                stracks[i].mean = mean
                stracks[i].covariance = cov

    @staticmethod
    def multi_gmc(stracks, H=np.eye(2, 3)):
        """_summary_

        Args:
            stracks (_type_): _description_
            H (_type_, optional): _description_. Defaults to np.eye(2, 3).
        """
        if len(stracks) > 0:
            multi_mean = np.asarray([st.mean.copy() for st in stracks])
            multi_covariance = np.asarray([st.covariance for st in stracks])

            R = H[:2, :2]
            R8x8 = np.kron(np.eye(4, dtype=float), R)
            t = H[:2, 2]

            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                mean = R8x8.dot(mean)
                mean[:2] += t
                cov = R8x8.dot(cov).dot(R8x8.transpose())

                stracks[i].mean = mean
                stracks[i].covariance = cov

    def activate(self, frame_id):
        """Start a new tracklet"""
        self.kalman_filter = KalmanFilter()
        self.id = self.next_id()

        if self.curr_feat is not None:
            self.update_features(self.curr_feat)

        self.mean, self.covariance = self.kalman_filter.initiate(self.xywh)

        self.tracklet_len = 0
        self.state = TrackState.Tracked
        if frame_id == 1:
            self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id

    def re_activate(self, new_track, frame_id, new_id=False):
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, new_track.xywh
        )
        if new_track.curr_feat is not None:
            self.update_features(new_track.curr_feat)
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.id = self.next_id()
        self.score = new_track.score
        self.cls = new_track.cls
        self.det_ind = new_track.det_ind

        self.update_cls(new_track.cls, new_track.score)

    def update(self, new_track, frame_id):
        """
        Update a matched track
        :type new_track: STrack
        :type frame_id: int
        :return:
        """
        self.frame_id = frame_id
        self.tracklet_len += 1

        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, new_track.xywh
        )

        if new_track.curr_feat is not None:
            self.update_features(new_track.curr_feat)

        self.state = TrackState.Tracked
        self.is_activated = True

        self.score = new_track.score
        self.cls = new_track.cls
        self.det_ind = new_track.det_ind
        self.update_cls(new_track.cls, new_track.score)

    @property
    def xyxy(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        if self.mean is None:
            ret = self.xywh.copy()  # (xc, yc, w, h)
        else:
            ret = self.mean[:4].copy()  # kf (xc, yc, w, h)
        ret = xywh2xyxy(ret)
        return ret


class BoTFaceSORT(object):
    def __init__(
        self,
        reid_weights,
        device,
        fp16,
        track_buffer: int = 30,
        match_thresh: float = 0.8,
        proximity_thresh: float = 0.5,
        face_appearance_thresh: float = 0.7,
        cmc_method: str = "None",
    ):
        self.tracked_stracks = []  # type: list[STrack]
        self.lost_stracks = []  # type: list[STrack]
        self.removed_stracks = []  # type: list[STrack]
        BaseTrack.clear_count()

        self.frame_id = 0

        self.match_thresh = match_thresh
        self.cmc_method = cmc_method
        self.buffer_size = track_buffer
        self.max_time_lost = self.buffer_size
        self.kalman_filter = KalmanFilter()

        # ReID module
        self.proximity_thresh = proximity_thresh
        self.face_appearance_thresh = face_appearance_thresh

        if "resnet" in str(reid_weights) and "onnx" not in str(reid_weights):
            self.face_model = AdafaceBackendTorch(
                weights=reid_weights, device=device, fp16=fp16
            )
        elif "onnx" in str(reid_weights):
            self.face_model = AdafaceBackendONNX(weights=reid_weights)
        self.cmc = SparseOptFlow()

    def update(self, face_dets, img, landmarks=None):
        assert isinstance(
            face_dets, np.ndarray
        ), f"Unsupported 'face_dets' input format '{type(face_dets)}', valid format is np.ndarray"
        assert isinstance(
            img, np.ndarray
        ), f"Unsupported 'img_numpy' input format '{type(img)}', valid format is np.ndarray"
        assert (
            len(face_dets.shape) == 2
        ), "Unsupported 'face_dets' dimensions, valid number of dimensions is two"

        self.frame_id += 1
        activated_starcks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []

        face_dets = np.hstack([face_dets, np.arange(len(face_dets)).reshape(-1, 1)])

        # Extract embeddings
        face_embs = self.face_model.get_features(img, face_dets[:, :4], landmarks)

        if len(face_dets) > 0:
            """Detections"""
            detections = [STrack(det, emb) for (det, emb) in zip(face_dets, face_embs)]
        else:
            detections = []

        """ Step 1. Filter tracked and unconfirmed stracks"""
        unconfirmed = []
        tracked_stracks = []
        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)

        """ Step 2. Bounding box prediction based on Kalman Filter"""
        strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)

        # Predict the current location with KF
        STrack.multi_predict(strack_pool)

        # Fix camera motion
        # if self.cmc_method == "sparseOptFlow":
        #     warp = self.cmc.apply(img, a_body_dets)
        #     STrack.multi_gmc(strack_pool, warp)
        #     STrack.multi_gmc(unconfirmed, warp)

        """ Step 3. First association: Association for matched bodies based on IoU distance"""
        iou_dists = iou_distance(strack_pool, detections)
        iou_dists_mask = iou_dists > self.proximity_thresh

        emb_dists = embedding_distance(strack_pool, detections) / 2.0
        emb_dists[emb_dists > self.face_appearance_thresh] = 1.0
        emb_dists[iou_dists_mask] = 1.0
        dists = np.minimum(iou_dists, emb_dists)

        # Get matched tracks, unmatched tracks and unmatched detections with current detections
        matches, u_track, u_detection = linear_assignment(
            dists, thresh=self.match_thresh
        )  # matched tracks, unmatched tracks, unmatched detections

        # Update tracklets management with first matched tracklets
        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        """ Step 3: Second association, with face appearance and IoU distance"""
        second_detections = [detections[i] for i in u_detection]

        r_tracked_stracks = [
            strack_pool[i]
            for i in u_track
            if strack_pool[i].state == TrackState.Tracked
        ]  # unmatched track 중 tracking 중인 것들

        remain_stracks = joint_stracks(r_tracked_stracks, unconfirmed)

        emb_dists = embedding_distance(remain_stracks, second_detections) / 2.0
        emb_dists[emb_dists > self.face_appearance_thresh] = 1.0
        dists = emb_dists

        matches, u_track, u_detection = linear_assignment(
            dists, thresh=self.match_thresh
        )

        for itracked, idet in matches:
            track = remain_stracks[itracked]
            det = second_detections[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        for it in u_track:
            track = remain_stracks[it]
            if not track.state == TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)

        """ Step 4: Init new stracks"""
        for inew in u_detection:
            track = second_detections[inew]
            track.activate(self.frame_id)
            activated_starcks.append(track)

        """ Step 5: Update state"""
        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)

        """ Merge """
        self.tracked_stracks = [
            t for t in self.tracked_stracks if t.state == TrackState.Tracked
        ]
        self.tracked_stracks = joint_stracks(self.tracked_stracks, activated_starcks)
        self.tracked_stracks = joint_stracks(self.tracked_stracks, refind_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed_stracks)
        self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(
            self.tracked_stracks, self.lost_stracks
        )

        output_stracks = [track for track in self.tracked_stracks if track.is_activated]
        outputs = []
        for t in output_stracks:
            output = []
            output.extend(t.xyxy)
            output.append(t.id)
            output.append(t.score)
            output.append(t.cls)
            output.append(t.det_ind)
            outputs.append(output)

        outputs = np.asarray(outputs)
        return outputs
