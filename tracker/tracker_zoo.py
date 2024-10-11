from types import SimpleNamespace

import yaml

from .utils import BOXMOT


def get_tracker_config(tracker_type):
    tracking_config = BOXMOT / "configs" / (tracker_type + ".yaml")
    return tracking_config


def create_tracker(
    tracker_type, tracker_config, reid_weights, device, half, sc=False, sm=False
):
    with open(tracker_config, "r") as f:
        cfg = yaml.load(f.read(), Loader=yaml.FullLoader)
    cfg = SimpleNamespace(**cfg)  # easier dict acces by dot, instead of ['']

    if tracker_type == "strongsort":
        from .trackers.strongsort.strong_sort import StrongSORT

        strongsort = StrongSORT(
            reid_weights,
            device,
            half,
            max_dist=cfg.max_dist,
            max_iou_dist=cfg.max_iou_dist,
            max_age=cfg.max_age,
            n_init=cfg.n_init,
            nn_budget=cfg.nn_budget,
            mc_lambda=cfg.mc_lambda,
            ema_alpha=cfg.ema_alpha,
        )
        return strongsort

    if tracker_type == "deepsort":
        from .trackers.deepsort.deep_sort import DeepSort

        deepsort = DeepSort(
            reid_weights,
            device,
            half,
            max_dist=cfg.max_dist,
            max_iou_dist=cfg.max_iou_dist,
            max_age=cfg.max_age,
            n_init=cfg.n_init,
            nn_budget=cfg.nn_budget,
        )
        return deepsort

    elif tracker_type == "ocsort":
        from .trackers.ocsort.ocsort import OCSort

        ocsort = OCSort(
            det_thresh=cfg.det_thresh,
            max_age=cfg.max_age,
            min_hits=cfg.min_hits,
            asso_threshold=cfg.iou_thresh,
            delta_t=cfg.delta_t,
            asso_func=cfg.asso_func,
            inertia=cfg.inertia,
            use_byte=cfg.use_byte,
        )
        return ocsort

    elif tracker_type == "bytetrack":
        from .trackers.bytetrack.byte_tracker import BYTETracker

        bytetracker = BYTETracker(
            track_thresh=cfg.track_thresh,
            match_thresh=cfg.match_thresh,
            track_buffer=cfg.track_buffer,
        )
        return bytetracker

    elif tracker_type == "botsort":
        from .trackers.botsort.bot_sort import BoTSORT

        botsort = BoTSORT(
            reid_weights,
            device,
            half,
            track_high_thresh=cfg.track_high_thresh,
            track_low_thresh=cfg.track_low_thresh,
            new_track_thresh=cfg.new_track_thresh,
            track_buffer=cfg.track_buffer,
            match_thresh=cfg.match_thresh,
            proximity_thresh=cfg.proximity_thresh,
            appearance_thresh=cfg.appearance_thresh,
            cmc_method=cfg.cmc_method,
        )
        return botsort
    elif tracker_type == "deepocsort":
        from .trackers.deepocsort.deep_ocsort import DeepOCSort

        deepocsort = DeepOCSort(
            reid_weights,
            device,
            half,
            det_thresh=cfg.det_thresh,
            max_age=cfg.max_age,
            min_hits=cfg.min_hits,
            iou_threshold=cfg.iou_thresh,
            delta_t=cfg.delta_t,
            asso_func=cfg.asso_func,
            w_association_emb=cfg.w_association_emb,
            inertia=cfg.inertia,
        )
        return deepocsort
    elif tracker_type == "hybridsort":
        from .trackers.hybridsort.hybridsort import HybridSORT

        hybridsort = HybridSORT(
            reid_weights,
            device,
            half,
            det_thresh=cfg.det_thresh,
            max_age=cfg.max_age,
            min_hits=cfg.min_hits,
            iou_threshold=cfg.iou_thresh,
            delta_t=cfg.delta_t,
            asso_func=cfg.asso_func,
            inertia=cfg.inertia,
            longterm_reid_weight=cfg.longterm_reid_weight,
            TCM_first_step_weight=cfg.TCM_first_step_weight,
            use_byte=cfg.use_byte,
        )
        return hybridsort
    elif tracker_type == "botfacesort":
        # Scene change detection
        if sc and sm:
            from .trackers.botfacesort.bot_facesort import BoTFaceSORT
        elif not sc and not sm:
            from .trackers.botfacesort.ablation.bot_facesort_base import BoTFaceSORT
        elif sc:
            if sm:
                from .trackers.botfacesort.ablation.bot_facesort_sc_sm import (
                    BoTFaceSORT,
                )
            else:
                from .trackers.botfacesort.ablation.bot_facesort_sc import BoTFaceSORT
        # Shared memory
        elif sm:
            if sm:
                from .trackers.botfacesort.ablation.bot_facesort_sm import BoTFaceSORT

        botfacesort = BoTFaceSORT(
            reid_weights,
            device,
            half,
            track_buffer=cfg.track_buffer,
            match_thresh=cfg.match_thresh,
            proximity_thresh=cfg.proximity_thresh,
            face_appearance_thresh=cfg.face_appearance_thresh,
            cmc_method=cfg.cmc_method,
        )
        return botfacesort
    else:
        print("No such tracker")
        exit()
