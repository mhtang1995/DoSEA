from collections import defaultdict
import logging
import numpy as np
from src.dataloader import domain_entity_types_reverse

logger = logging.getLogger(__name__)


def get_p_r_f(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp != 0 else 0
    r = tp / (tp + fn) if tp + fn != 0 else 0
    f1 = 2 * p * r / (p + r) if p + r != 0 else 0
    return np.array([p, r, f1])


def calculate_metric(gt, predict):
    """
        计算 tp fp fn
    """
    tp, fp, fn = 0, 0, 0
    for entity_predict in predict:
        flag = 0
        for entity_gt in gt:
            if entity_predict[0] == entity_gt[0] and entity_predict[1] == entity_gt[1]:
                flag = 1
                tp += 1
                break
        if flag == 0:
            fp += 1

    fn = len(gt) - tp

    return np.array([tp, fp, fn])


def mrc_decode(start_ids, end_ids, raw_token):
    predict_entities = []
    offsets = [i for i, x in enumerate(raw_token) if x == 'SEP']
    assert len(offsets) == 2
    offset1 = offsets[0]
    offset2 = offsets[1]
    for i, s_type in enumerate(start_ids):
        if s_type == 0 or i <= offset1:
            continue
        for j, e_type in enumerate(end_ids[i:]):
            if (i + j) > offset2:  #
                continue
            if s_type == e_type:
                tmp_ent = ''.join(raw_token[i:i + j + 1])
                predict_entities.append((tmp_ent, i))
                break

    return predict_entities


def mrc_evaluate(raw_tokens, ent_types, golden_starts, golden_ends, starts, ends, type_ids, domain):
    """
    mrc evaluater
    """
    role_metric_v1 = defaultdict()

    for raw_token, ent_type, golden_start, golden_end, start, end, type_id in zip(
            raw_tokens, ent_types, golden_starts, golden_ends, starts, ends, type_ids):

        ent_type = ent_type.item()
        start = (start * type_id).detach().numpy()
        end = (end * type_id).detach().numpy()
        golden_start = golden_start.detach().numpy()
        golden_end = golden_end.detach().numpy()

        gold_entities = mrc_decode(golden_start, golden_end, raw_token)
        # 两种测试方式
        pred_entities_v1 = mrc_decode(start, end, raw_token)
        metric_v1 = calculate_metric(gold_entities, pred_entities_v1)

        if ent_type not in role_metric_v1.keys():
            role_metric_v1[ent_type] = metric_v1
        else:
            role_metric_v1[ent_type] += metric_v1

    tp_v1, fp_v1, fn_v1 = 0, 0, 0
    for ent_type in role_metric_v1.keys():
        tp_v1 += role_metric_v1[ent_type][0]
        fp_v1 += role_metric_v1[ent_type][1]
        fn_v1 += role_metric_v1[ent_type][2]
        temp_metric = get_p_r_f(role_metric_v1[ent_type][0], role_metric_v1[ent_type][1], role_metric_v1[ent_type][2])
        metric_str = f'BASE TEST: type: {domain_entity_types_reverse[domain][ent_type]} precision: {temp_metric[0]:.4f}, ' \
                     f'recall: {temp_metric[1]:.4f}, f1: {temp_metric[2]:.4f}'
        logger.info(metric_str)

    mirco_metrics_v1 = get_p_r_f(tp_v1, fp_v1, fn_v1)

    logger.info(f'BASE TEST: [MIRCO] precision: {mirco_metrics_v1[0]:.4f}, '
                f'recall: {mirco_metrics_v1[1]:.4f}, f1: {mirco_metrics_v1[2]:.4f}')

    return mirco_metrics_v1[2]
