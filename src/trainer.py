
import copy
import json
import logging

import torch
from torch.utils.data import DataLoader, RandomSampler

from src.utils import load_model_and_parallel, build_optimizer_and_scheduler, save_model
from src.evaluater import get_p_r_f, mrc_decode, calculate_metric
from collections import defaultdict
from src.dataloader import domain_entity_types_reverse

logger = logging.getLogger(__name__)


def train(opt, model, source_train_dataset, target_train_dataset, length, dev_dataset):
    best_f1_v1 = 0
    best_step_v1 = 0

    source_train_sampler = RandomSampler(source_train_dataset)
    target_train_sampler = RandomSampler(target_train_dataset)

    target_train_loader = DataLoader(
        dataset=target_train_dataset,
        batch_size=opt.batch_size,
        sampler=target_train_sampler,
        num_workers=0
    )

    source_train_loader = DataLoader(
        dataset=source_train_dataset,
        batch_size=opt.batch_size * 2,
        sampler=source_train_sampler,
        num_workers=0
    )

    model, device = load_model_and_parallel(model, opt.gpu_ids)

    t_total = len(target_train_loader) * opt.train_epochs

    optimizer = build_optimizer_and_scheduler(opt, model)

    # Train
    logger.info("***** Running training *****")
    logger.info(f"  Num Epochs = {opt.train_epochs}")
    logger.info(f"  Total source training size = {source_train_dataset.nums}")
    logger.info(f"  Total target training size = {target_train_dataset.nums}")
    logger.info(f"  Total target training one epoch = {len(target_train_loader)}")
    logger.info(f"  Total source training one epoch = {len(source_train_loader)}")
    logger.info(f"  Total optimization steps = {t_total}")

    global_step = 0

    log_loss_steps = 20
    test_steps = int(length / opt.batch_size)
    avg_loss = 0.

    for epoch in range(opt.train_epochs):

        for source_batch_data, target_batch_data in zip(source_train_loader, target_train_loader):
            model.train()
            # target
            loss = model(
                token_ids=target_batch_data['token_ids'].to(device),
                attention_masks=target_batch_data['attention_masks'].to(device),
                token_type_ids=target_batch_data['token_type_ids'].to(device),
                start_ids=target_batch_data['start_ids'].to(device),
                end_ids=target_batch_data['end_ids'].to(device),
                ent_type=target_batch_data['ent_type'].to(device),
                contain_entity=target_batch_data['contain_entity'].to(device),
                cla=torch.tensor(1).to(device),
                task_rate=torch.tensor(opt.task_rate).to(device),
            )

            if len(opt.gpu_ids) > 1:
                loss = loss.mean()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), opt.max_grad_norm)
            optimizer.step()
            global_step += 1
            if global_step % log_loss_steps == 0:
                avg_loss /= log_loss_steps
                logger.info('Step: %d / %d ----> total loss: %.5f' % (global_step, t_total, avg_loss))
                avg_loss = 0.
            else:
                avg_loss += loss.item()

            if global_step % test_steps == 0:
                f1_v1 = validate(opt, model, dev_dataset, device)
                if best_f1_v1 < f1_v1:
                    logger.info('BASE TEST: find a better model in step %d, f1 is %f' % (global_step, f1_v1))
                    # best_model = copy.deepcopy(model)
                    best_f1_v1 = f1_v1
                    best_step_v1 = global_step
                else:
                    logger.info('BASE TEST: no better model find，f1 is %f, best_f1 is %f, best_step is %d'
                                % (f1_v1, best_f1_v1, best_step_v1))

            # source
            model.train()
            # 对数据进行分离
            split_source_batch_data = {}
            size = None
            for key, value in source_batch_data.items():
                if key != 'raw_tokens':
                    split_source_batch_data[key] = value.split(opt.batch_size, dim=0)
                    if size is None:
                        size = len(split_source_batch_data[key])

            for i in range(0, size):
                batch_data = {}
                for key, value in split_source_batch_data.items():
                    batch_data[key] = value[i]
                loss = model(
                    token_ids=batch_data['token_ids'].to(device),
                    attention_masks=batch_data['attention_masks'].to(device),
                    token_type_ids=batch_data['token_type_ids'].to(device),
                    start_ids=batch_data['start_ids'].to(device),
                    end_ids=batch_data['end_ids'].to(device),
                    ent_type=batch_data['ent_type'].to(device),
                    contain_entity=batch_data['contain_entity'].to(device),
                    rate=torch.tensor(opt.rate).to(device)
                )

                if len(opt.gpu_ids) > 1:
                    loss = loss.mean()
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), opt.max_grad_norm)
                optimizer.step()
    torch.cuda.empty_cache()
    logger.info('Train done')


def validate(opt, model, dataset, device):
    model.eval()

    role_metric_v1 = defaultdict()
    tp, fp, fn = 0, 0, 0
    loader = DataLoader(dataset=dataset, batch_size=36, num_workers=0)

    for step, batch_data in enumerate(loader):

        start_logits, end_logits, pred = model(
            token_ids=batch_data['token_ids'].to(device),
            attention_masks=batch_data['attention_masks'].to(device),
            token_type_ids=batch_data['token_type_ids'].to(device),
            ent_type=batch_data['ent_type'].to(device),
        )
        pred = pred.detach().cpu()
        start_logits = start_logits.detach().cpu()
        end_logits = end_logits.detach().cpu()
        ent_types = batch_data['ent_type'].detach().cpu()
        raw_tokens = list(map(list, zip(*batch_data['raw_tokens'])))
        token_type_ids = batch_data['token_type_ids'].detach().cpu()

        if pred is not None:
            for p, t in zip(pred, batch_data['contain_entity']):
                if p.item() >= 0.5 and t.item() == 1:
                    tp += 1
                elif p.item() >= 0.5 and t.item() != 1:
                    fp += 1
                elif p.item() < 0.5 and t.item() == 1:
                    fn += 1

        for raw_token, ent_type, golden_start, golden_end, start, end, token_id in zip(
                raw_tokens, ent_types, batch_data['start_ids'], batch_data['end_ids'],
                start_logits, end_logits, token_type_ids):
            ent_type = ent_type.item()
            # 0.5 is threshold
            start = (start * token_id).detach().numpy()
            start[start >= 0.5] = 1
            start[start < 0.5] = 0
            end = (end * token_id).detach().numpy()
            end[end >= 0.5] = 1
            end[end < 0.5] = 0

            golden_start = (golden_start * token_id).detach().numpy()
            golden_end = (golden_end * token_id).detach().numpy()

            gold_entities = mrc_decode(golden_start, golden_end, raw_token)

            pred_entities_v1 = mrc_decode(start, end, raw_token)
            metric_v1 = calculate_metric(gold_entities, pred_entities_v1)

            if ent_type not in role_metric_v1.keys():
                role_metric_v1[ent_type] = metric_v1
            else:
                role_metric_v1[ent_type] += metric_v1

    cla = get_p_r_f(tp, fp, fn)
    class_str = f'[CLASS] precision: {cla[0]:.4f}, recall: {cla[1]:.4f}, f1: {cla[2]:.4f}'
    logger.info(class_str)

    tp_v1, fp_v1, fn_v1 = 0, 0, 0
    for ent_type in role_metric_v1.keys():
        tp_v1 += role_metric_v1[ent_type][0]
        fp_v1 += role_metric_v1[ent_type][1]
        fn_v1 += role_metric_v1[ent_type][2]
        temp_metric = get_p_r_f(role_metric_v1[ent_type][0], role_metric_v1[ent_type][1], role_metric_v1[ent_type][2])
        metric_str = f'BASE TEST: type: {domain_entity_types_reverse[opt.target_domain][ent_type]} ' \
                     f'precision: {temp_metric[0]:.4f}, ' \
                     f'recall: {temp_metric[1]:.4f}, f1: {temp_metric[2]:.4f}'
        logger.info(metric_str)

    mirco_metrics_v1 = get_p_r_f(tp_v1, fp_v1, fn_v1)

    logger.info(f'BASE TEST: [MIRCO] precision: {mirco_metrics_v1[0]:.4f}, '
                f'recall: {mirco_metrics_v1[1]:.4f}, f1: {mirco_metrics_v1[2]:.4f}')

    return mirco_metrics_v1[2]



