import logging
import time
import argparse
import os
import torch
import random
import numpy as np
from datetime import timedelta
import pickle


class Args:
    @staticmethod
    def parse():
        parser = argparse.ArgumentParser()
        return parser

    @staticmethod
    def initialize(parser: argparse.ArgumentParser):

        parser.add_argument('--gpu_ids', type=str, default='7', help='-1 for cpu, "0,1" for multi gpu')

        parser.add_argument("--train_type", type=str, default="normal")

        parser.add_argument("--task_rate", type=float, default=0.1)

        parser.add_argument("--rate", type=float, default=0.3)

        parser.add_argument("--type_embedding_dim", type=int, default=64)

        parser.add_argument("--output_dir", type=str, default="./runs/", help="Experiment saved root path")

        parser.add_argument("--input_dir", type=str, default="./runs/", help="source domain model cpkt to load")

        parser.add_argument("--bert_dir", type=str, default="bert_base_cased", help="bert model")

        parser.add_argument("--domain_dir", default="./datasets/cross_domain", help="dataset dir")

        parser.add_argument("--source_domain", default="conll03", help="source domain")

        parser.add_argument("--target_domain", default="literature", help="target domain")

        parser.add_argument("--seed", type=int, default=666, help="random seed (three seeds: 555, 666, 777)")

        parser.add_argument("--max_seq_len", type=int, default=200)

        # train parameters
        parser.add_argument('--batch_size', default=36, type=int)

        parser.add_argument('--train_epochs', default=20, type=int, help='Max training epoch')

        parser.add_argument('--dropout_prob', default=0.3, type=float, help='drop out probability')

        parser.add_argument("--lr", type=float, default=2e-5, help="Bert Learning rate")

        parser.add_argument("--other_lr", type=float, default=1e-3, help="Learning rate except Bert")

        parser.add_argument('--weight_decay', default=0.00, type=float)

        parser.add_argument('--max_grad_norm', default=1.0, type=float, help='max grad clip')

        parser.add_argument('--warmup_proportion', default=0.1, type=float)

        parser.add_argument('--adam_epsilon', default=1e-8, type=float)

        return parser

    def get_parser(self):
        parser = self.parse()
        parser = self.initialize(parser)
        return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)


def init_experiment(args):
    set_seed(args.seed)

    args.output_dir = os.path.join(args.output_dir, args.target_domain)
    args.input_dir = os.path.join(args.input_dir, args.source_domain)
    # create a logger
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    logger = create_logger(os.path.join(args.output_dir, args.train_type + '.log'))
    logger.info('============ Initialized logger ============')
    logger.info('\n'.join('%s: %s' % (k, str(v))
                          for k, v in sorted(dict(vars(args)).items())))
    logger.info('The experiment will be stored in %s\n' % args.output_dir)
    return logger


class LogFormatter(object):

    def __init__(self):
        self.start_time = time.time()

    def format(self, record):
        elapsed_seconds = round(record.created - self.start_time)

        prefix = "%s - %s - %s" % (
            record.levelname,
            time.strftime('%x %X'),
            timedelta(seconds=elapsed_seconds)
        )
        message = record.getMessage()
        message = message.replace('\n', '\n' + ' ' * (len(prefix) + 3))
        return "%s - %s" % (prefix, message) if message else ''


def create_logger(filepath):
    # create log formatter
    log_formatter = LogFormatter()

    # create file handler and set level to debug
    if filepath is not None:
        file_handler = logging.FileHandler(filepath, "a")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(log_formatter)

    # create console handler and set level to info
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(log_formatter)

    # create logger and set level to debug
    logger = logging.getLogger()
    logger.handlers = []
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    if filepath is not None:
        logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    # reset logger elapsed time
    def reset_time():
        log_formatter.start_time = time.time()

    logger.reset_time = reset_time

    return logger