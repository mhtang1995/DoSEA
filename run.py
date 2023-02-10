import os
from transformers import BertTokenizer

from src.dataloader import read_query, get_dataset, domain_entity_types_reverse
from src.mrc_model import MRCModel
from src.options import Args, init_experiment
from src.trainer import train


def jointly_trainer():
    opt = Args().get_parser()
    logger = init_experiment(opt)

    # load tokenizer
    tokenizer = BertTokenizer(os.path.join(opt.bert_dir, 'vocab.txt'))

    # read query
    ent2query = read_query(os.path.join(opt.domain_dir, 'mrc_ent2query.json'), tokenizer)

    # read data of source domains
    source_domains = opt.source_domain.split(';')
    source_features = None
    for i in range(len(source_domains)):
        if i == len(source_domains)-1:
            source_train_dataset, length = get_dataset(
                opt,
                source_domains[i],
                "train",
                tokenizer,
                ent2query,
                other_features=source_features)
        else:
            features = get_dataset(
                opt,
                source_domains[i],
                "train",
                tokenizer,
                ent2query,
                get_features=True)
            if source_features:
                source_features = source_features + features
            else:
                source_features = features

    # read data of target domain
    target_train_dataset, length = get_dataset(
        opt,
        opt.target_domain,
        "train",
        tokenizer,
        ent2query,
        up_sample=length)

    target_dev_dataset, _ = get_dataset(opt, opt.target_domain, "test", tokenizer, ent2query)

    sun = max(list(domain_entity_types_reverse[opt.target_domain].keys())) + 1

    # create model
    model = MRCModel(
        bert_dir=opt.bert_dir,
        dropout_prob=opt.dropout_prob,
        type_embedding_dim=opt.type_embedding_dim,
        entity_types=sun,
        max_len=opt.max_seq_len,
        mid_linear_dims=128
    )

    train(opt, model, source_train_dataset, target_train_dataset, length, target_dev_dataset)
    logger.info("Source Train is done")


if __name__ == '__main__':
    jointly_trainer()





