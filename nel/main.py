import nel.dataset as D
from nel.mulrel_ranker import MulRelRanker
from nel.ed_ranker import EDRanker

import nel.utils as utils

from pprint import pprint

import argparse

parser = argparse.ArgumentParser()


datadir = 'data/generated/test_train_data/'
conll_path = 'data/basic_data/test_datasets/'
person_path = 'data/basic_data/p_e_m_data/persons.txt'
voca_emb_dir = 'data/generated/embeddings/word_ent_embs/'

ModelClass = MulRelRanker


# general args
parser.add_argument("--mode", type=str,
                    help="train or eval",
                    default='train')
parser.add_argument("--model_path", type=str,
                    help="model path to save/load",
                    default='')

# args for preranking (i.e. 2-step candidate selection)
parser.add_argument("--n_cands_before_rank", type=int,
                    help="number of candidates",
                    default=30)
parser.add_argument("--prerank_ctx_window", type=int,
                    help="size of context window for the preranking model",
                    default=50)
parser.add_argument("--keep_p_e_m", type=int,
                    help="number of top candidates to keep w.r.t p(e|m)",
                    default=4)
parser.add_argument("--keep_ctx_ent", type=int,
                    help="number of top candidates to keep w.r.t using context",
                    default=4)

# args for local model
parser.add_argument("--ctx_window", type=int,
                    help="size of context window for the local model",
                    default=100)
parser.add_argument("--tok_top_n", type=int,
                    help="number of top contextual words for the local model",
                    default=25)


# args for global model
parser.add_argument("--mulrel_type", type=str,
                    help="type for multi relation (rel-norm or ment-norm)",
                    default='ment-norm')
parser.add_argument("--n_rels", type=int,
                    help="number of relations",
                    default=5)
parser.add_argument("--hid_dims", type=int,
                    help="number of hidden neurons",
                    default=100)
parser.add_argument("--snd_local_ctx_window", type=int,
                    help="local ctx window size for relation scores",
                    default=6)
parser.add_argument("--dropout_rate", type=float,
                    help="dropout rate for relation scores",
                    default=0.3)


# args for training
parser.add_argument("--n_epochs", type=int,
                    help="max number of epochs",
                    default=200)
parser.add_argument("--dev_f1_change_lr", type=float,
                    help="dev f1 to change learning rate",
                    default=0.915)
parser.add_argument("--n_not_inc", type=int,
                    help="number of evals after dev f1 not increase",
                    default=10)
parser.add_argument("--eval_after_n_epochs", type=int,
                    help="number of epochs to eval",
                    default=5)
parser.add_argument("--learning_rate", type=float,
                    help="learning rate",
                    default=1e-4)
parser.add_argument("--margin", type=float,
                    help="margin",
                    default=0.01)

# args for LBP
parser.add_argument("--df", type=float,
                    help="dumpling factor (for LBP)",
                    default=0.5)
parser.add_argument("--n_loops", type=int,
                    help="number of LBP loops",
                    default=10)

# args for debugging
parser.add_argument("--print_rel", action='store_true')
parser.add_argument("--print_incorrect", action='store_true')


args = parser.parse_args()


if __name__ == "__main__":
    print('load conll at', datadir)
    conll = D.CoNLLDataset(datadir, person_path, conll_path)

    print('create model')
    word_voca, word_embeddings = utils.load_voca_embs(voca_emb_dir + 'dict.word',
                                                      voca_emb_dir + 'word_embeddings.npy')
    print('word voca size', word_voca.size())
    snd_word_voca, snd_word_embeddings = utils.load_voca_embs(voca_emb_dir + '/glove/dict.word',
                                                              voca_emb_dir + '/glove/word_embeddings.npy')
    print('snd word voca size', snd_word_voca.size())

    entity_voca, entity_embeddings = utils.load_voca_embs(voca_emb_dir + 'dict.entity',
                                                          voca_emb_dir + 'entity_embeddings.npy')
    config = {'hid_dims': args.hid_dims,
              'emb_dims': entity_embeddings.shape[1],
              'freeze_embs': True,
              'tok_top_n': args.tok_top_n,
              'margin': args.margin,
              'word_voca': word_voca,
              'entity_voca': entity_voca,
              'word_embeddings': word_embeddings,
              'entity_embeddings': entity_embeddings,
              'snd_word_voca': snd_word_voca,
              'snd_word_embeddings': snd_word_embeddings,
              'dr': args.dropout_rate,
              'args': args}

    if ModelClass == MulRelRanker:
        config['df'] = args.df
        config['n_loops'] = args.n_loops
        config['n_rels'] = args.n_rels
        config['mulrel_type'] = args.mulrel_type
    else:
        raise Exception('unknown model class')

    pprint(config)
    ranker = EDRanker(config=config)

    dev_datasets = [('aida-A', conll.testA),
                    ('aida-B', conll.testB),
                    ('msnbc', conll.msnbc),
                    ('aquaint', conll.aquaint),
                    ('ace2004', conll.ace2004),
                    ('clueweb', conll.clueweb),
                    ('wikipedia', conll.wikipedia)
                    ]

    if args.mode == 'train':
        print('training...')
        config = {'lr': args.learning_rate, 'n_epochs': args.n_epochs}
        pprint(config)
        ranker.train(conll.train, dev_datasets, config)

    elif args.mode == 'eval':
        org_dev_datasets = dev_datasets  # + [('aida-train', conll.train)]
        dev_datasets = []
        for dname, data in org_dev_datasets:
            dev_datasets.append((dname, ranker.get_data_items(data, predict=True)))
            print(dname, '#dev docs', len(dev_datasets[-1][1]))

        vecs = ranker.model.rel_embs.cpu().data.numpy()

        for di, (dname, data) in enumerate(dev_datasets):
            ranker.model._coh_ctx_vecs = []
            predictions = ranker.predict(data)
            print(dname, utils.tokgreen('micro F1: ' + str(D.eval(org_dev_datasets[di][1], predictions))))
