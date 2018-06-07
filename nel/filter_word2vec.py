import sys
from nel.vocabulary import Vocabulary
import nel.utils as utils
import numpy as np 


if __name__ == "__main__":
    core_voca_path = sys.argv[1]
    word_embs_dir = sys.argv[2]

    print('load core voca from', core_voca_path)
    core_voca = Vocabulary.load(core_voca_path)

    print('load full voca and embs')
    full_voca, full_embs = utils.load_voca_embs(word_embs_dir + '/all_dict.word', 
                                                word_embs_dir + '/all_word_embeddings.npy')

    print('select word ids')
    selected = []
    for word in core_voca.id2word: 
        word_id = full_voca.word2id.get(word, -1)
        if word_id >= 0: 
            selected.append(word_id)

    print('save...')
    selected_embs = full_embs[selected, :]
    np.save(word_embs_dir + '/word_embeddings', selected_embs)

    with open(word_embs_dir + '/dict.word', 'w', encoding='utf8') as f:
        for i in selected: 
            f.write(full_voca.id2word[i] + '\t1000\n')
