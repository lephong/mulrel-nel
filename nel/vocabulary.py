import re
import io

LOWER = False
DIGIT_0 = False
UNK_TOKEN = "#UNK#"

BRACKETS = {"-LCB-": "{", "-LRB-": "(", "-LSB-": "[", "-RCB-": "}", "-RRB-": ")", "-RSB-": "]"}


class Vocabulary:
    unk_token = UNK_TOKEN

    def __init__(self):
        self.word2id = {}
        self.id2word = []
        self.counts = []
        self.unk_id = 0

    @staticmethod
    def normalize(token, lower=LOWER, digit_0=DIGIT_0):
        if token in [Vocabulary.unk_token, "<s>", "</s>"]:
            return token
        elif token in BRACKETS:
            token = BRACKETS[token]
        else:
            if digit_0:
                token = re.sub("[0-9]", "0", token)

        if lower:
            return token.lower()
        else:
            return token

    @staticmethod
    def load(path):
        voca = Vocabulary()
        voca.load_from_file(path)
        return voca

    def load_from_file(self, path):
        self.word2id = {}
        self.id2word = []
        self.counts = []

        f = io.open(path, "r", encoding='utf-8', errors='ignore')
        for line in f:
            line = line.strip()
            comps = line.split('\t')
            if len(comps) == 0 or len(comps) > 2: 
                raise Exception('sthing wrong')

            token = Vocabulary.normalize(comps[0].strip())
            self.id2word.append(token)
            self.word2id[token] = len(self.id2word) - 1

            if len(comps) == 2:
                self.counts.append(float(comps[1]))
            else: 
                self.counts.append(1)

        f.close()

        if Vocabulary.unk_token not in self.word2id:
            self.id2word.append(Vocabulary.unk_token)
            self.word2id[Vocabulary.unk_token] = len(self.id2word) - 1
            self.counts.append(1)
            
        self.unk_id = self.word2id[Vocabulary.unk_token]

    def size(self):
        return len(self.id2word)

    def get_id(self, token):
        tok = Vocabulary.normalize(token)
        return self.word2id.get(tok, self.unk_id)
