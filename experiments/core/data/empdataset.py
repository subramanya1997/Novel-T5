import os
import numpy as np
#from core.utils.tensors import mktensor
from torch.utils.data import Dataset
import torch

class EmpatheticDataset(Dataset):

    def __init__(self, splitname, maxhistorylen, tokenizer_hist=None,
                 tokenizer_ans=None):
        self.csvfile = os.path.join("data/empatheticdialogues",
                                    f"{splitname}.csv")
        self.maxhistorylen = maxhistorylen

        self.data, self.ids = self.read_data()
        self.label2idx, self.idx2label = self.get_labels_dict()
        self.transforms = []
        # we use different tokenizers for context and answers in case its
        # needed!
        self.tokenizer_hist = tokenizer_hist
        self.tokenizer_ans = tokenizer_ans


    def read_data(self):
        data = []
        ids = []

        history = []
        lines = open(self.csvfile).readlines()

        for i in range(1, len(lines)):
            cparts = lines[i - 1].strip().split(",")
            sparts = lines[i].strip().split(",")
            if cparts[0] == sparts[0]:
                prevsent = cparts[5].replace("_comma_", ",")
                history.append(prevsent)
                utt_idx = int(sparts[1])

                if utt_idx%2 == 0:
                    # we have an answer from listener so we store!
                    prev_hist = " </s> ".join(history[-self.maxhistorylen:])
                    answer = sparts[5].replace("_comma_", ",")
                    emolabel = sparts[2]
                    data.append((prev_hist, answer, emolabel))
                    ids.append((sparts[0], sparts[1]))

            else:
                # we have new conversation so empty history
                history = []
        return data[:10000], ids[:10000]

    def get_labels_dict(self):
        label2idx = {}
        idx2label = {}
        counter = 0
        for data in self.data:
            history, ans, label = data
            if label not in label2idx:
                label2idx[label] = counter
                idx2label[counter] = label
                counter += 1
        return label2idx, idx2label

    def bert_transform_data(self, tokenize):
        alldata = []
        for data in self.data:
            history, answer, label = data
            new_hist = tokenize(history)
            new_ans = tokenize(answer)
            label_idx = self.label2idx[label]
            alldata.append((new_hist, new_ans, label_idx))
        self.data = alldata

    def map(self, t):
        self.transforms.append(t)
        return self

    def word_counts(self, tokenizer=None):
        voc_counts = {}
        for question, answer, label in self.data:
            if tokenizer is None:
                words, counts = np.unique(np.array(question.split(' ')),
                                          return_counts=True)
            else:
                words, counts = np.unique(np.array(tokenizer(question)),
                                          return_counts=True)
            for word, count in zip(words, counts):
                if word not in voc_counts.keys():
                    voc_counts[word] = count
                else:
                    voc_counts[word] += count

            if tokenizer is None:
                words, counts = np.unique(np.array(answer.split(' ')),
                                          return_counts=True)
            else:
                words, counts = np.unique(np.array(tokenizer(answer)),
                                          return_counts=True)
            for word, count in zip(words, counts):
                if word not in voc_counts.keys():
                    voc_counts[word] = count
                else:
                    voc_counts[word] += count

        return voc_counts

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        hist, ans, label = self.data[index]
        if self.transforms == []:
            hist = self.tokenizer_hist(hist)
            ans = self.tokenizer_ans(ans)
            hist = torch.tensor(hist['input_ids'],dtype=torch.long)
            ans = torch.tensor(ans['input_ids'],dtype=torch.long)
        else:
            for t in self.transforms:
                hist = t(hist)
                ans = t(ans)
        label = torch.tensor(self.label2idx[label])
        return hist, ans, label

    def getid(self, index):
        return self.ids[index]


if __name__ == "__main__":

    train_dataset = EmpatheticDataset('train', 10)