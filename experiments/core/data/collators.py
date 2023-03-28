import torch
from torch.nn.utils.rnn import pad_sequence

from core.utils.masks import pad_mask
#from core.utils.tensors import mktensor

class T5CollatorChat(object):
    def __init__(self, pad_indx=0, device='cpu'):
        self.pad_indx = pad_indx
        self.device = device

    def replace_pad_labels(self,mytensor,value):
        tmp = mytensor.clone()
        tmp[mytensor==0] = value
        return tmp

    def __call__(self, batch):
        inputs, targets, labels = map(list, zip(*batch))

        input_lengths = torch.tensor(
            [len(s) for s in inputs], device=self.device)
        targets_lengths = torch.tensor(
            [len(s) for s in targets], device=self.device)
        # attention mask
        max_length = max(input_lengths)
        inputs_pad_mask = pad_mask(input_lengths, max_length=max_length,
                                   device=self.device)
        max_length = max(targets_lengths)
        targets_pad_mask = pad_mask(targets_lengths, max_length=max_length,
                                   device=self.device)
        # Pad inputs and targets
        padded_inputs = (
            pad_sequence(inputs, batch_first=True,
                         padding_value=self.pad_indx)
                .to(self.device))
        padded_targets = (
            pad_sequence(targets, batch_first=True,
                         padding_value=self.pad_indx)
                .to(self.device))
        replaced_targ = self.replace_pad_labels(padded_targets, -100)
        return padded_inputs, inputs_pad_mask, padded_targets,replaced_targ, \
               targets_pad_mask


class NovelT5CollatorChat(object):
    def __init__(self, pad_indx=0, device='cpu'):
        self.pad_indx = pad_indx
        self.device = device

    def replace_pad_labels(self,mytensor,value):
        tmp = mytensor.clone()
        tmp[mytensor==0] = value
        return tmp

    def __call__(self, batch):
        inputs, targets, labels = map(list, zip(*batch))

        input_lengths = torch.tensor(
            [len(s) for s in inputs], device=self.device)
        targets_lengths = torch.tensor(
            [len(s) for s in targets], device=self.device)
        # attention mask
        max_length = max(input_lengths)
        inputs_pad_mask = pad_mask(input_lengths, max_length=max_length,
                                   device=self.device)
        max_length = max(targets_lengths)
        targets_pad_mask = pad_mask(targets_lengths, max_length=max_length,
                                   device=self.device)
        # Pad inputs and targets
        padded_inputs = (
            pad_sequence(inputs, batch_first=True,
                         padding_value=self.pad_indx)
                .to(self.device))
        padded_targets = (
            pad_sequence(targets, batch_first=True,
                         padding_value=self.pad_indx)
                .to(self.device))
        replaced_targ = self.replace_pad_labels(padded_targets, -100)
        emo_labels = torch.tensor(labels, dtype=torch.long)
        return padded_inputs, inputs_pad_mask, padded_targets,replaced_targ, \
               targets_pad_mask,emo_labels

class PadCollate():
    def __init__(self, eos_id):
        self.eos_id = eos_id
        
    def pad_collate(self, batch):
        input_ids, token_type_ids, labels =[], [], []
        for idx, seqs in enumerate(batch):
            input_ids.append(torch.LongTensor(seqs[0]))
            token_type_ids.append(torch.LongTensor(seqs[0]))
            labels.append(torch.LongTensor(seqs[2]))
            
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=self.eos_id)
        token_type_ids = torch.nn.utils.rnn.pad_sequence(token_type_ids, batch_first=True, padding_value=self.eos_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)
    
        return input_ids, token_type_ids, labels