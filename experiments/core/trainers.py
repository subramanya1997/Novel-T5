import torch.nn as nn
import os
import torch
import math
import time
from tqdm import tqdm
from typing import TypeVar
from core.utils.tensors import to_device
TrainerType = TypeVar('TrainerType', bound='Trainer')


class T5TransformerTrainer:

    def __init__(self, model,
                 optimizer,
                 patience,
                 scheduler=None,
                 checkpoint_dir=None,
                 clip=None,
                 device='cpu'):

        self.model = model.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.checkpoint_dir = checkpoint_dir
        self.clip = clip
        self.device = device
        self.patience = patience

    def calc_val_loss(self, val_loader):

        self.model.eval()
        with torch.no_grad():
            avg_val_loss = 0

            for index, batch in enumerate(tqdm(val_loader)):
                inputs = to_device(batch[0], device=self.device)
                inputs_att = to_device(batch[1], device=self.device)
                repl_targets = to_device(batch[3], device=self.device)

                outputs = self.model(input_ids=inputs,
                                     attention_mask=inputs_att,
                                     labels=repl_targets)
                lm_loss = outputs[0]
                avg_val_loss += lm_loss.item()

            avg_val_loss = avg_val_loss / len(val_loader)
            return avg_val_loss

    def print_epoch(self, epoch, avg_train_epoch_loss, avg_val_epoch_loss,
                    cur_patience, strt):

        print("Epoch {}:".format(epoch+1))
        print("Train loss: {} | Train PPL: {}".format(
            avg_train_epoch_loss, math.exp(avg_train_epoch_loss)))
        print("Val loss: {} | Val PPL: {}".format(avg_val_epoch_loss,
              math.exp(avg_val_epoch_loss)))
        print("Patience left: {}".format(self.patience-cur_patience))
        print("Time: {} mins".format((time.time() - strt) / 60.0))
        print("++++++++++++++++++")

    def save_epoch(self, epoch, loss=None):

        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        # torch.save(self.model.state_dict(), os.path.join(
        #     self.checkpoint_dir, '{}_{}.pth'.format(epoch, 'model_checkpoint')))
        # we use the proposed method for saving T5 model
        torch.save(self.model.state_dict(), os.path.join(
            self.checkpoint_dir, 'model_checkpoint'),
                   _use_new_zipfile_serialization=False)
        #self.model.save_pretrained(os.path.join(self.checkpoint_dir,'model_checkpoint'))
        torch.save(self.optimizer.state_dict(), os.path.join(
            self.checkpoint_dir,'optimizer_checkpoint'))

    def train_step(self, batch):
        self.model.train()
        self.optimizer.zero_grad()

        inputs = to_device(batch[0], device=self.device)
        inputs_att = to_device(batch[1], device=self.device)
        pad_targets = to_device(batch[2], device=self.device)
        repl_targets = to_device(batch[3], device=self.device)
        targets_att = to_device(batch[4], device=self.device)

        outputs = self.model(input_ids=inputs,
                             attention_mask=inputs_att,
                             labels=repl_targets)
        lm_loss = outputs[0]
        # print(lm_loss)
        pred_scores = outputs[1]
        last_hidden = outputs[2]
        return lm_loss, last_hidden

    def train_epochs(self, n_epochs, train_loader, val_loader):

        best_val_loss, cur_patience  = 10000, 0

        print("Training model....")
        self.model.train()

        for epoch in range(n_epochs):
            #if cur_patience == self.patience:
            #    break

            avg_train_loss = 0
            strt = time.time()

            for index, sample_batch in enumerate(tqdm(train_loader)):

                loss, _ = self.train_step(sample_batch)
                avg_train_loss += loss.item()
                loss.backward(retain_graph=False)
                if self.clip is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                                   self.clip)
                self.optimizer.step()
            avg_train_loss = avg_train_loss / len(train_loader)
            avg_val_loss = self.calc_val_loss(val_loader)

            if avg_val_loss < best_val_loss:
                self.save_epoch(epoch)
                best_val_loss = avg_val_loss
                cur_patience = 0
            else:
                cur_patience += 1
            self.print_epoch(epoch, avg_train_loss, avg_val_loss,
                             cur_patience, strt)

    def fit(self, train_loader, val_loader, epochs):
        self.train_epochs(epochs, train_loader, val_loader)


class NovelT5Trainer:

    def __init__(self, model,
                 optimizer,
                 patience,
                 criterion,
                 auxilary_loss_weight,
                 scheduler=None,
                 checkpoint_dir=None,
                 clip=None,
                 device='cpu'):

        self.model = model.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.checkpoint_dir = checkpoint_dir
        self.clip = clip
        self.device = device
        self.patience = patience
        self.criterion = criterion
        self.aux_weight = auxilary_loss_weight

    def print_epoch(self, epoch, avg_train_epoch_loss, avg_val_epoch_loss,
                    cur_patience, strt):

        print("Epoch {}:".format(epoch+1))
        print("Train loss: {} | Train PPL: {}".format(
            avg_train_epoch_loss, math.exp(avg_train_epoch_loss)))
        print("Val loss: {} | Val PPL: {}".format(avg_val_epoch_loss,
              math.exp(avg_val_epoch_loss)))
        print("Patience left: {}".format(self.patience-cur_patience))
        print("Time: {} mins".format((time.time() - strt) / 60.0))
        print("++++++++++++++++++")

    def save_epoch(self, epoch, loss=None):
        print("------------------ save {} ------------------".format(epoch))
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        if loss == None:
          torch.save(self.model.state_dict(), os.path.join(
              self.checkpoint_dir, 'model_checkpoint_{}'.format(epoch)),
                    _use_new_zipfile_serialization=False)
        else:
          torch.save(self.model.state_dict(), os.path.join(
              self.checkpoint_dir, 'model_checkpoint_{}'.format(loss)),
                    _use_new_zipfile_serialization=False)
        # we use the proposed method for saving T5 model
        # self.model.save_pretrained(os.path.join(self.checkpoint_dir,'model_checkpoint'))
        torch.save(self.optimizer.state_dict(), os.path.join(
            self.checkpoint_dir,'optimizer_checkpoint'.format(epoch, loss)))



    def calc_val_loss(self, val_loader):

        self.model.eval()
        with torch.no_grad():
            avg_val_loss = 0
            avg_val_lm_loss = 0

            for index, batch in enumerate(tqdm(val_loader)):
                inputs = to_device(batch[0], device=self.device)
                inputs_att = to_device(batch[1], device=self.device)
                pad_targets = to_device(batch[2], device=self.device)
                repl_targets = to_device(batch[3], device=self.device)
                targets_att = to_device(batch[4], device=self.device)
                emo_label = to_device(batch[5], device=self.device)

                outputs = self.model(emolabel=emo_label, input_ids=inputs,
                                     attention_mask=inputs_att,
                                     labels=repl_targets)
                lm_loss = outputs[0]
                lm_logits = outputs[1]
                clf_logits = outputs[2]
                clf_loss = self.criterion(clf_logits, emo_label)

                avg_val_loss = avg_val_loss + lm_loss + clf_loss
                avg_val_lm_loss = avg_val_lm_loss + lm_loss


            avg_val_loss = avg_val_loss / len(val_loader)
            avg_val_lm_loss = avg_val_lm_loss / len(val_loader)
            print("avg val loss {} ,   avg val lm_loss {}".format(
                avg_val_loss, avg_val_lm_loss))
            return avg_val_lm_loss


    def train_step(self, batch):
        self.model.train()
        self.optimizer.zero_grad()

        inputs = to_device(batch[0], device=self.device)
        inputs_att = to_device(batch[1], device=self.device)
        pad_targets = to_device(batch[2], device=self.device)
        repl_targets = to_device(batch[3], device=self.device)
        targets_att = to_device(batch[4], device=self.device)
        emo_label = to_device(batch[5], device=self.device)

        outputs = self.model(emolabel=emo_label, input_ids=inputs,
                             attention_mask=inputs_att,
                             labels=repl_targets)
        lm_loss = outputs[0]
        lm_logits = outputs[1]
        clf_logits = outputs[2]
        clf_loss = self.criterion(clf_logits, emo_label)
        return lm_loss, clf_loss

    def train_epochs(self, n_epochs, train_loader, val_loader):

        best_val_loss, cur_patience = 10000, 0

        print("Training model....")
        self.model.train()

        for epoch in range(n_epochs):
            iters=0
            #if cur_patience == self.patience:
            #    break

            avg_train_loss = 0
            avg_train_lm_loss = 0

            strt = time.time()

            for index, sample_batch in enumerate(tqdm(train_loader)):

                lm_loss, clf_loss = self.train_step(sample_batch)

                loss = lm_loss + self.aux_weight*clf_loss
                avg_train_loss += loss.item()
                avg_train_lm_loss += lm_loss.item()
                loss.backward(retain_graph=False)
                iters += 1
                if self.clip is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                                   self.clip)
                self.optimizer.step()
                if iters%400==0:
                    print("lm_loss {},   clf_loss  {}".format(lm_loss.item(),
                                                              self.aux_weight*clf_loss.item()))
                    print("total loss {}".format(loss.item()))
                    print("Train lm loss {}".format(avg_train_lm_loss/iters))
            avg_train_loss = avg_train_loss / len(train_loader)
            avg_train_lm_loss = avg_train_lm_loss / len(train_loader)
            print("avg train loss {} ".format(avg_train_loss))
            avg_val_loss = self.calc_val_loss(val_loader)

            if avg_val_loss < best_val_loss:
                self.save_epoch(epoch, 'Best')
                best_val_loss = avg_val_loss
                cur_patience = 0
            else:
                cur_patience += 1
            if epoch % 10 == 0:
              self.save_epoch(epoch)
            self.print_epoch(epoch, avg_train_lm_loss, avg_val_loss,
                             cur_patience, strt)

    def fit(self, train_loader, val_loader, epochs):
        self.train_epochs(epochs, train_loader, val_loader)