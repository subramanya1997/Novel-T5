import math
import torch
from tqdm import tqdm
from torch.optim import Adam
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import T5Tokenizer
from torchsummary import summary

from core.utils.parser import get_train_parser
from core.data.empdataset import EmpatheticDataset
from core.data.collators import NovelT5CollatorChat
from core.models.huggingface.novelT5 import T5Novel
from core.trainers import NovelT5Trainer


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(DEVICE)

# get args from cmdline
parser = get_train_parser()
options = parser.parse_args()

# load dataset
if options.dataset_name == "empchat":
    train_dataset = EmpatheticDataset("train", options.max_hist_len)
    val_dataset = EmpatheticDataset("valid", options.max_hist_len)
else:
    raise NotImplementedError

# make transforms
tokenizer = T5Tokenizer.from_pretrained('t5-base')
train_dataset.tokenizer_hist = tokenizer
train_dataset.tokenizer_ans = tokenizer
val_dataset.tokenizer_hist = tokenizer
val_dataset.tokenizer_ans = tokenizer

# load data
if options.dataset_name == "empchat":
    collator_fn = NovelT5CollatorChat(device='cpu')

train_loader = DataLoader(train_dataset, batch_size=options.batch_size,
                          drop_last=False, shuffle=True,
                          collate_fn=collator_fn)
val_loader = DataLoader(val_dataset, batch_size=options.batch_size,
                          drop_last=False, shuffle=True,
                          collate_fn=collator_fn)

# create model
model = T5Novel(model_version='t5-base',
                            num_classes=32,
                            device=DEVICE)

# load only pretrained lm model
if options.modelckpt is not None:
    state_dict = torch.load(options.modelckpt, map_location='cpu')
    model.load_state_dict(state_dict)
model.lm_model.config.output_hidden_states = True
model.lm_model.config.dropout_rate = 0.2
model.to(DEVICE)

# params and optimizer
numparams = sum([p.numel() for p in model.parameters()])
train_numparams = sum([p.numel() for p in model.parameters() if
                       p.requires_grad])
print('Total Parameters: {}'.format(numparams))
print('Trainable Parameters: {}'.format(train_numparams))
optimizer = Adam(
    [p for p in model.parameters() if p.requires_grad],
    lr=options.lr, weight_decay=1e-6)

if options.optimckpt is not None:
    state_dict = torch.load(options.optim, map_location='cpu')
    optimizer.load_state_dict(state_dict)

criterion = nn.CrossEntropyLoss(ignore_index=-100)
# create trainer
trainer = NovelT5Trainer(model=model,
                            optimizer=optimizer,
                            auxilary_loss_weight=options.multitask1,
                            patience=5, criterion=criterion,
                            scheduler=None,
                            checkpoint_dir=options.ckpt,
                            device=DEVICE)
# train model
trainer.fit(train_loader, val_loader, epochs=options.epochs)