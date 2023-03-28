import math
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from transformers import T5Tokenizer, T5ForConditionalGeneration

from core.utils.parser import get_train_parser
from core.data.empdataset import EmpatheticDataset
from core.data.collators import T5CollatorChat
from core.trainers import T5TransformerTrainer
from torchsummary import summary

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


"""Uncomment the above to use map with transforms"""
# we dont use map on dataset! so transforms will be [] and HuggingFace
# tokenizers will be applied
train_dataset.tokenizer_hist = tokenizer
train_dataset.tokenizer_ans = tokenizer
val_dataset.tokenizer_hist = tokenizer
val_dataset.tokenizer_ans = tokenizer

# load data
if options.dataset_name == "empchat":
    collator_fn = T5CollatorChat(device='cpu')

train_loader = DataLoader(train_dataset, batch_size=options.batch_size,
                          drop_last=False, shuffle=True,
                          collate_fn=collator_fn)
val_loader = DataLoader(val_dataset, batch_size=options.batch_size,
                          drop_last=False, shuffle=True,
                          collate_fn=collator_fn)

# create model
if options.modelckpt is not None:
    model = T5ForConditionalGeneration.from_pretrained(options.modelckpt)
else:
    model = T5ForConditionalGeneration.from_pretrained('t5-base')
model.config.output_hidden_states = True
model.config.dropout_rate=0.2
#print(summary(model))

# params and optimizer
numparams = sum([p.numel() for p in model.parameters()])
train_numparams = sum([p.numel() for p in model.parameters() if
                       p.requires_grad])
print('Total Parameters: {}'.format(numparams))
print('Trainable Parameters: {}'.format(train_numparams))
optimizer = Adam(
    [p for p in model.parameters() if p.requires_grad],
    lr=options.lr, weight_decay=0)

if options.optimckpt is not None:
    state_dict = torch.load(options.optim, map_location='cpu')
    optimizer.load_state_dict(state_dict)


#import ipdb;ipdb.set_trace()

# create trainer
trainer = T5TransformerTrainer(model=model, optimizer=optimizer,
                                 patience=5, scheduler=None,
                                 checkpoint_dir=options.ckpt, device=DEVICE)
# train model
trainer.fit(train_loader, val_loader, epochs=options.epochs)