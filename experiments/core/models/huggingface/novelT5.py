import torch.nn as nn

from transformers import T5ForConditionalGeneration


class T5Novel(nn.Module):
    """
    This is T5 model with 2 heads.
    An LM head + a classification head
    """
    def __init__(self, model_version, num_classes=3, device='cpu'):
        super(T5Novel, self).__init__()
        self.num_classes = num_classes
        self.lm_model = T5ForConditionalGeneration.from_pretrained(model_version)
        # self.clf_layer = nn.Linear(in_features=self.lm_model.config.d_model,
        #                            out_features=num_classes)
        self.clf_enc = nn.Sequential(
            nn.Linear(in_features=self.lm_model.config.d_model,
                      out_features=300,bias=True),
            # nn.Dropout(0.1),
            nn.BatchNorm1d(300),  # applying batch norm
            nn.ReLU(),
            #nn.GELU(),
            nn.Linear(in_features=300,
                      out_features=self.num_classes)

        )
        self.device = device

    def forward(self, *args, **kwargs):

        emo_label = kwargs['emolabel']
        kwargs.pop('emolabel', None)
        outputs = self.lm_model(**kwargs, output_hidden_states=True,
                                return_dict=True)
        lm_loss = outputs['loss']
        lm_logits = outputs['logits']
        dec_hidden_states = outputs['decoder_hidden_states']
        enc_last_hidden = outputs['encoder_last_hidden_state']
        enc_hidden_states = outputs['encoder_hidden_states']
        last_dec_hidden = dec_hidden_states[-1]
        enc_last_hidden_last_timestep = enc_last_hidden[:,-1,:]
        clf_logits = self.clf_enc(enc_last_hidden_last_timestep)
        return lm_loss, lm_logits, clf_logits