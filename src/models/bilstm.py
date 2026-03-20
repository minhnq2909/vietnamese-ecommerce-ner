import torch
import torch.nn as nn
from torchcrf import CRF


class BiLSTM_CRF(nn.Module):
  def __init__(self, vocab_size, tagset_size , pretrained_weights, hidden_size =128 ):
    super(BiLSTM_CRF, self).__init__()
    self.embedding = nn.Embedding.from_pretrained(
        embeddings=pretrained_weights,
        freeze=False,
        padding_idx = 0
    )

    self.lstm = nn.LSTM(input_size = 300,
                        hidden_size = hidden_size,
                        bidirectional = True,
                        batch_first = True)

    self.hidden2tag = nn.Linear(hidden_size*2, tagset_size)
    self.crf = CRF(tagset_size, batch_first=True)

  def forward(self, sentence, tags=None, mask=None):
    embeds = self.embedding(sentence)
    lstm_out, _ = self.lstm(embeds)
    linear_out = self.hidden2tag(lstm_out)

    if tags is not None:
      loss = -self.crf(linear_out, tags, mask=mask, reduction='mean')
      return loss
    else:
      return self.crf.decode(linear_out, mask=mask)