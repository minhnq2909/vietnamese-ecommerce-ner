import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence


class Vocabulary:
    def __init__(self, token_to_idx=None, use_unk=True):
        """
        Args:
            token_to_idx (dict): a pre-existing map of tokens to indices
        """
        if token_to_idx is None:
            token_to_idx = {}
        self._token_to_idx = token_to_idx

        self._idx_to_token = {idx: token
                              for token, idx in self._token_to_idx.items()}

        self.pad_index = 0

        if use_unk:
            self.unk_index = 1
        else:
            self.unk_index = -1

    def lookup_token(self, token):
        """Retrieve the index associated with the token
          or the UNK index if token isn't present.

        Args:
            token (str): the token to look up
        Returns:
            index (int): the index corresponding to the token
        Notes:
            `unk_index` needs to be >=0 (having been added into the Vocabulary)
              for the UNK functionality
        """
        if self.unk_index >= 0:
            return self._token_to_idx.get(token, self.unk_index)
        else:
            return self._token_to_idx[token]

    def lookup_index(self, index):
        """Return the token associated with the index

        Args:
            index (int): the index to look up
        Returns:
            token (str): the token corresponding to the index
        Raises:
            KeyError: if the index is not in the Vocabulary
        """
        if index not in self._idx_to_token:
            raise KeyError("the index (%d) is not in the Vocabulary" % index)
        return self._idx_to_token[index]

    def add_token(self, token):
        """Update mapping dicts based on the token.

        Args:
            token (str): the item to add into the Vocabulary
        Returns:
            index (int): the integer corresponding to the token
        """
        if token in self._token_to_idx:
            index = self._token_to_idx[token]
        else:
            index = len(self._token_to_idx)
            self._token_to_idx[token] = index
            self._idx_to_token[index] = token
        return index

    @classmethod
    def build_vocab(cls, sequences, use_unk=True):
        """Build vocabulary from a list of sequences
        A sequence may be a sequence of words or a sequence of tags.

        Arguments:
        ----------
            sequences (list): list of sequences, each sentence list of words
            or list of tags

        Return:
        ----------
            vocab (Vocabulary): a Vocabulary object
        """
        if use_unk:
            token_to_idx = {"<PAD>": 0, "<UNK>": 1}
        else:
            token_to_idx = {"<PAD>": 0}

        vocab = cls(token_to_idx, use_unk=use_unk)
        for s in sequences:
            for word in s:
                vocab.add_token(word)
        return vocab

    def __str__(self):
        return "<Vocabulary(size=%d)>" % len(self)

    def __len__(self):
        return len(self._token_to_idx)


# -- For BiLSTM --CRF --
class NERDataset(Dataset):
  def __init__(self, df, word_vocab, tag_vocab):
    self.sentence = df['text']
    self.tag = df['tag_seg']
    self.word_vocab = word_vocab
    self.tag_vocab = tag_vocab

  def __len__(self):
    return len(self.sentence)

  def __getitem__(self, idx):
    sentence = self.sentence[idx]
    tag = self.tag[idx]
    word_idx = [self.word_vocab.lookup_token(word) for word in sentence.split()]
    tag_idx = [self.tag_vocab.lookup_token(t) for t in tag]
    return  torch.tensor(word_idx),  torch.tensor(tag_idx)


def pad_collate_fn(batch):
    words_list = [item[0] for item in batch]
    tags_list = [item[1] for item in batch]

    lengths = torch.tensor([len(w) for w in words_list])

    padded_words = pad_sequence(words_list, batch_first=True, padding_value=0)
    padded_tags = pad_sequence(tags_list, batch_first=True, padding_value=0)

    return padded_words, padded_tags, lengths



# -- For PhoBERT --
def tokenize_and_align_labels(df, tokenizer, tag_vocab):
    batch_input_ids = []
    batch_attention_mask = []
    batch_labels = []

    for i in range(len(df["text"])):

        words = df["text"][i].split()

        ner_tags = df["tag_seg"][i]
        if isinstance(ner_tags, str):
            import ast
            ner_tags = ast.literal_eval(ner_tags)

        input_ids = [tokenizer.cls_token_id]
        label_ids = [-100] # Token <s>

        for word, label in zip(words, ner_tags):
            sub_words = tokenizer.tokenize(word)
            if not sub_words: # Bỏ qua nếu từ trống
                continue

            sub_word_ids = tokenizer.convert_tokens_to_ids(sub_words)
            input_ids.extend(sub_word_ids)

            label_ids.append(tag_vocab._token_to_idx[label])
            label_ids.extend([-100] * (len(sub_words) - 1))


        input_ids.append(tokenizer.sep_token_id)
        label_ids.append(-100)

        attention_mask = [1] * len(input_ids)

        batch_input_ids.append(input_ids)
        batch_attention_mask.append(attention_mask)
        batch_labels.append(label_ids)

    return {
        "input_ids": batch_input_ids,
        "attention_mask": batch_attention_mask,
        "labels": batch_labels
    }
