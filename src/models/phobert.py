from transformers import AutoModelForTokenClassification

def get_phobert_model(tag_vocab):
    pho_bert = AutoModelForTokenClassification.from_pretrained(
        "vinai/phobert-base-v2",
        num_labels=len(tag_vocab._token_to_idx),
        id2label=tag_vocab._idx_to_token,
        label2id=tag_vocab._token_to_idx
    )
    return pho_bert