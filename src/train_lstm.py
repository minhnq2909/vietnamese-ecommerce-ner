import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from configs import config
from data.preprocess import load_and_merge_data, get_train_val_test_splits
from data.dataset import Vocabulary, NERDataset, pad_collate_fn
from utils.phow2v_embedding import load_phow2v_matrix
from models.bilstm_crf import BiLSTM_CRF
from sklearn.metrics import confusion_matrix
from seqeval.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 1. Load and preprocess data
    df = load_and_merge_data(config.DATA_PATH_1, config.DATA_PATH_2, config.DATA_PATH_3)
    train_df, val_df, test_df = get_train_val_test_splits(df)

    # 2. Build Vocabulary
    word_vocab = Vocabulary.build_vocab(train_df['text'].apply(lambda x: x.split()))
    tag_vocab = Vocabulary.build_vocab(train_df['tag_seg'])

    # 3.  DataLoader
    train_dataset = NERDataset(train_df, word_vocab, tag_vocab)
    train_dataloader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, collate_fn=pad_collate_fn)
    val_dataset = NERDataset(val_df, word_vocab, tag_vocab)
    val_dataloader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, collate_fn=pad_collate_fn)
    test_dataset = NERDataset(test_df, word_vocab, tag_vocab)
    test_dataloader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False, collate_fn=pad_collate_fn)

    
    # 4. Load pre-trained PhoW2V embedding
    
    pretrained_weights = load_phow2v_matrix(
        word_vocab._token_to_idx, 
        config.PHOW2V_PATH, 
        config.EMBEDDING_DIM
    )

    # 5.  BiLSTM-CRF model
    model = BiLSTM_CRF(
        vocab_size=len(word_vocab), 
        tagset_size=len(tag_vocab), 
        pretrained_weights=pretrained_weights, 
        hidden_size=128
    )
    
    # 6. Setup Train
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 7. Train loop
    def training_loop(epochs, model, optimizer, train_dataloader, valid_dataloader, test_dataloader, device):
        best_val_loss = float('inf')
        best_model_state = None
        model.to(device)
        for epoch in range(epochs):
            running_loss = 0.0
            model.train()
            progress_bar = tqdm(train_dataloader, desc=f'Epoch {epoch+1}/{epochs}',leave=False)
            for words, tags, lengths in progress_bar:
                words = words.to(device)
                tags = tags.to(device)
                mask = (words != 0).bool()
                optimizer.zero_grad()
                loss = model(words, tags, mask = mask)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                progress_bar.set_postfix({'Loss': f"{loss.item():.4f}"})

            avg_train_loss = running_loss / len(train_dataloader)

    # Validation
            model.eval()
            val_loss = 0.0
            all_true_labels = []
            all_pred_labels = []

            with torch.no_grad():
                for words, tags, lengths in valid_dataloader:
                    words = words.to(device)
                    tags = tags.to(device)
                    mask = (words != 0).bool()
                    tag_scores = model(words, mask = mask)
        # loss
                    loss = model(words, tags, mask = mask)
                    val_loss += loss.item()
        # accuracy
                    for i in range(words.size(0)):
                        length = lengths[i]
                        true_labels_idx = tags[i][:length].tolist()
                        pred_labels_idx = tag_scores[i]

                        true_label = [tag_vocab.lookup_index(idx) for idx in true_labels_idx]
                        pred_label = [tag_vocab.lookup_index(idx) for idx in pred_labels_idx]

                        all_true_labels.append(true_label)
                        all_pred_labels.append(pred_label)

            avg_val_loss =   val_loss / len(valid_dataloader)
            val_acc = accuracy_score(all_true_labels, all_pred_labels)
            val_precision = precision_score(all_true_labels, all_pred_labels)
            val_recall = recall_score(all_true_labels, all_pred_labels)
            val_f1 = f1_score(all_true_labels, all_pred_labels)

            print(f"Epoch [{epoch+1}/{epochs}] | "
                f" Train Loss: {avg_train_loss:.4f} | "
                f" Val Loss: {avg_val_loss:.4f} | "
                f" Val Acc: {val_acc:.4f} |"
                f" Val Precision: {val_precision:.4f} | Val Recall: {val_recall:.4f} | Val F1: {val_f1:.4f}")

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_model_state =  model.state_dict()

  # Test
        model.load_state_dict(best_model_state)
        model.eval()
        all_true_labels = []
        all_pred_labels = []

  ##
        all_true = []
        all_pred = []
        with torch.no_grad():
            for words, tags, lengths in test_dataloader:
                words = words.to(device)
                tags = tags.to(device)
                mask = (words != 0).bool()
                tag_scores = model(words, mask = mask)
                for i in range(words.size(0)):
                    length = lengths[i]
                    true_labels_idx = tags[i][:length].tolist()
                    pred_labels_idx = tag_scores[i]
                    true_label = [tag_vocab.lookup_index(idx) for idx in true_labels_idx]
                    pred_label = [tag_vocab.lookup_index(idx) for idx in pred_labels_idx]
                    all_true_labels.append(true_label)
                    all_pred_labels.append(pred_label)
                    all_true.extend(true_label)
                    all_pred.extend(pred_label)
        test_acc = accuracy_score(all_true_labels, all_pred_labels)
        test_precision = precision_score(all_true_labels, all_pred_labels)
        test_recall = recall_score(all_true_labels, all_pred_labels)
        test_f1 = f1_score(all_true_labels, all_pred_labels)
        cm = confusion_matrix(all_true, all_pred, labels = label_list)
        report = classification_report(all_true_labels, all_pred_labels)
        result = {
            "test_acc": test_acc,
            "test_precision": test_precision,
            "test_recall": test_recall,
            "test_f1": test_f1,
            "report": report,
            "confusion_matrix": cm
            }
        return model, result




if __name__ == "__main__":
    main()