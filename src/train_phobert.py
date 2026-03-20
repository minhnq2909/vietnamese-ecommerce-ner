import argparse
from transformers import AutoTokenizer, Trainer, TrainingArguments, DataCollatorForTokenClassification
from datasets import Dataset

from configs import config
from data.preprocess import load_and_merge_data, get_train_val_test_splits
from data.dataset import Vocabulary, tokenize_and_align_labels
from models.phobert import get_phobert_model
from evaluation.metric import compute_metrics

def main():
    # 1. Load and preprocess data
    df = load_and_merge_data(config.DATA_PATH_1, config.DATA_PATH_2, config.DATA_PATH_3)
    train_df, val_df, test_df = get_train_val_test_splits(df)

    # 2. Build Vocabulary
    tag_vocab = Vocabulary.build_vocab(train_df['tag_seg'])
    
    # 3. Tokenize data using PhoBERT tokenizer
    train_data = Dataset.from_pandas(train_df)
    val_data = Dataset.from_pandas(val_df)
    test_data = Dataset.from_pandas(test_df)
    tokenizer = AutoTokenizer.from_pretrained(config.PHOBERT_MODEL_NAME)
    
    # Map tokenization
    tokenized_train = train_data.map(lambda x: tokenize_and_align_labels(x, tokenizer, tag_vocab), batched=True)
    tokenized_val = val_data.map(lambda x: tokenize_and_align_labels(x, tokenizer, tag_vocab), batched=True)
    tokenized_test = test_data.map(lambda x: tokenize_and_align_labels(x, tokenizer, tag_vocab), batched=True)
    # Remove original columns to avoid confusion with tokenized data
    cols_to_remove = train_data.column_names
    tokenized_train = tokenized_train.remove_columns(cols_to_remove)
    tokenized_val = tokenized_val.remove_columns(cols_to_remove)
    tokenized_test = tokenized_test.remove_columns(cols_to_remove)
    # 4. Load pre-trained PhoBERT model
    model = get_phobert_model(
        config.PHOBERT_MODEL_NAME, 
        num_labels=len(tag_vocab), 
        id2label=tag_vocab._idx_to_token, 
        label2id=tag_vocab._token_to_idx
    )

    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    # 5. Setup TrainingArguments
    training_args = TrainingArguments(
        output_dir="./checkpoints",
        evaluation_strategy="epoch",
        learning_rate=config.LEARNING_RATE,
        per_device_train_batch_size=config.BATCH_SIZE,
        num_train_epochs=config.EPOCHS,
        weight_decay=0.01,
        load_best_model_at_end=True,      # Save the best
        metric_for_best_model="f1",       # Best metric f1
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=lambda p: compute_metrics(p, list(tag_vocab._token_to_idx.keys()))
    )

    # 6. Train the model
    trainer.train()
    # 7. Evaluate on test set
    # Evaluate on test set
    test_results = trainer.predict(tokenized_test)
    metrics = test_results.metrics

    print(f"Precision: {metrics['test_precision']:.4f}")
    print(f"Recall: {metrics['test_recall']:.4f}")
    print(f"F1-Score: {metrics['test_f1']:.4f}")
if __name__ == "__main__":
    main()
    