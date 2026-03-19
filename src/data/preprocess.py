import pandas as pd
import json
from sklearn.model_selection import train_test_split

def load_and_merge_data(path1, path2, path3):
    df1 = pd.read_csv(path1).drop(columns=['annotation_id', 'annotator', 'created_at','id','updated_at','lead_time'])
    df2 = pd.read_csv(path2).drop(columns=['annotation_id', 'annotator', 'created_at','id','updated_at','lead_time'])
    df3 = pd.read_csv(path3).drop(columns=['annotation_id', 'annotator', 'created_at','id','updated_at','lead_time'])
    return pd.concat([df1, df2, df3], ignore_index=True)

def tagging(full_text, annotation):

  words = []
  word_index = []
  current_index = 0
  for word in full_text.split(" "):
    word_start = full_text.find(word, current_index)
    word_end = word_start + len(word)
    words.append(word)
    word_index.append((word_start, word_end))
    current_index = word_end
    tag_seq = ['O']*len(words)
  anno_list = json.loads(annotation)
  for ann in anno_list:
    ann_start = ann['start']
    ann_end = ann['end']
    ann_type = ann['labels'][0]

    entity_start = False
    for i, (w_start, w_end) in enumerate(word_index):
      if ann_start <= w_start and w_end <= ann_end:
        if not entity_start:
          tag_seq[i] = f"B-{ann_type}"
          entity_start = True
        else:
          tag_seq[i] = f"I-{ann_type}"
  return  tag_seq

def get_train_val_test_splits(df):
    df['tag_seg'] = df.apply(lambda x: tagging(x['text'], x['label']), axis=1)
    trainning_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    train_df, val_df = train_test_split(trainning_df, test_size=0.2, random_state=42)
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)