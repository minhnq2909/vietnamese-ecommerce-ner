from predictor import predict, bilstm_crf_model, idx2tag

result = predict("Giá vợt là 3m5", bilstm_crf_model, "BiLSTM-CRF")

print("BiLSTM-CRF predictions:")
print(f"Tokens: {result['tokens']}")
print(f"Labels: {result['labels']}")
print(f"Entities: {len(result['entities'])}")
for ent in result['entities']:
    print(f"  - {ent['label']}: {ent['text']} (tokens: {ent['tokens']})")
