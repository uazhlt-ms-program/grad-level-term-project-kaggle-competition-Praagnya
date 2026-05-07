"""Load best distilbert checkpoint and predict on test set."""
import re, warnings, os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm

warnings.filterwarnings("ignore")

MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"
MAX_LEN    = 256
BATCH_SIZE = 128   # larger batch for fast inference

def is_chinese(text):
    return bool(re.search(r'[\u4e00-\u9fff]', str(text)))

class ReviewDataset(Dataset):
    def __init__(self, texts, tokenizer, max_len):
        self.texts, self.tokenizer, self.max_len = texts, tokenizer, max_len
    def __len__(self): return len(self.texts)
    def __getitem__(self, idx):
        enc = self.tokenizer(self.texts[idx], max_length=self.max_len,
                             padding="max_length", truncation=True, return_tensors="pt")
        return {k: v.squeeze(0) for k, v in enc.items()}

def predict(model, loader, device):
    model.eval()
    preds = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="predicting", ncols=80):
            batch = {k: v.to(device) for k, v in batch.items()}
            preds.extend(model(**batch).logits.argmax(-1).cpu().numpy())
    return np.array(preds)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

test = pd.read_csv("data/test.csv").fillna("")
test["chinese"] = test.TEXT.apply(is_chinese)
test_ml = test[~test.chinese].reset_index(drop=True)
print(f"Test rows: {len(test)}  |  ML subset: {len(test_ml)}  |  Chinese: {test.chinese.sum()}")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=3, ignore_mismatched_sizes=True)
model.load_state_dict(torch.load("models/distilbert_sst2_best.pt", map_location=device))
model.to(device)
print("Loaded models/distilbert_sst2_best.pt")

ds     = ReviewDataset(test_ml["TEXT"].tolist(), tokenizer, MAX_LEN)
loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
preds_ml = predict(model, loader, device)

submission = pd.DataFrame({"ID": test["ID"], "LABEL": -1})
submission.loc[test.chinese.values,  "LABEL"] = 0
submission.loc[~test.chinese.values, "LABEL"] = preds_ml
assert (submission.LABEL == -1).sum() == 0

os.makedirs("submissions", exist_ok=True)
out = "submissions/submission_distilbert_sst2.csv"
submission.to_csv(out, index=False)
print(f"Saved -> {out}")
