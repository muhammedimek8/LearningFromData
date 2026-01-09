import os
import pandas as pd

BASE_DIR = "aclImdb"
OUTPUT_FILE = "data/dataset.csv"

def load_reviews(folder, label):
    texts = []
    labels = []
    for file in os.listdir(folder):
        if file.endswith(".txt"):
            with open(os.path.join(folder, file), encoding="utf-8") as f:
                texts.append(f.read())
                labels.append(label)
    return texts, labels

def main():
    pos_texts, pos_labels = load_reviews(f"{BASE_DIR}/train/pos", "positive")
    neg_texts, neg_labels = load_reviews(f"{BASE_DIR}/train/neg", "negative")

    texts = pos_texts + neg_texts
    labels = pos_labels + neg_labels

    df = pd.DataFrame({"text": texts, "label": labels})
    os.makedirs("data", exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)

    print(f"Dataset saved to {OUTPUT_FILE}")
    print(df["label"].value_counts())

if __name__ == "__main__":
    main()
