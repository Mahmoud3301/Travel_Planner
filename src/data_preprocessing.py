import re
from datasets import load_dataset

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-z0-9\s.,!?]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def load_and_preprocess_data():
    print("Loading dataset...")
    ds_train = load_dataset("osunlp/TravelPlanner", "train")["train"]
    ds_val = load_dataset("osunlp/TravelPlanner", "validation")["validation"]
    ds_test = load_dataset("osunlp/TravelPlanner", "test")["test"]

    print(f"Train size = {len(ds_train)}")
    print(f"Validation size = {len(ds_val)}")
    print(f"Test size = {len(ds_test)}")

    print("Preprocessing data...")
    def preprocess_dataset(dataset):
        processed_data = []
        for item in dataset:
            processed_item = {
                "query": preprocess_text(item["query"]),
                "output": preprocess_text(item["answer"]) if "answer" in item else ""
            }
            processed_data.append(processed_item)
        return processed_data

    ds_train_processed = preprocess_dataset(ds_train)
    ds_val_processed = preprocess_dataset(ds_val)
    ds_test_processed = preprocess_dataset(ds_test)

    return ds_train_processed, ds_val_processed, ds_test_processed

if __name__ == "__main__":
    train, val, test = load_and_preprocess_data()
    print("Data preprocessing completed!")
