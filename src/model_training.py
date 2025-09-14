import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from data_preprocessing import load_and_preprocess_data

class TravelDataset(torch.utils.data.Dataset):
    def __init__(self, data, tokenizer, max_length=256):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        text = f"### Instruction: {item['query']}\n### Response: {item['output']}"
        encoding = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': encoding['input_ids'].flatten()
        }

def train_model():
    print("Loading preprocessed data...")
    ds_train_processed, ds_val_processed, _ = load_and_preprocess_data()
    
    print("Loading model and tokenizer...")
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        quantization_config=bnb_config
    )

    print("Preparing model for LoRA training...")
    model = prepare_model_for_kbit_training(model)
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, peft_config)

    print("Creating datasets...")
    train_dataset = TravelDataset(ds_train_processed[:100], tokenizer)
    val_dataset = TravelDataset(ds_val_processed[:20], tokenizer)

    training_args = TrainingArguments(
        output_dir="./qwen_travelplanner_lora",
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=4,
        eval_strategy="steps",
        save_strategy="steps",
        logging_steps=10,
        eval_steps=50,
        save_steps=50,
        learning_rate=2e-4,
        num_train_epochs=1,
        fp16=torch.cuda.is_available(),
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer
    )

    print("Starting training...")
    trainer.train()
    trainer.save_model("./qwen_travelplanner_ft")
    print("Model training completed and saved!")

if __name__ == "__main__":
    train_model()
