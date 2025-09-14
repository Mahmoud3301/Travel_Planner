import faiss
import numpy as np
import torch  # إضافة استيراد torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
from data_preprocessing import load_and_preprocess_data

class RAGExplainer:
    def __init__(self, kb, embed_model, model, tokenizer, fewshot=None):
        self.kb = kb
        self.embedder = embed_model
        self.model = model
        self.tokenizer = tokenizer
        self.fewshot = fewshot or []
        
        # Build embeddings and index
        self.embeddings = self.embedder.encode(self.kb, convert_to_numpy=True)
        self.index = faiss.IndexFlatL2(self.embeddings.shape[1])
        self.index.add(self.embeddings)

    def retrieve(self, query, k=3):
        q_emb = self.embedder.encode([query], convert_to_numpy=True)
        D, I = self.index.search(q_emb, k)
        return [self.kb[i] for i in I[0]]

    def build_prompt(self, query, retrieved):
        prompt = "You are a travel planning assistant. Suggest itineraries with justification based on the context.\n\n"
        
        if self.fewshot:
            prompt += "Here are some examples:\n"
            for q, a in self.fewshot:
                prompt += f"User: {q}\nAssistant: {a}\n\n"
        
        prompt += "Context (similar knowledge):\n"
        for i, r in enumerate(retrieved, 1):
            prompt += f"{i}. {r}\n"
        
        prompt += f"\nNow answer this user query based on the context above:\nUser: {query}\nAssistant:"
        return prompt

    def explain(self, query, k=3, max_new_tokens=200):
        retrieved = self.retrieve(query, k)
        prompt = self.build_prompt(query, retrieved)
        
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs, 
                max_new_tokens=max_new_tokens, 
                do_sample=True, 
                top_k=50,
                temperature=0.7,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        answer = generated.replace(prompt, "").strip()
        return answer, retrieved

def build_rag_system():
    print("Building RAG system...")
    
    # Load data
    ds_train_processed, _, _ = load_and_preprocess_data()
    
    # Create knowledge base
    kb = [item["query"] for item in ds_train_processed] + [item["output"] for item in ds_train_processed if item["output"]]
    
    # Load embedder
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    
    # Load model
    try:
        tokenizer = AutoTokenizer.from_pretrained("./qwen_travelplanner_ft")
        model = AutoModelForCausalLM.from_pretrained(
            "./qwen_travelplanner_ft",
            device_map="auto",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )
    except:
        print("Using base model as fallback")
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
        model = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen2.5-0.5B-Instruct",
            device_map="auto",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )
    
    fewshot_examples = [
        ["Plan a 2-day trip to Rome focusing on history.",
         "Day 1: Colosseum, Roman Forum, Palatine Hill. Day 2: Vatican Museums, St. Peter's Basilica. Balanced ancient + religious history."]
    ]
    
    explainer = RAGExplainer(kb, embed_model, model, tokenizer, fewshot=fewshot_examples)
    print("RAG system built successfully!")
    return explainer

if __name__ == "__main__":
    rag_system = build_rag_system()
