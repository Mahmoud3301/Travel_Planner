# app.py (stateless, one-shot Travel Planner Generator)
import streamlit as st
import torch
import faiss
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer

# local modules
from rag_system import RAGExplainer
from voice_processing import transcribe_audio, get_language_code
from data_preprocessing import preprocess_text

# ---------------- Settings ----------------
MODEL_DIR = "./qwen_travelplanner_ft"
EMBED_MODEL = "all-MiniLM-L6-v2"

st.set_page_config(page_title="Travel Planner Generator", page_icon="🌍", layout="wide")

# ---------------- Loaders (cached) ----------------
@st.cache_resource(show_spinner=False)
def load_model():
    """Load tokenizer and model. Uses device_map auto; returns tokenizer and model."""
    try:
        tok = AutoTokenizer.from_pretrained(MODEL_DIR)
        mdl = AutoModelForCausalLM.from_pretrained(
            MODEL_DIR,
            device_map="auto",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        )
        return tok, mdl
    except Exception:
        tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        mdl = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen2.5-0.5B-Instruct",
            device_map="auto",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        )
        return tok, mdl

@st.cache_resource(show_spinner=False)
def load_embedder():
    return SentenceTransformer(EMBED_MODEL)

@st.cache_data(show_spinner=False)
def load_kb():
    """Load and preprocess knowledge base. Returns list of strings."""
    try:
        train = load_dataset("osunlp/TravelPlanner", "train")["train"]
        kb_q = [preprocess_text(x["query"]) for x in train]
        kb_a = [preprocess_text(x["answer"]) for x in train if "answer" in x]
        return kb_q + kb_a
    except Exception as e:
        # fallback sample KB
        return [
            "plan a trip to paris for 3 days",
            "best hotels in new york with budget under 200",
            "romantic destinations in italy for couples",
            "family vacation ideas in florida",
            "adventure travel options in costa rica",
        ]

@st.cache_data(show_spinner=False)
def build_faiss_index(kb):
    embedder = load_embedder()
    embeddings = embedder.encode(kb, convert_to_numpy=True)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index, embeddings, kb

# ---------------- App UI ----------------
def main():
    st.title("🌍 Travel Planner Generator (One-shot)")
    st.write("Enter your travel request (text or voice). The app will generate a single travel plan — no multi-turn chat or session memory.")

    with st.spinner("Loading models and knowledge base..."):
        tok, mdl = load_model()
        kb = load_kb()
        index, embeddings, kb = build_faiss_index(kb)
        embedder = load_embedder()

    explainer = RAGExplainer(kb, embedder, mdl, tok)

    # Sidebar settings
    st.sidebar.header("Settings")
    k = st.sidebar.slider("Top-k retrieved docs", 1, 5, 3)
    language = st.sidebar.radio("Voice input language", ["English", "Arabic"], index=0)

    # Input form
    st.header("Input")
    st.write("You can either type your request or record a short voice message. Recording is optional.")

    col1, col2 = st.columns([3,1])
    with col1:
        text_input = st.text_area("Type your travel request:", height=150)
    with col2:
        audio_blob = st.audio_input("Record (optional)")
        if audio_blob is not None:
            with st.spinner("Transcribing audio..."):
                transcript = transcribe_audio(audio_blob, language=get_language_code(language))
            if transcript and not transcript.lower().startswith(("error", "could not")):
                st.success("Transcription OK")
                st.write("Transcribed:")
                st.write(transcript)
                # if user didn't type text, use transcript
                if not text_input.strip():
                    text_input = transcript
            else:
                st.error("Transcription failed: " + str(transcript))

    # Generate button (one-shot)
    if st.button("Generate Travel Plan"):
        query = (text_input or "").strip()
        if not query:
            st.warning("Please type a request or record audio first.")
        else:
            with st.spinner("Retrieving knowledge and generating plan..."):
                answer, docs = explainer.explain(query, k=k)

            st.subheader("Generated Travel Plan")
            st.write(answer)

            if docs:
                st.subheader("Retrieved supporting documents (one-shot)")
                for i, d in enumerate(docs, start=1):
                    with st.expander(f"Doc {i}"):
                        st.write(d)

    st.markdown("---")
    st.caption("Note: This app is stateless — each request is handled independently (one-shot). Powered by a fine-tuned model + RAG.")

if __name__ == "__main__":
    main()


