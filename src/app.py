# app.py (stateless, one-shot Travel Planner Generator)
%%writefile app.py
# app.py (stateless, one-shot Travel Planner Generator)
import streamlit as st
import torch
import faiss
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer

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
    try:
        train = load_dataset("osunlp/TravelPlanner", "train")["train"]
        kb_q = [preprocess_text(x["query"]) for x in train]
        kb_a = [preprocess_text(x["answer"]) for x in train if "answer" in x]
        return kb_q + kb_a
    except Exception:
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

# ---------------- App ----------------
def main():
    st.title("🌍 Travel Planner Generator (One-shot)")
    st.write("Generate instant travel plans from text or voice. Stateless — no chat memory.")

    with st.spinner("Loading models and knowledge base..."):
        tok, mdl = load_model()
        kb = load_kb()
        index, embeddings, kb = build_faiss_index(kb)
        embedder = load_embedder()

    explainer = RAGExplainer(kb, embedder, mdl, tok)

    st.sidebar.header("Settings")
    k = st.sidebar.slider("Top-k retrieved docs", 1, 5, 3)
    language = st.sidebar.radio("Voice input language", ["English", "Arabic"], index=0)

    st.header("Input")
    st.write("Type your request or record a voice note — if both are used, you’ll be asked before replacing text.")

    # Session state for the text
    if "text_query" not in st.session_state:
        st.session_state.text_query = ""

    # Text input
    text_input = st.text_area("✍️ Type your travel request:", value=st.session_state.text_query, height=150)

    # Voice input
    audio_blob = st.audio_input("🎤 Record (optional)")

    if audio_blob is not None:
        with st.spinner("Transcribing audio..."):
            transcript = transcribe_audio(audio_blob, language=get_language_code(language))
        if transcript and not transcript.lower().startswith(("error", "could not")):
            st.success("✅ Transcription successful")
            st.write("**Transcribed text:**", transcript)

            # Decide if we replace existing text
            if not text_input.strip():
                st.session_state.text_query = transcript
                st.info("Text box auto-filled with your voice input ✅")
            else:
                replace = st.radio(
                    "Replace existing text with voice input?",
                    ["No", "Yes"],
                    horizontal=True,
                    index=0,
                )
                if replace == "Yes":
                    st.session_state.text_query = transcript
                    st.success("✅ Text box updated with voice input")
        else:
            st.error(f"❌ Transcription failed: {transcript}")

    # Update text query variable
    text_query = st.session_state.text_query or text_input

    # Generate button
    if st.button("🚀 Generate Travel Plan"):
        query = (text_query or "").strip()
        if not query:
            st.warning("⚠️ Please enter a request or record audio first.")
        else:
            with st.spinner("Generating your travel plan..."):
                answer, docs = explainer.explain(query, k=k)

            st.subheader("🧭 Generated Travel Plan")
            st.write(answer)

            if docs:
                st.subheader("📚 Supporting Documents")
                for i, d in enumerate(docs, start=1):
                    with st.expander(f"Document {i}"):
                        st.write(d)

    st.markdown("---")
    st.caption("This is a stateless one-shot generator — every input is processed independently.")

if __name__ == "__main__":
    main()



