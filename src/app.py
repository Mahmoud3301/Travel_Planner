import streamlit as st
import torch, re, faiss
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer

# your modules
from rag_system import RAGExplainer
from voice_processing import transcribe_audio, get_language_code
from data_preprocessing import preprocess_text

# ---------------- Settings ----------------
MODEL_DIR   = "./qwen_travelplanner_ft"
EMBED_MODEL = "all-MiniLM-L6-v2"

st.set_page_config(page_title="Travel Planner + RAG + Voice", page_icon="🌍", layout="wide")

# ---------------- Loaders ----------------
@st.cache_resource
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

@st.cache_resource
def load_embedder():
    return SentenceTransformer(EMBED_MODEL)

@st.cache_data
def load_data():
    train = load_dataset("osunlp/TravelPlanner", "train")["train"]
    kb_q = [preprocess_text(x["query"]) for x in train]
    kb_a = [preprocess_text(x["answer"]) for x in train if "answer" in x]
    return kb_q + kb_a

@st.cache_data
def build_faiss_index(kb):
    embedder = load_embedder()
    X = embedder.encode(kb, convert_to_numpy=True)
    index = faiss.IndexFlatL2(X.shape[1])
    index.add(X)
    return index, X, kb

# ---------------- App ----------------
def main():
    st.title("🌍 Travel Planner Assistant + RAG + Voice")
    st.caption("Type or speak. Whisper → transcript → RAG → Qwen answer with retrieved context.")

    # ---- init session keys
    for k, v in {
        "text_query": "",
        "mic_open": False,
        # pending_text is used to safely inject text into the input on the next run
        "pending_text": None,
    }.items():
        st.session_state.setdefault(k, v)

    # ---- BEFORE rendering widgets: apply any pending transcript safely
    if st.session_state.get("pending_text"):
        st.session_state["text_query"] = st.session_state.pop("pending_text")

    with st.spinner("Loading model & data…"):
        tok, mdl = load_model()
        kb_data  = load_data()
        index, X, kb = build_faiss_index(kb_data)
        embedder = load_embedder()

    explainer = RAGExplainer(
        kb, embedder, mdl, tok,
        fewshot=[["Plan a 2-day trip to Rome focusing on history.",
                  "Day 1: Colosseum, Roman Forum, Palatine Hill. Day 2: Vatican Museums, St. Peter's Basilica."]]
    )

    # ---- Sidebar
    st.sidebar.header("⚙️ Settings")
    k          = st.sidebar.slider("Top-k retrieved", 1, 5, 3)
    languageUI = st.sidebar.radio("Input language", ["English", "Arabic"], index=0)

    # ---- Styles
    st.markdown("""
    <style>
    .input-wrap {background:#f6f7f8;border:1px solid #e5e7eb;border-radius:9999px;
                 padding:10px 12px;display:flex;align-items:center;gap:8px;}
    .sendbtn{width:36px;height:36px;border-radius:9999px;border:none;background:#111;color:#fff;font-weight:600;}
    .reveal{animation:slideDown .18s ease-out;}
    @keyframes slideDown{from{opacity:0;transform:translateY(-6px)} to{opacity:1;transform:translateY(0)}}
    </style>
    """, unsafe_allow_html=True)

    # ---- Input row
    st.markdown('<div class="input-wrap">', unsafe_allow_html=True)
    c1, c2, c3 = st.columns([12,1,1], gap="small")

    with c1:
        st.text_input(
            "Enter your travel query",
            key="text_query",                        # single widget key owned by Streamlit
            placeholder="e.g., 3 days in Paris with museums and cafés",
            label_visibility="collapsed",
        )
    with c2:
        if st.button("🎙️", key="mic_toggle", help="Record / stop", use_container_width=True):
            st.session_state["mic_open"] = not st.session_state["mic_open"]
    with c3:
        send_clicked = st.button("↑", key="send_btn", help="Send", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # ---- Recording bar (built-in recorder)
    if st.session_state["mic_open"]:
        st.markdown("<div class='reveal'></div>", unsafe_allow_html=True)
        st.caption("Click to record, then stop. Transcript will fill the text box.")
        audio_blob = st.audio_input("Record your question", key="audio_input_inline")
        if audio_blob is not None:
            with st.spinner("Transcribing..."):
                transcript = transcribe_audio(audio_blob, language=get_language_code(languageUI))
            if transcript and not transcript.lower().startswith(("error", "could not")):
                # IMPORTANT: store into pending_text, NOT directly into text_query
                st.session_state["pending_text"] = transcript
                st.session_state["mic_open"] = False
                st.rerun()   # next run will copy pending_text -> text_query BEFORE rendering the widget
            else:
                st.error(f"Transcription failed: {transcript}")

    # ---- Execute on send/plan
    run_clicked = send_clicked or st.button("🔍 Plan Trip")
    if run_clicked:
        q = (st.session_state.get("text_query") or "").strip()
        if not q:
            st.warning("Please type a query or record audio first.")
        else:
            with st.spinner("Generating response..."):
                ans, docs = explainer.explain(q, k=k)

            col1, col2 = st.columns([2,3])
            with col1:
                st.markdown(
                    """
                    <div style="background-color:#e6f7ff;
                                padding:20px; border-radius:12px;
                                box-shadow:0 0 8px rgba(0,0,0,0.1);
                                color:#000000;">
                    <h3 style="color:#0050b3;">Assistant Suggestion</h3>
                    </div>
                    """, unsafe_allow_html=True
                )
                st.markdown(
                    f"<div style='color:#000; font-size:16px; line-height:1.6; font-weight:500;'>{ans}</div>",
                    unsafe_allow_html=True
                )
            with col2:
                st.write("### 📚 Retrieved Context")
                for i, d in enumerate(docs, 1):
                    with st.expander(f"Document {i}"):
                        st.write(d)

    st.markdown("---")
    st.markdown("<center><sub>⚡ Qwen (fine-tuned) + FAISS RAG | Whisper ASR | Streamlit</sub></center>",
                unsafe_allow_html=True)

if __name__ == "__main__":
    main()
