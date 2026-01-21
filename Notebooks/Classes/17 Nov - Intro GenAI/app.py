import os, json
from typing import List
import numpy as np
import gradio as gr
import faiss
from sentence_transformers import SentenceTransformer

HF_TOKEN = os.getenv("HF_TOKEN")

# ---------- Paths ----------
APP_DIR    = os.path.dirname(__file__)
ASSETS_DIR = os.path.join(APP_DIR, "assets")

CORPUS_JSON = os.path.join(ASSETS_DIR, "corpus.json")
FAISS_MAIN  = os.path.join(ASSETS_DIR, "faiss_ip_768.index")

# ---------- Load corpus ----------
with open(CORPUS_JSON, "r", encoding="utf-8") as f:
    corpus = json.load(f)  # [{"title", "text"}, ...]

# ---------- Load FAISS index ----------
if not os.path.exists(FAISS_MAIN):
    raise FileNotFoundError(f"Missing FAISS index at {FAISS_MAIN}")
index = faiss.read_index(FAISS_MAIN)

# Infer dimension from index
EMB_DIM = index.d

# ---------- Model ----------
model = SentenceTransformer("google/embeddinggemma-300m", token=HF_TOKEN)

# ---------- Search ----------
def do_search(query: str, top_k: int = 5) -> List[List[str]]:
    if not query.strip():
        return []
    q_emb = model.encode_query(query, normalize_embeddings=True, convert_to_numpy=True).astype("float32")
    scores, idxs = index.search(q_emb[None, :], top_k)
    rows = []
    for score, i in zip(scores[0], idxs[0]):
        if i == -1: continue
        item = corpus[i]
        snippet = item["text"][:380] + ("…" if len(item["text"]) > 380 else "")
        rows.append([f"{score:.4f}", item["title"], snippet])
    return rows

# ---------- Similarity ----------
def do_similarity(text_a: str, text_b: str) -> float:
    a = model.encode_document([text_a], normalize_embeddings=True, convert_to_numpy=True)[0]
    b = model.encode_document([text_b], normalize_embeddings=True, convert_to_numpy=True)[0]
    return float(np.dot(a, b))

# ---------- UI ----------
with gr.Blocks(title="EmbeddingGemma FAISS Search") as demo:
    gr.Markdown("# Simple FAISS Search (No Matryoshka)")

    with gr.Tabs():
        with gr.TabItem("Search"):
            q = gr.Textbox(label="Query")
            topk = gr.Slider(1, 20, value=5, step=1, label="Top-K")
            run = gr.Button("Search")
            out = gr.Dataframe(headers=["score", "title", "snippet"], wrap=True)
            run.click(lambda query, k: do_search(query, int(k)), [q, topk], out)

        with gr.TabItem("Similarity"):
            a = gr.Textbox(lines=4, label="Text A")
            b = gr.Textbox(lines=4, label="Text B")
            sim_btn = gr.Button("Compute")
            sim_out = gr.Number(label="Cosine similarity")
            sim_btn.click(lambda x, y: do_similarity(x, y), [a, b], sim_out)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
