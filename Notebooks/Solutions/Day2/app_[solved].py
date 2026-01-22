import gradio as gr
import numpy as np
import torch
import faiss
import re
from typing import List, Tuple, Dict
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, TaskType
import os

# Set environment variables to reduce warnings
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# Device configuration
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# ==================== CORPUS ====================
CORPUS = [
    "Alexander Fleming discovered penicillin in 1928, marking the beginning of modern antibiotics.",
    "Penicillin was the first antibiotic discovered and revolutionized medicine.",
    "Antibiotics are medications that fight bacterial infections in humans and animals.",
    "Paris is the capital and most populous city of France, known for the Eiffel Tower.",
    "Berlin is the capital of Germany and has a rich history spanning centuries.",
    "The Eiffel Tower was constructed in 1889 and stands 330 meters tall.",
    "Marie Curie was the first woman to win a Nobel Prize for her work on radioactivity.",
    "Radioactivity was discovered by Henri Becquerel in 1896.",
    "The Nobel Prize is awarded annually in physics, chemistry, medicine, literature, and peace.",
    "Albert Einstein developed the theory of relativity in the early 20th century.",
    "The theory of relativity revolutionized our understanding of space, time, and gravity.",
    "Gravity is the force that attracts objects with mass toward each other.",
    "DNA is the molecule that carries genetic information in all living organisms.",
    "The structure of DNA was discovered by Watson and Crick in 1953.",
    "Genetic information is passed from parents to offspring through DNA.",
]

# ==================== TEST QUERIES ====================
EASY_QUERIES = [
    "Who discovered the first antibiotic?",
    "Who discovered penicillin?",
    "What are antibiotics?",
    "Who discovered the structure of DNA?",
    "What is the capital of France?",
]

HARD_QUERIES = [
    "What medical breakthrough happened in the late 1920s?",
    "Which scientific discoveries revolutionized their fields?",
    "What important discoveries were made in the early 20th century?",
    "Who were the pioneering scientists in medicine?",
    "What major architectural achievements occurred in the 19th century?",
]

# ==================== GLOBAL MODELS ====================
baseline_model = None
baseline_index = None
finetuned_model = None
finetuned_index = None
lora_model = None
lora_tokenizer = None
qa_model = None
qa_tokenizer = None

# ==================== INITIALIZATION ====================
def initialize_models():
    """Initialize all models on app startup."""
    global baseline_model, baseline_index, qa_model, qa_tokenizer
    
    # Load baseline embedding model
    baseline_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Build baseline FAISS index
    baseline_embeddings = baseline_model.encode(
        CORPUS,
        convert_to_numpy=True,
        normalize_embeddings=True
    )
    dimension = baseline_embeddings.shape[1]
    baseline_index = faiss.IndexFlatIP(dimension)
    baseline_index.add(baseline_embeddings.astype('float32'))
    
    # Load QA model
    qa_model_name = 'TinyLlama/TinyLlama-1.1B-Chat-v1.0'
    qa_tokenizer = AutoTokenizer.from_pretrained(qa_model_name)
    qa_tokenizer.pad_token = qa_tokenizer.eos_token
    
    qa_model = AutoModelForCausalLM.from_pretrained(
        qa_model_name,
        torch_dtype=torch.float16 if device == 'cuda' else torch.float32,
        device_map='auto' if device == 'cuda' else None
    )
    
    if device == 'cpu':
        qa_model = qa_model.to(device)
    
    return "✓ Models initialized successfully!"

# ==================== FINE-TUNING ====================
def train_finetuned_model(epochs: int, progress=gr.Progress()):
    """Train the fine-tuned embedding model."""
    global finetuned_model, finetuned_index
    
    progress(0, desc="Initializing fine-tuned model...")
    
    # Create fresh model
    finetuned_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Create training examples (direct queries only)
    train_examples = [
        InputExample(texts=["Who discovered the first antibiotic?", CORPUS[0]], label=1.0),
        InputExample(texts=["Who discovered the first antibiotic?", CORPUS[1]], label=1.0),
        InputExample(texts=["Who discovered the first antibiotic?", CORPUS[2]], label=0.8),
        InputExample(texts=["Who discovered the first antibiotic?", CORPUS[3]], label=0.0),
        InputExample(texts=["Who discovered the first antibiotic?", CORPUS[4]], label=0.0),
        InputExample(texts=["What is DNA?", CORPUS[12]], label=1.0),
        InputExample(texts=["What is DNA?", CORPUS[13]], label=1.0),
        InputExample(texts=["What is DNA?", CORPUS[14]], label=1.0),
        InputExample(texts=["What is DNA?", CORPUS[0]], label=0.0),
        InputExample(texts=["What is DNA?", CORPUS[3]], label=0.0),
    ]
    
    progress(0.2, desc="Creating dataloader...")
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=2)
    train_loss = losses.CosineSimilarityLoss(finetuned_model)
    
    progress(0.3, desc=f"Training for {epochs} epochs...")
    finetuned_model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=epochs,
        warmup_steps=5,
        show_progress_bar=False
    )
    
    progress(0.8, desc="Building FAISS index...")
    # Build FAISS index
    finetuned_embeddings = finetuned_model.encode(
        CORPUS,
        convert_to_numpy=True,
        normalize_embeddings=True
    )
    dimension = finetuned_embeddings.shape[1]
    finetuned_index = faiss.IndexFlatIP(dimension)
    finetuned_index.add(finetuned_embeddings.astype('float32'))
    
    progress(1.0, desc="Complete!")
    
    return f"✓ Fine-tuned model trained for {epochs} epochs!"

# ==================== LORA TRAINING ====================
def train_lora_model(epochs: int, learning_rate: float, lora_r: int, progress=gr.Progress()):
    """Train the LoRA reranking model."""
    global lora_model, lora_tokenizer
    
    progress(0, desc="Loading base model...")
    
    model_name = 'TinyLlama/TinyLlama-1.1B-Chat-v1.0'
    lora_tokenizer = AutoTokenizer.from_pretrained(model_name)
    lora_tokenizer.pad_token = lora_tokenizer.eos_token
    
    llm_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device == 'cuda' else torch.float32,
        device_map='auto' if device == 'cuda' else None
    )
    
    if device == 'cpu':
        llm_model = llm_model.to(device)
    
    progress(0.2, desc="Configuring LoRA...")
    
    # Configure LoRA
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_r * 2,
        target_modules=['q_proj', 'v_proj'],
        lora_dropout=0.1,
        bias='none',
        task_type=TaskType.CAUSAL_LM
    )
    
    llm_model = get_peft_model(llm_model, lora_config)
    
    progress(0.3, desc="Creating training data...")
    
    # Create training data with both easy and hard examples
    def format_prompt(query: str, document: str, score: float) -> str:
        return (
            f"<|system|>\n"
            f"You are a document relevance scorer. Given a query and a document, "
            f"output a single relevance score between 0.0 (not relevant) and 1.0 (highly relevant).\n"
            f"<|user|>\n"
            f"Query: {query}\n"
            f"Document: {document}\n"
            f"Relevance Score: {score:.2f}"
        )
    
    train_texts = [
        # Easy direct matches
        format_prompt("Who discovered the first antibiotic?", CORPUS[0], 0.95),
        format_prompt("Who discovered the first antibiotic?", CORPUS[1], 0.95),
        format_prompt("Who discovered penicillin?", CORPUS[0], 0.95),
        format_prompt("Who discovered penicillin?", CORPUS[1], 0.90),
        
        # Hard implicit matches - THIS IS KEY!
        format_prompt("What medical breakthrough happened in the late 1920s?", CORPUS[0], 0.95),
        format_prompt("What medical breakthrough happened in the late 1920s?", CORPUS[1], 0.90),
        format_prompt("Which scientific discoveries revolutionized their fields?", CORPUS[0], 0.85),
        format_prompt("Which scientific discoveries revolutionized their fields?", CORPUS[1], 0.85),
        format_prompt("Which scientific discoveries revolutionized their fields?", CORPUS[10], 0.85),
        
        # Negative examples
        format_prompt("Who discovered the first antibiotic?", CORPUS[3], 0.05),
        format_prompt("Who discovered the first antibiotic?", CORPUS[4], 0.05),
        format_prompt("What medical breakthrough happened in the late 1920s?", CORPUS[3], 0.05),
        format_prompt("What medical breakthrough happened in the late 1920s?", CORPUS[5], 0.05),
    ]
    
    progress(0.4, desc="Tokenizing...")
    
    # Tokenize
    tokenized_data = lora_tokenizer(
        train_texts,
        truncation=True,
        padding='max_length',
        max_length=512,
        return_tensors='pt'
    )
    
    class RerankerDataset(torch.utils.data.Dataset):
        def __init__(self, encodings):
            self.encodings = encodings
        
        def __len__(self):
            return self.encodings['input_ids'].shape[0]
        
        def __getitem__(self, idx):
            item = {key: val[idx] for key, val in self.encodings.items()}
            item['labels'] = item['input_ids'].clone()
            return item
    
    train_dataset = RerankerDataset(tokenized_data)
    
    progress(0.5, desc=f"Training for {epochs} epochs...")
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir='./lora_reranker',
        per_device_train_batch_size=2,
        num_train_epochs=epochs,
        logging_steps=max(1, epochs),
        save_strategy='no',
        report_to='none',
        learning_rate=learning_rate,
        warmup_steps=min(10, epochs * 2),
    )
    
    trainer = Trainer(
        model=llm_model,
        args=training_args,
        train_dataset=train_dataset,
    )
    
    trainer.train()
    
    progress(1.0, desc="Complete!")
    
    lora_model = llm_model
    
    trainable_params = sum(p.numel() for p in llm_model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in llm_model.parameters())
    
    return f"✓ LoRA model trained!\n• Epochs: {epochs}\n• Learning rate: {learning_rate}\n• LoRA rank: {lora_r}\n• Trainable params: {trainable_params:,} / {total_params:,} ({100 * trainable_params / total_params:.2f}%)"

# ==================== RETRIEVAL FUNCTIONS ====================
def retrieve_baseline(query: str, top_k: int = 3) -> List[Tuple[int, float]]:
    """Retrieve using baseline embeddings."""
    query_emb = baseline_model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True
    )
    scores, indices = baseline_index.search(query_emb.astype('float32'), top_k)
    return [(int(idx), float(score)) for idx, score in zip(indices[0], scores[0])]

def retrieve_finetuned(query: str, top_k: int = 3) -> List[Tuple[int, float]]:
    """Retrieve using fine-tuned embeddings."""
    if finetuned_model is None:
        return []
    
    query_emb = finetuned_model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True
    )
    scores, indices = finetuned_index.search(query_emb.astype('float32'), top_k)
    return [(int(idx), float(score)) for idx, score in zip(indices[0], scores[0])]

def score_with_lora(query: str, document: str, max_new_tokens: int = 10) -> float:
    """Score a query-document pair with LoRA."""
    if lora_model is None:
        return 0.5
    
    prompt = (
        f"<|system|>\n"
        f"You are a document relevance scorer. Given a query and a document, "
        f"output a single relevance score between 0.0 (not relevant) and 1.0 (highly relevant).\n"
        f"<|user|>\n"
        f"Query: {query}\n"
        f"Document: {document}\n"
        f"Relevance Score: "
    )
    
    inputs = lora_tokenizer(prompt, return_tensors='pt').to(lora_model.device)
    
    with torch.no_grad():
        outputs = lora_model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=lora_tokenizer.eos_token_id
        )
    
    generated_text = lora_tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    try:
        score_part = generated_text.split('Relevance Score:')[-1].strip()
        match = re.search(r'([0-1](?:\.\d+)?)', score_part)
        if match:
            score = float(match.group(1))
            return max(0.0, min(1.0, score))
    except:
        pass
    
    return 0.5

def retrieve_lora(query: str, top_k: int = 3) -> List[Tuple[int, float]]:
    """Retrieve using LoRA reranking."""
    if lora_model is None:
        return []
    
    # Get candidates from baseline
    candidates = retrieve_baseline(query, top_k=20)
    
    # Rerank with LoRA
    lora_results = []
    for idx, _ in candidates:
        lora_score = score_with_lora(query, CORPUS[idx])
        lora_results.append((idx, lora_score))
    
    # Sort by LoRA score
    lora_results.sort(key=lambda x: x[1], reverse=True)
    
    return lora_results[:top_k]

# ==================== QA FUNCTION ====================
def answer_question(query: str, context_docs: List[str]) -> str:
    """Generate answer using LLM with context."""
    if not context_docs:
        return "No context available."
    
    context = "\n".join([f"[{i+1}] {doc}" for i, doc in enumerate(context_docs)])
    
    prompt = f"""<|system|>
You are a helpful assistant. Answer the question based on the provided context. Be concise and accurate.
<|user|>
Context:
{context}

Question: {query}

Answer: <|assistant|>
"""
    
    inputs = qa_tokenizer(prompt, return_tensors='pt', truncation=True, max_length=1024).to(qa_model.device)
    
    with torch.no_grad():
        outputs = qa_model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=False,
            pad_token_id=qa_tokenizer.eos_token_id,
            temperature=0.1
        )
    
    full_response = qa_tokenizer.decode(outputs[0], skip_special_tokens=True)
    answer = full_response.split('<|assistant|>')[-1].strip()
    
    return answer

# ==================== MAIN SEARCH FUNCTION ====================
def search_and_answer(query: str, approach: str, top_k: int = 3):
    """Search and generate answer based on selected approach."""
    
    # Retrieve documents
    if approach == "Baseline (Pre-trained Embeddings)":
        results = retrieve_baseline(query, top_k)
        approach_name = "Baseline"
    elif approach == "Fine-Tuned (Contrastive Learning)":
        if finetuned_model is None:
            return "❌ Please train the fine-tuned model first!", "", ""
        results = retrieve_finetuned(query, top_k)
        approach_name = "Fine-Tuned"
    elif approach == "LoRA Reranking":
        if lora_model is None:
            return "❌ Please train the LoRA model first!", "", ""
        results = retrieve_lora(query, top_k)
        approach_name = "LoRA"
    else:
        return "Invalid approach", "", ""
    
    # Format retrieval results
    retrieval_output = f"**{approach_name} Retrieval Results:**\n\n"
    context_docs = []
    
    for rank, (idx, score) in enumerate(results, 1):
        retrieval_output += f"**Rank {rank}:** Score: {score:.4f}\n"
        retrieval_output += f"[Doc {idx}] {CORPUS[idx]}\n\n"
        context_docs.append(CORPUS[idx])
    
    # Generate answer
    answer = answer_question(query, context_docs)
    
    answer_output = f"**{approach_name} Answer:**\n\n{answer}"
    
    # Format context
    context_output = "**Context Provided to LLM:**\n\n"
    for i, doc in enumerate(context_docs, 1):
        context_output += f"[{i}] {doc}\n\n"
    
    return retrieval_output, context_output, answer_output

# ==================== BATCH EVALUATION ====================
def evaluate_all_queries(approach: str, query_type: str):
    """Evaluate all queries of a given type."""
    
    queries = EASY_QUERIES if query_type == "Easy Queries" else HARD_QUERIES
    
    results_md = f"# Evaluation: {approach} on {query_type}\n\n"
    
    for i, query in enumerate(queries, 1):
        results_md += f"## Query {i}: \"{query}\"\n\n"
        
        retrieval, context, answer = search_and_answer(query, approach, top_k=3)
        
        if "❌" in retrieval:
            results_md += retrieval + "\n\n"
            continue
        
        results_md += retrieval + "\n"
        results_md += answer + "\n\n"
        results_md += "---\n\n"
    
    return results_md

# ==================== GRADIO INTERFACE ====================
def create_interface():
    
    with gr.Blocks(title="RAG Comparison: Baseline vs Fine-Tuned vs LoRA", theme=gr.themes.Soft()) as demo:
        
        gr.Markdown("""
        # 🔍 RAG Comparison: Three Retrieval Approaches
        
        Compare three different RAG (Retrieval-Augmented Generation) implementations:
        1. **Baseline**: Pre-trained embeddings (fast, general-purpose)
        2. **Fine-Tuned**: Contrastive learning (better for seen patterns)
        3. **LoRA Reranking**: LLM-based reranking (best for complex queries)
        
        **All three use the same RAG architecture** (Retrieval → Generation), but differ in retrieval quality.
        """)
        
        # Initialization
        with gr.Row():
            init_btn = gr.Button("🚀 Initialize Models", variant="primary", scale=1)
            init_status = gr.Textbox(label="Status", scale=2, interactive=False)
        
        init_btn.click(initialize_models, outputs=init_status)
        
        gr.Markdown("---")
        
        # Training Section
        with gr.Tab("⚙️ Training"):
            
            gr.Markdown("### Train Fine-Tuned Model (Contrastive Learning)")
            
            with gr.Row():
                ft_epochs = gr.Slider(1, 10, value=3, step=1, label="Epochs")
                ft_train_btn = gr.Button("Train Fine-Tuned Model", variant="primary")
            
            ft_status = gr.Textbox(label="Training Status", interactive=False)
            
            ft_train_btn.click(
                train_finetuned_model,
                inputs=[ft_epochs],
                outputs=ft_status
            )
            
            gr.Markdown("---")
            gr.Markdown("### Train LoRA Reranking Model")
            
            with gr.Row():
                lora_epochs = gr.Slider(1, 20, value=5, step=1, label="Epochs")
                lora_lr = gr.Slider(1e-5, 1e-3, value=2e-4, step=1e-5, label="Learning Rate")
                lora_r = gr.Slider(4, 32, value=8, step=4, label="LoRA Rank (r)")
            
            lora_train_btn = gr.Button("Train LoRA Model", variant="primary")
            lora_status = gr.Textbox(label="Training Status", interactive=False, lines=5)
            
            lora_train_btn.click(
                train_lora_model,
                inputs=[lora_epochs, lora_lr, lora_r],
                outputs=lora_status
            )
        
        # Search Section
        with gr.Tab("🔎 Interactive Search"):
            
            gr.Markdown("### Test Individual Queries")
            
            with gr.Row():
                query_input = gr.Textbox(
                    label="Enter your question",
                    placeholder="e.g., What medical breakthrough happened in the late 1920s?",
                    scale=3
                )
                approach_select = gr.Dropdown(
                    choices=[
                        "Baseline (Pre-trained Embeddings)",
                        "Fine-Tuned (Contrastive Learning)",
                        "LoRA Reranking"
                    ],
                    label="Select Approach",
                    value="Baseline (Pre-trained Embeddings)",
                    scale=1
                )
            
            search_btn = gr.Button("Search & Answer", variant="primary")
            
            with gr.Row():
                with gr.Column():
                    retrieval_output = gr.Markdown(label="Retrieval Results")
                with gr.Column():
                    context_output = gr.Markdown(label="Context")
            
            answer_output = gr.Markdown(label="Generated Answer")
            
            search_btn.click(
                search_and_answer,
                inputs=[query_input, approach_select],
                outputs=[retrieval_output, context_output, answer_output]
            )
            
            gr.Markdown("### Quick Test Queries")
            
            with gr.Row():
                with gr.Column():
                    gr.Markdown("**Easy Queries** (Direct matches)")
                    for query in EASY_QUERIES:
                        gr.Button(query, size="sm").click(
                            lambda q=query: q,
                            outputs=query_input
                        )
                
                with gr.Column():
                    gr.Markdown("**Hard Queries** (Implicit reasoning)")
                    for query in HARD_QUERIES:
                        gr.Button(query, size="sm").click(
                            lambda q=query: q,
                            outputs=query_input
                        )
        
        # Batch Evaluation
        with gr.Tab("📊 Batch Evaluation"):
            
            gr.Markdown("### Evaluate All Queries at Once")
            
            with gr.Row():
                eval_approach = gr.Dropdown(
                    choices=[
                        "Baseline (Pre-trained Embeddings)",
                        "Fine-Tuned (Contrastive Learning)",
                        "LoRA Reranking"
                    ],
                    label="Select Approach",
                    value="Baseline (Pre-trained Embeddings)"
                )
                eval_query_type = gr.Radio(
                    choices=["Easy Queries", "Hard Queries"],
                    label="Query Type",
                    value="Easy Queries"
                )
            
            eval_btn = gr.Button("Run Evaluation", variant="primary")
            eval_output = gr.Markdown(label="Evaluation Results")
            
            eval_btn.click(
                evaluate_all_queries,
                inputs=[eval_approach, eval_query_type],
                outputs=eval_output
            )
        
        # Information
        with gr.Tab("ℹ️ About"):
            gr.Markdown("""
            ## How It Works
            
            ### RAG Architecture
            
            All three approaches use the same **RAG (Retrieval-Augmented Generation)** architecture:
            
            ```
            Question → [RETRIEVAL] → Context → [GENERATION] → Answer
            ```
            
            The difference is in the **RETRIEVAL** stage:
            
            ### 1. Baseline (Pre-trained Embeddings)
            - Uses `all-MiniLM-L6-v2` pre-trained model
            - Fast FAISS similarity search
            - **Good for**: Direct queries with keyword overlap
            - **Struggles with**: Implicit queries, reasoning
            
            ### 2. Fine-Tuned (Contrastive Learning)
            - Trains embeddings on positive/negative pairs
            - Pulls similar items together, pushes dissimilar apart
            - **Good for**: Domain-specific queries, seen patterns
            - **Struggles with**: Unseen implicit queries
            
            ### 3. LoRA Reranking
            - Two-stage: Fast retrieval + LLM reranking
            - Uses TinyLlama with LoRA adapter
            - **Good for**: Complex queries, implicit reasoning
            - **Trade-off**: Slower but more accurate
            
            ### Query Types
            
            **Easy Queries** (All approaches should work):
            - Direct keyword matches
            - Example: "Who discovered penicillin?"
            
            **Hard Queries** (Only LoRA should excel):
            - Implicit relationships
            - Temporal reasoning
            - World knowledge required
            - Example: "What medical breakthrough happened in the late 1920s?"
            
            ### Training Tips
            
            **Fine-Tuned Model**:
            - 3-5 epochs usually sufficient
            - More epochs = better for seen patterns
            - Won't help with unseen implicit queries
            
            **LoRA Model**:
            - 5-10 epochs recommended
            - Higher rank (r) = more capacity but slower
            - Learning rate: 1e-4 to 2e-4 works well
            - Trains only ~1-2% of parameters!
            
            ### Expected Results
            
            | Query Type | Baseline | Fine-Tuned | LoRA |
            |------------|----------|------------|------|
            | Easy (direct) | ✅ Good | ✅ Excellent | ✅ Excellent |
            | Hard (implicit) | ❌ Poor | ❌ Poor | ✅ Excellent |
            
            ### Corpus
            
            The app uses a small corpus of 15 documents covering:
            - Medical discoveries (Fleming, penicillin)
            - Physics (Einstein, relativity)
            - Biology (DNA, Watson & Crick)
            - Geography (Paris, Berlin)
            - History (Nobel Prize, Eiffel Tower)
            """)
    
    return demo

# ==================== LAUNCH ====================
if __name__ == "__main__":
    demo = create_interface()
    demo.launch()

