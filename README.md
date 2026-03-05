# 🔍 Production RAG Pipeline

A modular, production-grade **Retrieval-Augmented Generation** system. Supports multiple LLM backends, semantic + MMR retrieval, RAGAS-style evaluation, streaming, and a full chat UI.

---

## 🏗️ Architecture

```
Documents (PDF / TXT)
        │
        ▼
┌───────────────┐
│  DocumentLoader│  langchain-community loaders (TextLoader, PyMuPDFLoader)
└───────┬───────┘
        │
        ▼
┌───────────────┐
│DocumentChunker│  RecursiveCharacterTextSplitter (size=1000, overlap=200)
└───────┬───────┘
        │
        ▼
┌──────────────────┐
│ EmbeddingManager │  SentenceTransformer · all-MiniLM-L6-v2 · 384-dim · L2-norm
└───────┬──────────┘
        │
        ▼
┌─────────────┐
│ VectorStore │  ChromaDB PersistentClient · cosine distance · metadata filtering
└──────┬──────┘
       │
       │◄── query_embedding
       ▼
┌──────────────┐
│ RAGRetriever │  Top-K retrieval  OR  MMR (diversity-aware) retrieval
└──────┬───────┘
       │  retrieved_docs
       ▼
┌─────────────┐
│ RAGGenerator│  Prompt builder + LLM backend (Groq / OpenAI / Ollama)
└──────┬──────┘
       │
       ▼
  RAGResponse (answer + sources + token usage)
```

---

## 📂 Project Structure

```
rag-pipeline/
│
├── rag/
│   ├── __init__.py       ← Public API exports
│   ├── loader.py         ← Document loading (TXT, PDF, directories)
│   ├── chunker.py        ← RecursiveCharacterTextSplitter wrapper
│   ├── embedder.py       ← SentenceTransformer embedding manager
│   ├── vectorstore.py    ← ChromaDB persistent store
│   ├── retriever.py      ← Top-K + MMR retrieval
│   ├── generator.py      ← LLM backends + prompt builder + RAGResponse
│   └── pipeline.py       ← End-to-end orchestrator
│
├── app.py                ← Streamlit chat UI
├── evaluate.py           ← RAGAS-style evaluation (5 metrics)
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

---

## ⚡ Quickstart

### 1. Install dependencies

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Set your API key

```bash
cp .env.example .env
# Edit .env and add your GROQ_API_KEY (free at console.groq.com)
```

### 3. Run the chat UI

```bash
streamlit run app.py
```

### 4. Or use the Python API directly

```python
from dotenv import load_dotenv
load_dotenv()

from rag import RAGPipeline

pipeline = RAGPipeline(provider="groq")
pipeline.index(["./data/pdf", "./data/text_files"])

response = pipeline.query("What is the attention mechanism?")
print(response.answer)
print(response.format_sources())
print(f"Tokens used: {response.total_tokens}")
```

---

## 🔧 Component Details

### DocumentLoader

```python
from rag.loader import DocumentLoader

loader = DocumentLoader()
docs = loader.load_file("report.pdf")           # single file
docs = loader.load_directory("./data")          # whole directory
docs = loader.load_from_paths(["a.pdf", "./b"]) # mixed list
```

### DocumentChunker

```python
from rag.chunker import DocumentChunker

chunker = DocumentChunker(chunk_size=1000, chunk_overlap=200)
chunks = chunker.split(docs)
```

**Choosing chunk size:**
| Use case | chunk_size | chunk_overlap |
|---|---|---|
| Precise fact retrieval | 300–500 | 50 |
| General Q&A | 800–1200 | 150–200 |
| Long-form summarization | 1500–2000 | 300 |

### EmbeddingManager

```python
from rag.embedder import EmbeddingManager

embedder = EmbeddingManager("all-MiniLM-L6-v2")
embeddings = embedder.embed(["text one", "text two"])  # shape (2, 384)
query_emb  = embedder.embed_query("my question")       # shape (384,)
```

**Model options:**
| Model | Dim | Speed | Quality |
|---|---|---|---|
| `all-MiniLM-L6-v2` | 384 | ⚡ Fast | ✅ Good |
| `all-mpnet-base-v2` | 768 | 🐢 Slower | ✅✅ Better |
| `BAAI/bge-small-en-v1.5` | 384 | ⚡ Fast | ✅✅ Retrieval-optimized |

### RAGRetriever

```python
from rag.retriever import RAGRetriever

retriever = RAGRetriever(vector_store, embedding_manager)

# Standard top-K
results = retriever.retrieve("What is Python?", top_k=5, score_threshold=0.3)

# MMR (diverse results, avoids near-duplicate chunks)
results = retriever.retrieve_with_mmr("What is Python?", top_k=5, lambda_mult=0.5)
```

**MMR lambda_mult guide:**
- `1.0` → pure relevance (same as standard retrieval)
- `0.5` → balanced (recommended)
- `0.0` → maximum diversity

### RAGGenerator

```python
from rag.generator import RAGGenerator, GenerationConfig

config = GenerationConfig(model="llama3-8b-8192", temperature=0.2, max_tokens=1024)
generator = RAGGenerator(provider="groq", config=config)

response = generator.generate(question, retrieved_docs)
print(response.answer)
print(response.format_sources())

# Streaming
for token in generator.stream_generate(question, retrieved_docs):
    print(token, end="", flush=True)
```

**Provider setup:**
| Provider | Setup | Models |
|---|---|---|
| **Groq** (recommended) | Free API key at [console.groq.com](https://console.groq.com) | llama3-8b, llama3-70b, mixtral |
| **OpenAI** | Paid API key | gpt-4o-mini, gpt-4o |
| **Ollama** | Install [ollama.com](https://ollama.com), run `ollama pull llama3` | Any local model |

---

## 🖥️ Streamlit UI Features

- Upload multiple PDF/TXT files directly in the browser
- Configurable chunk size, retrieval settings, LLM parameters — all from sidebar
- Streaming responses with live token display
- Source citations with similarity scores per response
- Toggle MMR for diverse context retrieval
- Persistent chat history within session

---

## 🧠 Key Design Decisions

**Why cosine similarity?**
Vectors are L2-normalized before storage (`normalize_embeddings=True`), making cosine similarity equivalent to dot product — faster and numerically stable.

**Why ChromaDB?**
Lightweight, no external server needed, persists to disk, supports metadata filtering. Perfect for single-machine / demo deployments. For production scale, swap to Pinecone, Weaviate, or pgvector.

**Why RecursiveCharacterTextSplitter?**
It tries natural boundaries first (paragraphs → sentences → words) before falling back to character splitting, preserving semantic coherence better than fixed-size chunking.

**Why MMR?**
Top-K retrieval can return 5 nearly identical chunks from the same paragraph. MMR penalizes redundancy, ensuring the LLM sees diverse context — especially important for multi-topic queries.

---

## 📊 Evaluation

Built-in RAGAS-style evaluation with 5 metrics — no external dependencies needed.

| Metric | What it measures | Needs ground truth |
|---|---|---|
| **Faithfulness** | Are answer claims supported by the retrieved context? | No |
| **Answer Relevancy** | Does the answer address the question? | No |
| **Context Precision** | Are retrieved chunks actually relevant to the query? | No |
| **Context Recall** | Does the context cover the ground-truth answer? | Yes |
| **Answer Correctness** | Semantic similarity of answer to reference answer | Yes |

**Single query evaluation:**
```python
from evaluate import evaluate_pipeline_response

report = evaluate_pipeline_response(
    pipeline=pipeline,
    question="What is the attention mechanism?",
    ground_truth="Attention weights positions using query-key dot products.",
)
print(report.summary())
```

**Output:**
```
============================================================
QUESTION : What is the attention mechanism?
ANSWER   : The attention mechanism maps queries to key-value pairs...
------------------------------------------------------------
METRICS:
  faithfulness              ████████████████░░░░  0.821
  answer_relevancy          ██████████████████░░  0.903
  context_precision         ███████████████░░░░░  0.750
  context_recall            █████████████████░░░  0.867
  answer_correctness        ████████████████░░░░  0.812
------------------------------------------------------------
  MEAN SCORE                ████████████████░░░░  0.831
  Latency: 1243 ms
============================================================
```

**Batch evaluation with CSV export:**
```python
from evaluate import RAGEvaluator, EvalSample

samples = [EvalSample(question=q, answer=a, contexts=c, ground_truth=g) for ...]
evaluator = RAGEvaluator(embedding_manager=embedder, generator=gen)
batch = evaluator.evaluate_batch(samples, output_csv="results.csv")
print(batch.summary())
```

---

## 📄 License

MIT
