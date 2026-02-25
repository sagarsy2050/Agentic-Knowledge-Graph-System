# 🧠 Agentic Knowledge Graph System
### End-to-End Research Intelligence Platform — 100% Local with Ollama LLM

> Build, query, and reason over scientific knowledge graphs — entirely on your local machine.
> No API keys. No cloud. Just powerful local AI.

---

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    AGENTIC KG SYSTEM                            │
│                                                                 │
│  📥 INPUT SOURCES          🤖 AI AGENTS             📤 OUTPUT  │
│  ┌─────────────┐          ┌─────────────┐          ┌────────┐  │
│  │   ArXiv     │          │  Research   │          │  Q&A   │  │
│  │ Semantic S. │ ────────▶│   Agent     │──────┐   │ Hypo.  │  │
│  │ Local PDFs  │          └─────────────┘      │   │ Review │  │
│  └─────────────┘                               │   └────────┘  │
│                            ┌─────────────┐     │               │
│                            │ Extraction  │     ▼               │
│                            │   Agent     │  ┌──────────────┐   │
│                            └──────┬──────┘  │  Knowledge   │   │
│                                   │         │    Graph     │   │
│  🧠 LOCAL LLM              ┌──────▼──────┐  │  (Neo4j /   │   │
│  ┌─────────────┐          │  Reasoning  │  │  NetworkX)  │   │
│  │   Ollama    │          │   Agent     │  └──────────────┘   │
│  │ llama3.1:8b │ ◀────────│             │◀─────────────────   │
│  │ nomic-embed │          └─────────────┘                     │
│  └─────────────┘                                              │
└─────────────────────────────────────────────────────────────────┘
```

## 🚀 Quick Start

### 1. Install Ollama (Local LLM)
```bash
# Linux/Mac
curl -fsSL https://ollama.ai/install.sh | sh

# Windows: Download from https://ollama.ai

# Start Ollama
ollama serve

# Pull required models
ollama pull llama3.1:8b          # Main reasoning model (~4.7GB)
ollama pull llama3.2:3b          # Lightweight alternative (~2GB)
ollama pull nomic-embed-text     # Embedding model (~274MB)
```

### 2. Install & Run the System
```bash
# Clone or download this project
cd agentic-kg-system

# Run automated setup
chmod +x setup.sh && ./setup.sh

# OR manual install:
pip install -r requirements.txt

# Launch Web UI
streamlit run ui/app.py
# → Open: http://localhost:8501
```

### 3. Optional: Neo4j (Persistent Graph)
```bash
# With Docker (recommended)
docker-compose up -d neo4j
# Neo4j Browser: http://localhost:7474
# Username: neo4j | Password: password123

# Without Docker: https://neo4j.com/download/
```

---

## 📁 Project Structure

```
agentic-kg-system/
│
├── core/
│   ├── llm_engine.py        # Ollama LLM interface (generate, embed, chat)
│   └── orchestrator.py      # Master pipeline coordinator
│
├── agents/
│   ├── research_agent.py    # Fetch papers (ArXiv, Semantic Scholar, PDFs)
│   ├── extraction_agent.py  # Extract entities, relations, claims with LLM
│   └── reasoning_agent.py   # Q&A, hypotheses, literature review
│
├── graph/
│   └── knowledge_graph.py   # Neo4j + NetworkX fallback graph
│
├── ui/
│   └── app.py               # Streamlit web interface
│
├── config/
│   └── settings.py          # System configuration
│
├── cli.py                   # Command line interface
├── requirements.txt         # Python dependencies
├── docker-compose.yml       # Neo4j container
└── setup.sh                 # Automated setup script
```

---

## 💻 CLI Usage

```bash
# Check system status
python cli.py setup

# Research a topic (fetches & processes papers automatically)
python cli.py research "transformer attention mechanisms" --max-papers 15
python cli.py research "CRISPR gene therapy" --sources arxiv,semantic_scholar
python cli.py research "quantum computing" --year-from 2022 --output results.json

# Load local PDF papers
python cli.py research "my topic" --local-dir /path/to/my/papers/

# Ask questions about your knowledge graph
python cli.py ask "What are the main challenges in this field?"
python cli.py ask "Which methods show the best performance?"

# Generate novel research hypotheses
python cli.py hypotheses "protein folding mechanisms"

# Auto-generate literature review
python cli.py review "deep reinforcement learning" --style survey -o review.md
python cli.py review "cancer immunotherapy" --style executive

# Interactive chat
python cli.py chat

# View graph statistics
python cli.py stats
```

---

## 🌐 Web UI Features

| Tab | Feature |
|-----|---------|
| 🔬 Research Pipeline | Automated paper fetching, extraction, and KG building |
| 💬 Chat & Q&A | Conversational interface with KG-grounded answers |
| 🗺️ Knowledge Graph | Browse, search, visualize your knowledge graph |
| 💡 Hypotheses | AI-generated novel research hypotheses |
| 📖 Literature Review | Auto-generated literature reviews (4 styles) |
| 📊 Analytics | System stats, node/edge counts, session tracking |

---

## ⚙️ Configuration

Edit `config/settings.py` to customize:

```python
# Change LLM model
CONFIG.ollama.primary_model = "llama3.1:70b"    # More powerful
CONFIG.ollama.primary_model = "mistral:7b"       # Alternative
CONFIG.ollama.primary_model = "phi3:mini"        # Fastest

# Adjust pipeline settings
CONFIG.agents.max_research_papers = 20
CONFIG.agents.confidence_threshold = 0.8

# Neo4j credentials
CONFIG.neo4j.password = "your_password"
```

---

## 🤖 Supported Ollama Models

| Model | Size | Best For |
|-------|------|----------|
| `llama3.1:8b` | 4.7GB | Best balance (recommended) |
| `llama3.2:3b` | 2.0GB | Fast, lower RAM systems |
| `llama3.1:70b` | 40GB | Best quality (needs GPU) |
| `mistral:7b` | 4.1GB | Great reasoning |
| `phi3:mini` | 2.3GB | Very fast, good quality |
| `gemma2:9b` | 5.4GB | Strong analytical |
| `qwen2.5:7b` | 4.4GB | Excellent multilingual |
| `nomic-embed-text` | 274MB | Embeddings (required) |

---

## 🔧 System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| RAM | 8GB | 16GB+ |
| Storage | 10GB | 50GB+ |
| CPU | Any modern | 8+ cores |
| GPU | Not required | NVIDIA (speeds up LLM) |
| Python | 3.10+ | 3.11+ |

---

## 🔬 Example Research Workflow

```
1. Start: "I want to understand Graph Neural Networks for drug discovery"

2. System automatically:
   ├── Fetches 15 papers from ArXiv + Semantic Scholar
   ├── Extracts 200+ entities (proteins, algorithms, datasets, methods)
   ├── Finds 150+ relations (method USES dataset, paper EXTENDS theory)
   ├── Identifies 50+ scientific claims and findings
   └── Builds a rich knowledge graph

3. You can then:
   ├── Ask: "What GNN architectures work best for molecular graphs?"
   ├── Ask: "What datasets are most commonly used?"
   ├── Generate: Novel hypotheses about unexplored drug-protein interactions
   ├── Write: Full literature review in academic style
   └── Export: Knowledge graph as JSON/GraphML for further analysis
```

---

## 📊 Knowledge Graph Schema

```
Nodes:
  (:Paper)      - Research papers with title, abstract, year, authors
  (:Author)     - Researchers with name, affiliation
  (:Concept)    - Scientific concepts and ideas
  (:Method)     - Research methods and algorithms
  (:Dataset)    - Datasets used in research
  (:Entity)     - Any scientific entity (chemicals, genes, etc.)
  (:Hypothesis) - Claims, findings, and generated hypotheses
  (:Topic)      - Top-level research topics

Relations:
  AUTHORED_BY, CITES, USES_METHOD, HAS_ENTITY
  MAKES_CLAIM, RELATED_TO, EXTENDS, CONTRADICTS
  SUPPORTS, USES_DATASET, PART_OF, HAS_PAPER
```

---

## 🛠️ Troubleshooting

**Ollama not connecting:**
```bash
ollama serve   # Must be running in background
ollama list    # Check available models
```

**No papers found:**
- Check internet connection (ArXiv/S2 require internet)

- Try different search terms
- Use local PDFs with `--local-dir`

**Slow performance:**
- Use smaller model: `llama3.2:3b`
- Reduce `--max-papers`
- Enable GPU in Ollama: `CUDA_VISIBLE_DEVICES=0 ollama serve`

**Neo4j issues:**
- System works fine without Neo4j (uses in-memory graph)
- For persistence, use Docker: `docker-compose up -d neo4j`

---

