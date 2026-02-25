# 🧠 Agentic Knowledge Graph System
## Complete Command Reference

---

## ✅ BEFORE EVERY SESSION (Start These First)

**Step 1 — Ollama must be running** (skip if already running)
```
ollama serve
```
Open a NEW terminal window and keep this running in the background.

---

## 🔬 RESEARCH COMMANDS

### Research any scientific topic:
```
python cli.py research "transformer attention mechanisms" --max-papers 10
```
```
python cli.py research "CRISPR gene editing" --max-papers 10
```
```
python cli.py research "quantum computing error correction" --max-papers 10
```
```
python cli.py research "deep learning image classification" --max-papers 15
```
```
python cli.py research "protein folding algorithms" --max-papers 10
```

### Research with both ArXiv and Semantic Scholar:
```
python cli.py research "neural networks" --max-papers 10 --sources arxiv,semantic_scholar
```

### Research and save results to a file:
```
python cli.py research "transformer attention mechanisms" --max-papers 10 -o results.json
```

### Research papers from a specific year onwards:
```
python cli.py research "large language models" --max-papers 10 --year-from 2022
```

### Research from your own local PDF folder:
```
python cli.py research "my topic" --local-dir C:\Users\YourName\Documents\papers
```

---

## 💬 ASK QUESTIONS

### Ask anything about your researched topic:
```
python cli.py ask "What are the key methods in this field?"
```
```
python cli.py ask "What are the main challenges?"
```
```
python cli.py ask "What datasets are commonly used?"
```
```
python cli.py ask "What are the best performing approaches?"
```
```
python cli.py ask "What are the open research problems?"
```
```
python cli.py ask "Summarize the key findings"
```
```
python cli.py ask "Which papers are most important?"
```

---

## 💡 GENERATE HYPOTHESES

### Generate novel research ideas:
```
python cli.py hypotheses "transformer attention mechanisms"
```
```
python cli.py hypotheses "CRISPR gene editing"
```
```
python cli.py hypotheses "quantum error correction"
```

---

## 📖 GENERATE LITERATURE REVIEW

### Academic style (default):
```
python cli.py review "transformer attention mechanisms" --style academic -o review.md
```

### Survey paper style:
```
python cli.py review "transformer attention mechanisms" --style survey -o review.md
```

### Simple non-expert summary:
```
python cli.py review "transformer attention mechanisms" --style executive -o review.md
```

### Technical expert style:
```
python cli.py review "transformer attention mechanisms" --style technical -o review.md
```

---

## 💬 CHAT MODE (Ask multiple questions interactively)

```
python cli.py chat
```
Then just type your questions. Type `exit` to quit.

---

## 📊 CHECK GRAPH STATISTICS

```
python cli.py stats
```

---

## 🔧 CHECK SYSTEM STATUS

```
python cli.py setup
```

---

## 🌐 LAUNCH WEB UI (Full Visual Interface)

```
streamlit run ui/app.py
```
Then open browser and go to:
```
http://localhost:8501
```

---

## 🔁 TYPICAL WORKFLOW (Run in This Order)

```
python cli.py research "your topic here" --max-papers 10
```
```
python cli.py ask "What are the key findings?"
```
```
python cli.py ask "What are the main challenges?"
```
```
python cli.py hypotheses "your topic here"
```
```
python cli.py review "your topic here" --style academic -o my_review.md
```
```
python cli.py stats
```

---

## ⚠️ IF SOMETHING GOES WRONG

### Ollama not running error:
```
ollama serve
```

### Import errors — wrong folder:
```
cd E:\kN graphes
python cli.py setup
```

### Check Ollama models installed:
```
ollama list
```

### Pull a model if missing:
```
ollama pull llama3.1:8b
```
```
ollama pull nomic-embed-text
```

---

## 📁 OUTPUT FILES

| File | What it contains |
|------|-----------------|
| `results.json` | Full pipeline results with all papers and entities |
| `review.md` | Generated literature review (open in any text editor) |
| `exports/` folder | Knowledge graph exports |

---

*System runs 100% locally. No internet needed after paper fetch. No API keys required.*
