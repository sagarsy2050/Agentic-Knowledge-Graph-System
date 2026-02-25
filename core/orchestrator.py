import time
import json
from pathlib import Path
from typing import List, Dict, Optional, Callable
from loguru import logger

from core.llm_engine import LLM
from graph.knowledge_graph import create_knowledge_graph
from agents.research_agent import ResearchAgent
from agents.extraction_agent import ExtractionAgent
from agents.reasoning_agent import ReasoningAgent
from config.settings import CONFIG


class AgenticKGOrchestrator:

    def __init__(self, on_progress=None):
        self.on_progress = on_progress or (lambda msg, pct: logger.info(f"[{pct}%] {msg}"))
        logger.info("Initializing Agentic KG System...")
        self.llm = LLM
        self.graph = create_knowledge_graph()
        self.research_agent = ResearchAgent()
        self.extraction_agent = ExtractionAgent()
        self.reasoning_agent = ReasoningAgent(self.graph)
        self.session_stats = {
            "papers_processed": 0,
            "entities_extracted": 0,
            "relations_found": 0,
            "claims_made": 0,
            "topics_researched": []
        }
        logger.success("System initialized and ready")

    def check_system(self):
        status = {"ollama": False, "graph": False, "models": []}
        if self.llm.check_connection():
            status["ollama"] = True
            status["models"] = self.llm.list_models()
        try:
            stats = self.graph.get_stats()
            status["graph"] = True
            status["graph_stats"] = stats
        except:
            status["graph"] = False
        return status

    def ensure_models(self, models=None):
        if models is None:
            models = [CONFIG.ollama.primary_model, CONFIG.ollama.embedding_model]
        available = self.llm.list_models()
        for model in models:
            model_base = model.split(":")[0]
            if not any(model_base in m for m in available):
                logger.info(f"Model {model} not found. Pulling...")
                self.llm.pull_model(model)

    def run_research_pipeline(self, topic, max_papers=10, sources=None,
                               local_dir=None, year_from=None):
        if sources is None:
            sources = ["arxiv"]
        start_time = time.time()
        results = {
            "topic": topic,
            "papers": [],
            "entities": [],
            "relations": [],
            "claims": [],
            "insights": {},
            "stats": {}
        }

        self.on_progress(f"Starting research pipeline for: '{topic}'", 0)

        # Step 1: Collect Papers
        self.on_progress("Fetching papers from sources...", 10)
        all_papers = []

        if sources:
            papers = self.research_agent.research_topic(
                topic, max_papers=max_papers, sources=sources, year_from=year_from
            )
            all_papers.extend(papers)

        if local_dir and Path(local_dir).exists():
            self.on_progress("Loading local documents...", 20)
            local_papers = self.research_agent.load_local_papers(local_dir)
            all_papers.extend(local_papers)

        if not all_papers:
            logger.warning("No papers found!")
            return results

        logger.info(f"Total papers to process: {len(all_papers)}")
        results["papers"] = all_papers

        # Step 2: Add Topic Node
        topic_id = f"topic_{''.join(c if c.isalnum() else '_' for c in topic[:30]).lower()}"
        self.graph.add_node(topic_id, "Topic", {
            "name": topic,
            "description": f"Research topic: {topic}",
            "paper_count": len(all_papers),
            "created_at": time.strftime("%Y-%m-%d")
        })

        # Step 3: Process Each Paper
        self.on_progress("Extracting knowledge from papers...", 30)
        all_entities = []
        all_relations = []
        all_claims = []

        for i, paper in enumerate(all_papers):
            progress = 30 + int((i / len(all_papers)) * 50)
            paper_title = paper.get("title", "Unknown")[:60]
            self.on_progress(f"Processing [{i+1}/{len(all_papers)}]: {paper_title}", progress)

            paper_node = self.research_agent.build_paper_node(paper)
            paper_id = paper_node["id"]
            self.graph.add_node(paper_id, "Paper", paper_node)
            self.graph.add_edge(topic_id, paper_id, "HAS_PAPER", {"relevance": 1.0})

            # Add authors
            authors = paper.get("authors", [])
            if isinstance(authors, list):
                for author_name in authors[:5]:
                    if author_name:
                        author_id = "author_" + "".join(c if c.isalnum() else '_' for c in author_name[:30]).lower()
                        self.graph.add_node(author_id, "Author", {"name": author_name})
                        self.graph.add_edge(paper_id, author_id, "AUTHORED_BY", {})

            # Get text for extraction
            text = self.research_agent.get_paper_full_text(paper)
            if not text.strip():
                continue

            # Extract knowledge with LLM
            entities = self.extraction_agent.extract_entities(text, domain=topic)
            relations = self.extraction_agent.extract_relations(text, entities) if entities else []
            claims = self.extraction_agent.extract_claims(text)

            # Store in graph
            self.extraction_agent.link_entities_to_graph(
                entities, relations, claims, self.graph, paper_id=paper_id
            )

            all_entities.extend(entities)
            all_relations.extend(relations)
            all_claims.extend(claims)

            self.session_stats["papers_processed"] += 1
            self.session_stats["entities_extracted"] += len(entities)
            self.session_stats["relations_found"] += len(relations)
            self.session_stats["claims_made"] += len(claims)

            time.sleep(0.2)

        results["entities"] = all_entities
        results["relations"] = all_relations
        results["claims"] = all_claims

        # Step 4: Generate Insights
        self.on_progress("Generating research insights...", 85)
        landscape = self.reasoning_agent.analyze_research_landscape(topic)
        summaries = [
            f"{p.get('title','')} ({p.get('year','')}): {p.get('abstract','')[:200]}"
            for p in all_papers[:10]
        ]
        gaps = self.extraction_agent.identify_research_gaps(summaries)

        results["insights"] = {
            "landscape": landscape,
            "research_gaps": gaps,
            "paper_count": len(all_papers),
            "entity_count": len(all_entities),
            "relation_count": len(all_relations)
        }

        # Step 5: Final Stats
        self.on_progress("Finalizing...", 95)
        elapsed = time.time() - start_time
        graph_stats = self.graph.get_stats()

        results["stats"] = {
            "elapsed_seconds": round(elapsed, 1),
            "papers_collected": len(all_papers),
            "entities_extracted": len(all_entities),
            "relations_found": len(all_relations),
            "claims_found": len(all_claims),
            "graph_total_nodes": graph_stats.get("total_nodes", 0),
            "graph_total_edges": graph_stats.get("total_edges", 0)
        }

        self.session_stats["topics_researched"].append(topic)
        self.on_progress(f"Pipeline complete! Processed {len(all_papers)} papers in {elapsed:.0f}s", 100)
        logger.success(f"Pipeline results: {results['stats']}")
        return results

    def ask(self, question):
        return self.reasoning_agent.answer_question(question)

    def chat(self, message):
        return self.reasoning_agent.chat(message)

    def generate_hypotheses(self, topic):
        return self.reasoning_agent.generate_hypothesis(topic)

    def write_literature_review(self, topic, style="academic"):
        return self.reasoning_agent.generate_literature_review(topic, style)

    def compare(self, approach1, approach2):
        return self.reasoning_agent.compare_approaches(approach1, approach2)

    def get_graph_data(self, query="", limit=50):
        nodes = self.graph.search_nodes(query, limit=limit) if query else self.graph.search_nodes("", limit=limit)
        return {"nodes": nodes, "stats": self.graph.get_stats()}

    def get_session_stats(self):
        return {**self.session_stats, **self.graph.get_stats()}

    def export_graph(self, format="json", path=None):
        export_path = path or str(CONFIG.export_dir / f"kg_export.{format}")
        if format == "graphml":
            if hasattr(self.graph, 'export_graphml'):
                self.graph.export_graphml(export_path)
        elif format == "json":
            data = {
                "stats": self.graph.get_stats(),
                "session": self.session_stats
            }
            with open(export_path, 'w') as f:
                json.dump(data, f, indent=2, default=str)
        logger.success(f"Graph exported to: {export_path}")
        return export_path
