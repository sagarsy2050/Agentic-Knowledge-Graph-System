import json
from typing import List, Dict, Optional
from loguru import logger
from core.llm_engine import LLM


class ReasoningAgent:

    def __init__(self, graph, vector_store=None):
        self.graph = graph
        self.vector_store = vector_store
        self.conversation_history = []

    def answer_question(self, question, use_graph=True):
        logger.info(f"Answering: {question}")
        sources = []
        graph_context = ""

        if use_graph:
            keywords = self._extract_keywords(question)
            all_nodes = []
            for kw in keywords[:5]:
                nodes = self.graph.search_nodes(kw, limit=5)
                all_nodes.extend(nodes)
            seen = set()
            unique_nodes = []
            for n in all_nodes:
                nid = n.get("id", n.get("name", ""))
                if nid not in seen:
                    seen.add(nid)
                    unique_nodes.append(n)
            sources = unique_nodes[:15]
            graph_context = self._format_nodes_for_context(sources)

        system_prompt = """You are an expert scientific research assistant.
Answer questions based on the provided knowledge graph context.
Be precise, cite sources when available, acknowledge uncertainty."""

        prompt = f"""Research Question: {question}

KNOWLEDGE GRAPH CONTEXT:
{graph_context if graph_context else "No specific graph data found. Answer from general knowledge."}

Provide a comprehensive answer including:
1. Direct answer
2. Supporting evidence from context
3. Related topics worth exploring
4. Any limitations or caveats

Answer:"""

        answer_text = LLM.generate(prompt, system=system_prompt, temperature=0.2, max_tokens=2048)

        self.conversation_history.append({"role": "user", "content": question})
        self.conversation_history.append({"role": "assistant", "content": answer_text})

        return {
            "question": question,
            "answer": answer_text,
            "sources": sources[:10],
            "graph_nodes_used": len(sources),
            "keywords_used": keywords if use_graph else []
        }

    def generate_hypothesis(self, topic, constraints=None):
        logger.info(f"Generating hypotheses for: {topic}")
        nodes = self.graph.search_nodes(topic, limit=20)
        context = self._format_nodes_for_context(nodes)
        constraints_text = f"\nConstraints: {constraints}" if constraints else ""

        prompt = f"""Based on the following knowledge graph data about "{topic}", generate novel testable scientific hypotheses.

KNOWLEDGE GRAPH DATA:
{context}
{constraints_text}

Generate 3-5 novel research hypotheses as JSON:
{{
  "hypotheses": [
    {{
      "hypothesis": "Clear testable hypothesis statement",
      "rationale": "Why this is interesting",
      "novelty": "What makes this novel",
      "testing_approach": "How could this be tested",
      "impact": "Potential scientific impact",
      "confidence": 0.8
    }}
  ]
}}"""

        result = LLM.extract_json(prompt)
        hypotheses = result.get("hypotheses", [])

        for i, hyp in enumerate(hypotheses):
            hyp_id = f"hyp_{topic[:20].replace(' ','_')}_{i}"
            self.graph.add_node(hyp_id, "Hypothesis", {
                "name": hyp.get("hypothesis", "")[:200],
                "text": hyp.get("hypothesis", ""),
                "rationale": hyp.get("rationale", ""),
                "status": "generated",
                "generated_for": topic
            })

        logger.success(f"Generated {len(hypotheses)} hypotheses")
        return hypotheses

    def generate_literature_review(self, topic, style="academic"):
        logger.info(f"Generating literature review for: {topic}")
        paper_nodes = self.graph.search_nodes(topic, label="Paper", limit=20)
        concept_nodes = self.graph.search_nodes(topic, label="Concept", limit=15)
        papers_context = self._format_papers_for_review(paper_nodes)
        concepts_context = self._format_concepts_for_review(concept_nodes)

        style_instructions = {
            "academic": "Write in formal academic style with clear sections.",
            "survey": "Write as a comprehensive survey paper with clear categorization.",
            "executive": "Write as an executive summary for non-specialists using plain language.",
            "technical": "Write in technical detail for domain experts."
        }.get(style, "Write in formal academic style.")

        prompt = f"""Generate a comprehensive literature review on "{topic}".

{style_instructions}

PAPERS IN KNOWLEDGE GRAPH:
{papers_context}

KEY CONCEPTS:
{concepts_context}

Write a structured literature review with these sections:
1. Introduction and Motivation
2. Background and Definitions
3. Key Approaches and Methods
4. Major Findings and Contributions
5. Challenges and Open Problems
6. Future Directions
7. Conclusion"""

        review = LLM.generate(prompt, temperature=0.3, max_tokens=4096)
        return review

    def analyze_research_landscape(self, topic):
        nodes = self.graph.search_nodes(topic, limit=50)
        stats = self.graph.get_stats()
        papers = [n for n in nodes if n.get("label") == "Paper" or "Paper" in n.get("labels", [])]
        concepts = [n for n in nodes if n.get("label") == "Concept"]
        methods = [n for n in nodes if n.get("label") == "Method"]

        context = f"""
Papers found: {len(papers)}
Concepts identified: {len(concepts)}
Methods: {len(methods)}
Sample papers: {self._format_papers_for_review(papers[:8])}
Key concepts: {', '.join([c.get('name', '') for c in concepts[:15]])}
"""
        prompt = f"""Analyze the research landscape for "{topic}" based on this knowledge graph data.

{context}

Return JSON:
{{
  "overview": "2-3 sentence overview of the field",
  "maturity": "emerging|growing|mature|declining",
  "key_themes": ["theme1", "theme2", "theme3"],
  "dominant_approaches": ["approach1", "approach2"],
  "key_challenges": ["challenge1", "challenge2"],
  "trend_direction": "where field is heading",
  "hot_topics": ["active research areas"]
}}"""

        result = LLM.extract_json(prompt)
        result["stats"] = stats
        result["topic"] = topic
        return result

    def compare_approaches(self, approach1, approach2):
        nodes1 = self.graph.search_nodes(approach1, limit=10)
        nodes2 = self.graph.search_nodes(approach2, limit=10)
        context1 = self._format_nodes_for_context(nodes1)
        context2 = self._format_nodes_for_context(nodes2)

        prompt = f"""Compare these two research approaches.

APPROACH 1: {approach1}
{context1}

APPROACH 2: {approach2}
{context2}

Return JSON:
{{
  "similarities": ["similar aspect 1"],
  "differences": [{{"aspect": "performance", "approach1": "...", "approach2": "..."}}],
  "use_cases_1": ["when to use approach 1"],
  "use_cases_2": ["when to use approach 2"],
  "recommendation": "when to prefer which and why",
  "hybrid_potential": "can they be combined?"
}}"""
        return LLM.extract_json(prompt)

    def chat(self, message):
        quick_search = self.graph.search_nodes(message[:100], limit=5)
        context_snippet = self._format_nodes_for_context(quick_search) if quick_search else ""
        messages = self.conversation_history[-10:]
        system = """You are an expert AI research assistant with access to a scientific knowledge graph.
Help users explore research topics, find connections, and generate insights."""
        if context_snippet:
            full_message = f"{message}\n\n[Relevant KG context: {context_snippet[:500]}]"
        else:
            full_message = message
        messages.append({"role": "user", "content": full_message})
        response = LLM.chat(messages, temperature=0.3)
        self.conversation_history.append({"role": "user", "content": message})
        self.conversation_history.append({"role": "assistant", "content": response})
        return response

    def _extract_keywords(self, text):
        prompt = f"""Extract 3-5 key search terms from this research question as a JSON array.
Question: {text}
Return only JSON array like: ["term1", "term2", "term3"]"""
        result = LLM.extract_json(prompt)
        if isinstance(result, list):
            return result[:5]
        return text.split()[:4]

    def _format_nodes_for_context(self, nodes):
        if not nodes:
            return "No relevant nodes found."
        parts = []
        for node in nodes[:15]:
            name = node.get("name", node.get("title", "Unknown"))
            desc = node.get("description", node.get("abstract", ""))[:200]
            label = node.get("label", "Node")
            parts.append(f"[{label}] {name}: {desc}")
        return "\n".join(parts)

    def _format_papers_for_review(self, papers):
        if not papers:
            return "No papers found."
        parts = []
        for paper in papers[:20]:
            title = paper.get("title", "Unknown")
            year = paper.get("year", "")
            authors = paper.get("authors", "")
            abstract = paper.get("abstract", "")[:300]
            parts.append(f"- {title} ({authors}, {year}): {abstract}")
        return "\n".join(parts)

    def _format_concepts_for_review(self, concepts):
        if not concepts:
            return ""
        return ", ".join([c.get("name", "") for c in concepts if c.get("name")])
