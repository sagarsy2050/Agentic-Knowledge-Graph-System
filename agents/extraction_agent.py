import hashlib
import re
from typing import List, Dict, Optional
from loguru import logger
from core.llm_engine import LLM


class ExtractionAgent:

    def extract_entities(self, text, domain="science"):
        prompt = f"""Extract all scientific entities from this {domain} text.

TEXT:
{text[:3000]}

Return JSON array of entities:
{{
  "entities": [
    {{
      "name": "entity name",
      "type": "one of: Concept|Method|Material|Dataset|Algorithm|Theory|Chemical|Disease|Gene|Protein|Tool|Metric",
      "description": "brief description from text",
      "importance": "high|medium|low"
    }}
  ]
}}"""
        result = LLM.extract_json(prompt, '{"entities": [...]}')
        entities = result.get("entities", [])
        for ent in entities:
            ent["id"] = self._make_id(ent.get("name", ""))
        logger.info(f"Extracted {len(entities)} entities")
        return entities

    def extract_relations(self, text, entities):
        if len(entities) < 2:
            return []
        entity_list = "\n".join([f"- {e['name']} ({e.get('type','Entity')})" for e in entities[:30]])
        prompt = f"""Given these entities from scientific text, identify relationships between them.

ENTITIES:
{entity_list}

TEXT EXCERPT:
{text[:2000]}

Return JSON:
{{
  "relations": [
    {{
      "source": "entity name 1",
      "target": "entity name 2",
      "relation": "one of: USES|PRODUCES|IMPROVES|EXTENDS|PART_OF|CAUSES|INHIBITS|VALIDATES|CONTRADICTS|SUPPORTS|DERIVES_FROM|APPLIES_TO|RELATED_TO",
      "description": "brief explanation",
      "confidence": 0.8
    }}
  ]
}}"""
        result = LLM.extract_json(prompt, '{"relations": [...]}')
        relations = result.get("relations", [])
        logger.info(f"Extracted {len(relations)} relations")
        return relations

    def extract_claims(self, text):
        prompt = f"""Extract key scientific claims and findings from this text.

TEXT:
{text[:3000]}

Return JSON:
{{
  "claims": [
    {{
      "text": "the claim or finding statement",
      "type": "finding|hypothesis|limitation|contribution|future_work",
      "confidence": "high|medium|speculative"
    }}
  ]
}}"""
        result = LLM.extract_json(prompt, '{"claims": [...]}')
        claims = result.get("claims", [])
        for claim in claims:
            claim["id"] = f"claim_{hashlib.md5(claim.get('text','').encode()).hexdigest()[:8]}"
        logger.info(f"Extracted {len(claims)} claims")
        return claims

    def extract_paper_metadata(self, text, title=""):
        prompt = f"""Extract structured metadata from this research paper.

TITLE: {title}
TEXT: {text[:2000]}

Return JSON:
{{
  "title": "paper title",
  "year": 2024,
  "domain": "primary research domain",
  "problem": "what problem does this paper solve",
  "approach": "methodology used",
  "key_contributions": ["contribution 1"],
  "keywords": ["keyword1", "keyword2"]
}}"""
        result = LLM.extract_json(prompt)
        return result

    def identify_research_gaps(self, papers_summaries):
        combined = "\n\n---\n\n".join(papers_summaries[:10])
        prompt = f"""Analyze these research paper summaries and identify research gaps.

PAPERS:
{combined[:4000]}

Return JSON:
{{
  "gaps": [
    {{
      "description": "description of the research gap",
      "importance": "high|medium|low",
      "suggested_approach": "potential way to address this gap"
    }}
  ],
  "emerging_themes": ["theme1", "theme2"]
}}"""
        result = LLM.extract_json(prompt)
        return result.get("gaps", [])

    def generate_summary(self, text, max_words=150):
        prompt = f"""Summarize this scientific text in {max_words} words or less.
Focus on: main problem, approach, key finding, significance.
TEXT: {text[:3000]}
Write only the summary."""
        return LLM.generate(prompt, temperature=0.2)

    def link_entities_to_graph(self, entities, relations, claims, graph, paper_id=None):
        added = {"entities": 0, "relations": 0, "claims": 0}
        name_to_id = {}

        for entity in entities:
            node_id = entity.get("id", self._make_id(entity.get("name", "")))
            name_to_id[entity.get("name", "")] = node_id
            properties = {
                "name": entity.get("name", ""),
                "description": entity.get("description", ""),
                "importance": entity.get("importance", "medium"),
            }
            graph.add_node(node_id, entity.get("type", "Entity"), properties)
            if paper_id:
                graph.add_edge(paper_id, node_id, "HAS_ENTITY", {})
            added["entities"] += 1

        for rel in relations:
            source_id = name_to_id.get(rel.get("source", ""))
            target_id = name_to_id.get(rel.get("target", ""))
            if source_id and target_id and float(rel.get("confidence", 0)) >= 0.5:
                props = {
                    "description": str(rel.get("description", ""))[:500],
                    "confidence": float(rel.get("confidence", 0.7))
                }
                graph.add_edge(source_id, target_id, rel.get("relation", "RELATED_TO"), props)
                added["relations"] += 1

        for claim in claims:
            claim_id = claim.get("id", self._make_id(claim.get("text", "")[:50]))
            props = {
                "name": claim.get("text", "")[:200],
                "text": claim.get("text", ""),
                "claim_type": claim.get("type", "finding"),
                "confidence": claim.get("confidence", "medium")
            }
            graph.add_node(claim_id, "Hypothesis", props)
            if paper_id:
                graph.add_edge(paper_id, claim_id, "MAKES_CLAIM", {})
            added["claims"] += 1

        logger.success(f"Linked to graph: {added}")
        return added

    def _make_id(self, name):
        clean = re.sub(r'[^a-z0-9]', '_', name.lower().strip())
        return f"{clean[:40]}_{hashlib.md5(name.encode()).hexdigest()[:6]}"
