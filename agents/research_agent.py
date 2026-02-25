import hashlib
import time
from pathlib import Path
from typing import List, Dict, Optional
from loguru import logger
import httpx
from config.settings import CONFIG


class ArXivFetcher:

    BASE_URL = "https://export.arxiv.org/api/query"

    def search(self, query, max_results=10, sort_by="relevance"):
        params = {
            "search_query": f"all:{query}",
            "max_results": max_results,
            "sortBy": sort_by,
            "sortOrder": "descending"
        }
        try:
            logger.info(f"Searching ArXiv: '{query}' (max={max_results})")
            response = httpx.get(
                self.BASE_URL,
                params=params,
                timeout=30,
                follow_redirects=True
            )
            if response.status_code != 200:
                logger.error(f"ArXiv API error: {response.status_code}")
                return []
            papers = self._parse_arxiv_response(response.text)
            logger.success(f"Found {len(papers)} papers on ArXiv")
            return papers
        except Exception as e:
            logger.error(f"ArXiv search failed: {e}")
            return []

    def _parse_arxiv_response(self, xml_text):
        import xml.etree.ElementTree as ET
        papers = []
        ns = {
            'atom': 'http://www.w3.org/2005/Atom',
        }
        try:
            root = ET.fromstring(xml_text)
            for entry in root.findall('atom:entry', ns):
                paper = {}
                title_el = entry.find('atom:title', ns)
                paper['title'] = title_el.text.strip().replace('\n', ' ') if title_el is not None else ""
                summary_el = entry.find('atom:summary', ns)
                paper['abstract'] = summary_el.text.strip() if summary_el is not None else ""
                id_el = entry.find('atom:id', ns)
                if id_el is not None:
                    arxiv_url = id_el.text.strip()
                    paper['url'] = arxiv_url
                    paper['arxiv_id'] = arxiv_url.split('/')[-1]
                    paper['id'] = f"arxiv_{paper['arxiv_id']}"
                authors = []
                for author_el in entry.findall('atom:author', ns):
                    name_el = author_el.find('atom:name', ns)
                    if name_el is not None:
                        authors.append(name_el.text)
                paper['authors'] = authors
                pub_el = entry.find('atom:published', ns)
                if pub_el is not None:
                    pub_str = pub_el.text[:10]
                    paper['year'] = int(pub_str[:4])
                    paper['published_date'] = pub_str
                cats = entry.findall('atom:category', ns)
                paper['categories'] = [c.get('term', '') for c in cats]
                for link in entry.findall('atom:link', ns):
                    if link.get('type') == 'application/pdf':
                        paper['pdf_url'] = link.get('href', '')
                        break
                paper['source'] = 'arxiv'
                if paper.get('title'):
                    papers.append(paper)
        except ET.ParseError as e:
            logger.error(f"XML parse error: {e}")
        return papers

    def fetch_by_id(self, arxiv_id):
        params = {"id_list": arxiv_id, "max_results": 1}
        try:
            response = httpx.get(self.BASE_URL, params=params, timeout=30, follow_redirects=True)
            papers = self._parse_arxiv_response(response.text)
            return papers[0] if papers else None
        except Exception as e:
            logger.error(f"ArXiv fetch failed for {arxiv_id}: {e}")
            return None


class SemanticScholarFetcher:

    BASE_URL = "https://api.semanticscholar.org/graph/v1"

    def search(self, query, max_results=10):
        params = {
            "query": query,
            "limit": max_results,
            "fields": "title,abstract,year,authors,citationCount,url"
        }
        try:
            response = httpx.get(
                f"{self.BASE_URL}/paper/search",
                params=params,
                timeout=30,
                follow_redirects=True,
                headers={"User-Agent": "ResearchKG/1.0"}
            )
            if response.status_code == 200:
                data = response.json()
                papers = []
                for item in data.get("data", []):
                    paper = {
                        "id": f"s2_{item.get('paperId', '')}",
                        "title": item.get("title", ""),
                        "abstract": item.get("abstract", "") or "",
                        "year": item.get("year", 0),
                        "authors": [a.get("name", "") for a in item.get("authors", [])],
                        "citations": item.get("citationCount", 0),
                        "url": item.get("url", ""),
                        "source": "semantic_scholar"
                    }
                    papers.append(paper)
                logger.success(f"Semantic Scholar: found {len(papers)} papers")
                return papers
            else:
                logger.warning(f"Semantic Scholar returned {response.status_code}")
                return []
        except Exception as e:
            logger.warning(f"Semantic Scholar unavailable: {e}")
            return []


class LocalDocumentLoader:

    def load_pdf(self, file_path):
        try:
            from pypdf import PdfReader
            reader = PdfReader(file_path)
            text_parts = [page.extract_text() for page in reader.pages if page.extract_text()]
            return {
                "id": f"local_{hashlib.md5(file_path.encode()).hexdigest()[:8]}",
                "title": Path(file_path).stem,
                "text": "\n\n".join(text_parts),
                "abstract": "\n\n".join(text_parts)[:500],
                "pages": len(reader.pages),
                "source": "local_pdf",
                "file_path": file_path,
                "authors": [],
                "year": 0
            }
        except ImportError:
            logger.warning("pypdf not installed")
            return {}
        except Exception as e:
            logger.error(f"PDF load failed: {e}")
            return {}

    def load_text(self, file_path):
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                text = f.read()
            return {
                "id": f"local_{hashlib.md5(file_path.encode()).hexdigest()[:8]}",
                "title": Path(file_path).stem,
                "text": text,
                "abstract": text[:500],
                "source": "local_text",
                "file_path": file_path,
                "authors": [],
                "year": 0
            }
        except Exception as e:
            logger.error(f"Text load failed: {e}")
            return {}

    def load_directory(self, directory, extensions=None):
        if extensions is None:
            extensions = [".pdf", ".txt", ".md"]
        docs = []
        dir_path = Path(directory)
        for ext in extensions:
            for file_path in dir_path.glob(f"**/*{ext}"):
                logger.info(f"Loading: {file_path.name}")
                if ext == ".pdf":
                    doc = self.load_pdf(str(file_path))
                else:
                    doc = self.load_text(str(file_path))
                if doc:
                    docs.append(doc)
                time.sleep(0.1)
        logger.success(f"Loaded {len(docs)} documents from {directory}")
        return docs


class ResearchAgent:

    def __init__(self):
        self.arxiv = ArXivFetcher()
        self.semantic_scholar = SemanticScholarFetcher()
        self.local_loader = LocalDocumentLoader()

    def research_topic(self, topic, max_papers=15, sources=None, year_from=None):
        if sources is None:
            sources = ["arxiv"]
        logger.info(f"Researching topic: '{topic}'")
        all_papers = []
        seen_titles = set()
        papers_per_source = max(max_papers // len(sources), 5)

        for source in sources:
            if source == "arxiv":
                papers = self.arxiv.search(topic, max_results=papers_per_source + 5)
            elif source == "semantic_scholar":
                papers = self.semantic_scholar.search(topic, max_results=papers_per_source + 5)
            else:
                continue
            for paper in papers:
                title = paper.get("title", "").lower().strip()
                if title and title not in seen_titles:
                    if year_from and paper.get("year", 0) and paper.get("year", 0) < year_from:
                        continue
                    seen_titles.add(title)
                    all_papers.append(paper)
            time.sleep(0.5)

        all_papers = all_papers[:max_papers]
        logger.success(f"Collected {len(all_papers)} papers for '{topic}'")
        return all_papers

    def load_local_papers(self, directory):
        return self.local_loader.load_directory(directory)

    def get_paper_full_text(self, paper):
        parts = []
        if paper.get("title"):
            parts.append(f"TITLE: {paper['title']}")
        if paper.get("abstract"):
            parts.append(f"ABSTRACT:\n{paper['abstract']}")
        if paper.get("text"):
            parts.append(f"CONTENT:\n{paper['text'][:3000]}")
        return "\n\n".join(parts)

    def build_paper_node(self, paper):
        return {
            "id": paper.get("id", f"paper_{hashlib.md5(paper.get('title','').encode()).hexdigest()[:8]}"),
            "title": paper.get("title", "Unknown"),
            "abstract": paper.get("abstract", "")[:1000],
            "year": paper.get("year", 0),
            "authors": ", ".join(paper.get("authors", [])[:5]) if isinstance(paper.get("authors"), list) else str(paper.get("authors", "")),
            "citations": paper.get("citations", 0),
            "url": paper.get("url", ""),
            "source": paper.get("source", "unknown"),
            "categories": ", ".join(paper.get("categories", [])) if isinstance(paper.get("categories"), list) else "",
        }
