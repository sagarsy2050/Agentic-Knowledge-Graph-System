import os
from pathlib import Path
from dataclasses import dataclass, field

BASE_DIR = Path(__file__).parent.parent

@dataclass
class OllamaConfig:
    base_url: str = "http://localhost:11434"
    primary_model: str = "llama3.1:8b"
    embedding_model: str = "nomic-embed-text"
    temperature: float = 0.1
    max_tokens: int = 4096
    timeout: int = 120

@dataclass
class Neo4jConfig:
    uri: str = "bolt://localhost:7687"
    username: str = "neo4j"
    password: str = "password123"
    database: str = "neo4j"

@dataclass
class AgentConfig:
    max_iterations: int = 10
    max_research_papers: int = 20
    confidence_threshold: float = 0.75

@dataclass
class SystemConfig:
    ollama: OllamaConfig = field(default_factory=OllamaConfig)
    neo4j: Neo4jConfig = field(default_factory=Neo4jConfig)
    agents: AgentConfig = field(default_factory=AgentConfig)
    data_dir: Path = BASE_DIR / "data"
    log_dir: Path = BASE_DIR / "logs"
    export_dir: Path = BASE_DIR / "exports"

CONFIG = SystemConfig()

for d in [CONFIG.data_dir, CONFIG.log_dir, CONFIG.export_dir]:
    d.mkdir(parents=True, exist_ok=True)
