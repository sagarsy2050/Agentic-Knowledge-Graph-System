import json
from typing import List, Dict, Optional
from loguru import logger
from datetime import datetime
import networkx as nx
from config.settings import CONFIG

try:
    from neo4j import GraphDatabase
    NEO4J_AVAILABLE = True
except ImportError:
    NEO4J_AVAILABLE = False


class InMemoryGraph:

    def __init__(self):
        self.graph = nx.MultiDiGraph()
        self.node_data = {}
        logger.info("Using in-memory NetworkX graph")

    def add_node(self, node_id, label, properties):
        self.graph.add_node(node_id, label=label, **properties)
        self.node_data[node_id] = {"label": label, **properties}
        return node_id

    def add_edge(self, source, target, relation, properties=None):
        if properties is None:
            properties = {}
        self.graph.add_edge(source, target, relation=relation, **properties)
        return True

    def search_nodes(self, query, label=None, limit=20):
        results = []
        q = query.lower()
        for node_id, data in self.node_data.items():
            if label and data.get("label") != label:
                continue
            if not q or q in json.dumps(data).lower():
                results.append({"id": node_id, **data})
        return results[:limit]

    def get_neighbors(self, node_id, depth=1):
        if node_id not in self.graph:
            return {"nodes": [], "edges": []}
        nodes_set = set([node_id])
        for _ in range(depth):
            new_nodes = set()
            for n in list(nodes_set):
                new_nodes.update(self.graph.successors(n))
                new_nodes.update(self.graph.predecessors(n))
            nodes_set.update(new_nodes)
        sub = self.graph.subgraph(nodes_set)
        return {
            "nodes": [{"id": n, **self.node_data.get(n, {})} for n in sub.nodes()],
            "edges": [{"source": u, "target": v, **d} for u, v, d in sub.edges(data=True)]
        }

    def get_stats(self):
        node_types = {}
        for nid, data in self.node_data.items():
            lbl = data.get("label", "Unknown")
            node_types[lbl] = node_types.get(lbl, 0) + 1
        return {
            "total_nodes": self.graph.number_of_nodes(),
            "total_edges": self.graph.number_of_edges(),
            "node_types": node_types,
            "edge_types": {},
            "backend": "NetworkX (in-memory)"
        }

    def export_graphml(self, path):
        nx.write_graphml(self.graph, path)

    def run_cypher(self, query, params=None):
        return []

    def close(self):
        pass


class Neo4jGraph:

    def __init__(self):
        cfg = CONFIG.neo4j
        self.driver = GraphDatabase.driver(cfg.uri, auth=(cfg.username, cfg.password))
        self.database = cfg.database
        # Actually verify connection works before claiming success
        self.driver.verify_connectivity()
        self._setup_schema()
        logger.success("Neo4j Knowledge Graph connected")

    def _setup_schema(self):
        constraints = [
            "CREATE CONSTRAINT paper_id IF NOT EXISTS FOR (p:Paper) REQUIRE p.id IS UNIQUE",
            "CREATE CONSTRAINT author_id IF NOT EXISTS FOR (a:Author) REQUIRE a.id IS UNIQUE",
            "CREATE CONSTRAINT concept_id IF NOT EXISTS FOR (c:Concept) REQUIRE c.id IS UNIQUE",
            "CREATE CONSTRAINT entity_id IF NOT EXISTS FOR (e:Entity) REQUIRE e.id IS UNIQUE",
        ]
        with self.driver.session(database=self.database) as session:
            for c in constraints:
                try:
                    session.run(c)
                except:
                    pass

    def add_node(self, node_id, label, properties):
        properties["id"] = node_id
        properties["updated_at"] = datetime.now().isoformat()
        prop_str = ", ".join([f"n.{k} = ${k}" for k in properties.keys()])
        query = f"MERGE (n:{label} {{id: $id}}) SET {prop_str} RETURN n.id"
        with self.driver.session(database=self.database) as session:
            result = session.run(query, **properties)
            record = result.single()
            return record[0] if record else node_id

    def add_edge(self, source_id, target_id, relation, properties=None):
        if properties is None:
            properties = {}
        properties["created_at"] = datetime.now().isoformat()
        props_str = "{" + ", ".join([f"{k}: ${k}" for k in properties.keys()]) + "}"
        query = f"""
        MATCH (a {{id: $source_id}})
        MATCH (b {{id: $target_id}})
        MERGE (a)-[r:{relation} {props_str}]->(b)
        RETURN type(r)
        """
        with self.driver.session(database=self.database) as session:
            try:
                result = session.run(query, source_id=source_id, target_id=target_id, **properties)
                return result.single() is not None
            except Exception as e:
                logger.warning(f"Edge add warning: {e}")
                return False

    def search_nodes(self, query, label=None, limit=20):
        if label:
            cypher = f"""
            MATCH (n:{label})
            WHERE toLower(coalesce(n.name,'')) CONTAINS toLower($q)
               OR toLower(coalesce(n.title,'')) CONTAINS toLower($q)
               OR toLower(coalesce(n.description,'')) CONTAINS toLower($q)
            RETURN n, labels(n) as labels LIMIT $limit
            """
        else:
            cypher = """
            MATCH (n)
            WHERE toLower(coalesce(n.name,'')) CONTAINS toLower($q)
               OR toLower(coalesce(n.title,'')) CONTAINS toLower($q)
               OR toLower(coalesce(n.abstract,'')) CONTAINS toLower($q)
            RETURN n, labels(n) as labels LIMIT $limit
            """
        with self.driver.session(database=self.database) as session:
            result = session.run(cypher, q=query, limit=limit)
            nodes = []
            for record in result:
                nd = dict(record["n"])
                nd["labels"] = record["labels"]
                nd["label"] = record["labels"][0] if record["labels"] else "Node"
                nodes.append(nd)
            return nodes

    def get_neighbors(self, node_id, depth=2):
        cypher = f"""
        MATCH path = (start {{id: $node_id}})-[*1..{depth}]-(neighbor)
        RETURN nodes(path) as nodes, relationships(path) as rels
        """
        with self.driver.session(database=self.database) as session:
            try:
                result = session.run(cypher, node_id=node_id)
                all_nodes = {}
                all_edges = []
                for record in result:
                    for node in record["nodes"]:
                        nid = node.get("id", str(node.id))
                        all_nodes[nid] = dict(node)
                    for rel in record["rels"]:
                        all_edges.append({
                            "source": rel.start_node.get("id"),
                            "target": rel.end_node.get("id"),
                            "relation": rel.type,
                        })
                return {"nodes": list(all_nodes.values()), "edges": all_edges}
            except Exception as e:
                logger.error(f"Neighbor query failed: {e}")
                return {"nodes": [], "edges": []}

    def run_cypher(self, query, params=None):
        if params is None:
            params = {}
        with self.driver.session(database=self.database) as session:
            result = session.run(query, **params)
            return [dict(r) for r in result]

    def get_stats(self):
        try:
            with self.driver.session(database=self.database) as session:
                node_result = session.run(
                    "MATCH (n) RETURN labels(n)[0] as label, count(n) as count ORDER BY count DESC"
                )
                edge_result = session.run(
                    "MATCH ()-[r]->() RETURN type(r) as rel, count(r) as count ORDER BY count DESC"
                )
                node_types = {r["label"]: r["count"] for r in node_result}
                edge_types = {r["rel"]: r["count"] for r in edge_result}
            return {
                "total_nodes": sum(node_types.values()),
                "total_edges": sum(edge_types.values()),
                "node_types": node_types,
                "edge_types": edge_types,
                "backend": "Neo4j"
            }
        except Exception as e:
            logger.error(f"Stats error: {e}")
            return {"total_nodes": 0, "total_edges": 0, "node_types": {}, "edge_types": {}, "backend": "Neo4j"}

    def export_graphml(self, path):
        logger.info(f"Neo4j export to {path} not implemented — use Neo4j Browser export")

    def close(self):
        self.driver.close()


def create_knowledge_graph():
    if not NEO4J_AVAILABLE:
        logger.warning("neo4j package not installed, using in-memory graph")
        return InMemoryGraph()
    try:
        return Neo4jGraph()
    except Exception as e:
        logger.warning(f"Neo4j failed ({e}), using in-memory graph")
        return InMemoryGraph()