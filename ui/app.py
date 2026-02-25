# ui/app.py
"""
🧠 Agentic Knowledge Graph System - Web UI
Built with Streamlit for interactive research exploration

Run with: streamlit run ui/app.py
"""
import sys
import os
import json
import time
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
from loguru import logger

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="🧠 Agentic Knowledge Graph",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: #f8f9fa;
        border: 1px solid #e9ecef;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .status-ok { color: #28a745; font-weight: bold; }
    .status-err { color: #dc3545; font-weight: bold; }
    .agent-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        border-radius: 10px;
        padding: 1.5rem;
        margin: 0.5rem 0;
    }
    .insight-box {
        background: #e8f4f8;
        border-left: 4px solid #17a2b8;
        padding: 1rem;
        border-radius: 0 8px 8px 0;
        margin: 0.5rem 0;
    }
    .hypothesis-box {
        background: #e8f8e8;
        border-left: 4px solid #28a745;
        padding: 1rem;
        border-radius: 0 8px 8px 0;
        margin: 0.5rem 0;
    }
    .stTabs [data-baseweb="tab"] { font-size: 1rem; }
</style>
""", unsafe_allow_html=True)


# ─── Initialize Session ────────────────────────────────────────────────────────
@st.cache_resource
def load_orchestrator():
    """Load and cache the orchestrator"""
    from core.orchestrator import AgenticKGOrchestrator
    return AgenticKGOrchestrator()


def get_system():
    """Get or create orchestrator in session"""
    if "orchestrator" not in st.session_state:
        with st.spinner("🚀 Initializing AI System..."):
            st.session_state.orchestrator = load_orchestrator()
    return st.session_state.orchestrator


# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🧠 KG System")
    
    # System Status
    if st.button("🔍 Check System Status"):
        system = get_system()
        status = system.check_system()
        
        if status["ollama"]:
            st.success(f"✅ Ollama: Connected")
            if status["models"]:
                st.info(f"Models: {', '.join(status['models'][:3])}")
        else:
            st.error("❌ Ollama: Not running")
            st.code("# Start Ollama:\nollama serve\n\n# Pull a model:\nollama pull llama3.1:8b")
        
        if status.get("graph"):
            stats = status.get("graph_stats", {})
            st.success(f"✅ Graph: {stats.get('total_nodes', 0)} nodes, {stats.get('total_edges', 0)} edges")
        else:
            st.warning("⚠️ Graph: Fallback mode (no Neo4j)")
    
    st.divider()
    
    # Quick Model Settings
    st.markdown("### ⚙️ Settings")
    model_choice = st.selectbox(
        "LLM Model",
        ["llama3.1:8b", "llama3.2:3b", "mistral:7b", "phi3:mini", "gemma2:9b", "qwen2.5:7b"],
        index=0,
        help="Must be pulled via: ollama pull <model>"
    )
    
    max_papers = st.slider("Max Papers to Fetch", 3, 30, 10)
    
    sources = st.multiselect(
        "Data Sources",
        ["arxiv", "semantic_scholar"],
        default=["arxiv"]
    )
    
    year_from = st.number_input("Papers from year", min_value=1990, max_value=2025, value=2020)
    
    st.divider()
    st.markdown("### 📖 Quick Guide")
    st.markdown("""
1. **Research** - Enter topic to fetch & analyze papers
2. **Chat** - Ask questions about your KG
3. **Explore** - Browse the knowledge graph
4. **Generate** - Create hypotheses & reviews
5. **Export** - Download your KG data
    """)


# ─── Header ──────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🧠 Agentic Knowledge Graph System</h1>
    <p>End-to-End Research Intelligence • Powered by Local LLM (Ollama)</p>
</div>
""", unsafe_allow_html=True)

# ─── Main Tabs ────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🔬 Research Pipeline",
    "💬 Chat & Q&A", 
    "🗺️ Knowledge Graph",
    "💡 Hypotheses",
    "📖 Literature Review",
    "📊 Analytics"
])


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1: RESEARCH PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown("## 🔬 Research Pipeline")
    st.markdown("Enter a research topic and the system will autonomously fetch papers, extract knowledge, and build your knowledge graph.")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        topic = st.text_input(
            "Research Topic",
            placeholder="e.g., 'transformer attention mechanisms', 'CRISPR gene editing', 'quantum error correction'",
            help="Be specific for better results"
        )
    
    with col2:
        local_dir = st.text_input("Local Docs Folder", placeholder="/path/to/pdfs", help="Optional: folder with PDF files")
    
    run_col, clear_col = st.columns([2, 1])
    
    with run_col:
        run_pipeline = st.button("🚀 Run Research Pipeline", type="primary", use_container_width=True)
    
    with clear_col:
        if st.button("🗑️ Clear Results", use_container_width=True):
            if "pipeline_results" in st.session_state:
                del st.session_state["pipeline_results"]
            st.rerun()
    
    if run_pipeline and topic:
        system = get_system()
        system.llm.primary_model = model_choice
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        def update_progress(message, percent):
            progress_bar.progress(percent / 100)
            status_text.info(f"⏳ {message}")
        
        system.on_progress = update_progress
        
        with st.spinner(""):
            results = system.run_research_pipeline(
                topic=topic,
                max_papers=max_papers,
                sources=sources,
                local_dir=local_dir if local_dir else None,
                year_from=year_from
            )
        
        progress_bar.progress(100)
        status_text.success("✅ Pipeline Complete!")
        st.session_state["pipeline_results"] = results
        st.session_state["current_topic"] = topic
    
    # Show Results
    if "pipeline_results" in st.session_state:
        results = st.session_state["pipeline_results"]
        stats = results.get("stats", {})
        
        st.divider()
        st.markdown("### 📊 Pipeline Results")
        
        cols = st.columns(5)
        metrics = [
            ("📄 Papers", stats.get("papers_collected", 0)),
            ("🔷 Entities", stats.get("entities_extracted", 0)),
            ("🔗 Relations", stats.get("relations_found", 0)),
            ("💡 Claims", stats.get("claims_found", 0)),
            ("⏱️ Time (s)", stats.get("elapsed_seconds", 0)),
        ]
        for col, (label, value) in zip(cols, metrics):
            col.metric(label, value)
        
        st.divider()
        
        # Papers Found
        papers = results.get("papers", [])
        if papers:
            st.markdown(f"### 📚 Papers Collected ({len(papers)})")
            for paper in papers[:15]:
                with st.expander(f"📄 {paper.get('title', 'Unknown')[:80]}... ({paper.get('year', '')})"):
                    cols = st.columns([2, 1])
                    with cols[0]:
                        st.write(f"**Authors:** {', '.join(paper.get('authors', [])[:4])}")
                        abstract = paper.get('abstract', 'No abstract available')
                        st.write(f"**Abstract:** {abstract[:500]}...")
                    with cols[1]:
                        st.write(f"**Source:** {paper.get('source', 'unknown')}")
                        st.write(f"**Citations:** {paper.get('citations', 'N/A')}")
                        if paper.get('url'):
                            st.link_button("🔗 Open Paper", paper['url'])
        
        # Research Gaps
        insights = results.get("insights", {})
        gaps = insights.get("research_gaps", [])
        if gaps:
            st.markdown("### 🔍 Research Gaps Identified")
            for gap in gaps[:5]:
                with st.container():
                    st.markdown(f"""
<div class="insight-box">
<b>Gap:</b> {gap.get('description', '')}<br>
<b>Importance:</b> {gap.get('importance', 'medium')}<br>
<b>Suggested Approach:</b> {gap.get('suggested_approach', '')}
</div>
""", unsafe_allow_html=True)
        
        # Landscape
        landscape = insights.get("landscape", {})
        if landscape.get("overview"):
            st.markdown("### 🌍 Research Landscape")
            st.info(landscape.get("overview", ""))
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown("**Key Themes**")
                for theme in landscape.get("key_themes", []):
                    st.markdown(f"• {theme}")
            with col2:
                st.markdown("**Key Challenges**")
                for ch in landscape.get("key_challenges", []):
                    st.markdown(f"• {ch}")
            with col3:
                st.markdown("**Hot Topics**")
                for ht in landscape.get("hot_topics", []):
                    st.markdown(f"• {ht}")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2: CHAT & Q&A
# ═══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("## 💬 Research Assistant Chat")
    st.markdown("Ask questions about your knowledge graph. The AI will reason over collected research to answer.")
    
    # Initialize chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    # Chat display
    chat_container = st.container()
    with chat_container:
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])
                if "sources" in msg and msg["sources"]:
                    with st.expander(f"📚 Sources ({len(msg['sources'])} nodes used)"):
                        for src in msg["sources"][:5]:
                            st.write(f"• **{src.get('name', src.get('title', 'Unknown'))}**: {src.get('description', src.get('abstract', ''))[:150]}")
    
    # Input
    col1, col2 = st.columns([4, 1])
    with col1:
        user_input = st.chat_input("Ask about your research... (e.g., 'What are the key methods in this field?')")
    
    # Quick Questions
    st.markdown("**Quick Questions:**")
    qcols = st.columns(3)
    quick_questions = [
        "What are the main research trends?",
        "What datasets are commonly used?",
        "What are the key challenges?",
        "Compare the main approaches",
        "What are open research problems?",
        "Summarize the key findings"
    ]
    
    selected_q = None
    for i, (qcol, q) in enumerate(zip(qcols * 2, quick_questions)):
        if qcol.button(q, key=f"qq_{i}", use_container_width=True):
            selected_q = q
    
    question = user_input or selected_q
    
    if question:
        system = get_system()
        
        with st.spinner("🤔 Reasoning over knowledge graph..."):
            result = system.ask(question)
        
        st.session_state.chat_history.append({
            "role": "user",
            "content": question
        })
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": result["answer"],
            "sources": result.get("sources", [])
        })
        
        st.rerun()
    
    if st.button("🗑️ Clear Chat"):
        st.session_state.chat_history = []
        if "orchestrator" in st.session_state:
            st.session_state.orchestrator.reasoning_agent.conversation_history = []
        st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3: KNOWLEDGE GRAPH EXPLORER
# ═══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("## 🗺️ Knowledge Graph Explorer")
    
    system = get_system()
    stats = system.graph.get_stats()
    
    # Stats Overview
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Nodes", stats.get("total_nodes", 0))
    col2.metric("Total Edges", stats.get("total_edges", 0))
    col3.metric("Backend", stats.get("backend", "Unknown").split("(")[0])
    col4.metric("Topics Researched", len(system.session_stats.get("topics_researched", [])))
    
    st.divider()
    
    # Search
    search_query = st.text_input("🔍 Search Knowledge Graph", placeholder="Search for concepts, papers, methods...")
    search_label = st.selectbox("Filter by type", ["All", "Paper", "Concept", "Entity", "Method", "Author", "Hypothesis"])
    
    if search_query:
        label_filter = None if search_label == "All" else search_label
        nodes = system.graph.search_nodes(search_query, label=label_filter, limit=30) if hasattr(system.graph, 'search_nodes') else []
        
        if nodes:
            st.markdown(f"**Found {len(nodes)} matching nodes:**")
            
            for node in nodes:
                name = node.get("name", node.get("title", "Unknown"))
                node_type = node.get("label", "Node")
                desc = node.get("description", node.get("abstract", ""))
                
                with st.expander(f"[{node_type}] {name}"):
                    st.json({k: v for k, v in node.items() if k not in ["label", "labels"] and v})
        else:
            st.info("No nodes found. Try running the Research Pipeline first!")
    
    st.divider()
    
    # Graph Visualization
    st.markdown("### 📊 Graph Statistics")
    
    node_types = stats.get("node_types", {})
    if node_types:
        import plotly.express as px
        import pandas as pd
        
        df = pd.DataFrame(list(node_types.items()), columns=["Type", "Count"])
        df = df.sort_values("Count", ascending=False)
        
        col1, col2 = st.columns(2)
        with col1:
            fig = px.bar(df, x="Type", y="Count", title="Node Types Distribution", 
                        color="Count", color_continuous_scale="Blues")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig2 = px.pie(df, values="Count", names="Type", title="Node Distribution",
                         hole=0.4)
            st.plotly_chart(fig2, use_container_width=True)
    
    # Edge types
    edge_types = stats.get("edge_types", {})
    if edge_types:
        st.markdown("#### Relationship Types")
        edge_df = pd.DataFrame(list(edge_types.items()), columns=["Relation", "Count"])
        edge_df = edge_df.sort_values("Count", ascending=False)
        st.dataframe(edge_df, use_container_width=True, hide_index=True)
    
    # Export
    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        if st.button("📥 Export Graph (JSON)"):
            path = system.export_graph(format="json")
            try:
                with open(path, 'r') as f:
                    data = f.read()
                st.download_button("⬇️ Download JSON", data, "knowledge_graph.json", "application/json")
            except:
                st.success(f"Exported to: {path}")
    
    with col2:
        if st.button("📥 Export Graph (GraphML)"):
            path = system.export_graph(format="graphml")
            try:
                with open(path, 'r') as f:
                    data = f.read()
                st.download_button("⬇️ Download GraphML", data, "knowledge_graph.graphml", "text/xml")
            except:
                st.success(f"Exported to: {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4: HYPOTHESIS GENERATION
# ═══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown("## 💡 Research Hypothesis Generator")
    st.markdown("Generate novel, testable research hypotheses from your knowledge graph using AI reasoning.")
    
    hyp_topic = st.text_input(
        "Topic for Hypothesis Generation",
        value=st.session_state.get("current_topic", ""),
        placeholder="Enter a specific topic..."
    )
    
    constraints = st.text_area(
        "Constraints (optional)",
        placeholder="e.g., 'Focus on clinical applications', 'Consider only unsupervised methods', 'Budget < $10k'"
    )
    
    if st.button("🔮 Generate Hypotheses", type="primary"):
        if hyp_topic:
            system = get_system()
            
            with st.spinner("🧠 Generating novel hypotheses..."):
                hypotheses = system.generate_hypotheses(hyp_topic)
            
            if hypotheses:
                st.success(f"Generated {len(hypotheses)} novel research hypotheses!")
                
                for i, hyp in enumerate(hypotheses):
                    st.markdown(f"""
<div class="hypothesis-box">
<h4>Hypothesis {i+1}</h4>
<b>🎯 Statement:</b> {hyp.get('hypothesis', '')}<br><br>
<b>💭 Rationale:</b> {hyp.get('rationale', '')}<br><br>
<b>✨ Novelty:</b> {hyp.get('novelty', '')}<br><br>
<b>🔬 Testing Approach:</b> {hyp.get('testing_approach', '')}<br><br>
<b>📈 Impact:</b> {hyp.get('impact', '')}<br>
<b>🎰 Confidence:</b> {hyp.get('confidence', 0.5)}
</div>
""", unsafe_allow_html=True)
                    st.divider()
            else:
                st.warning("Could not generate hypotheses. Try running the Research Pipeline first!")
        else:
            st.warning("Please enter a topic")
    
    st.divider()
    
    # Compare Approaches
    st.markdown("### ⚖️ Compare Research Approaches")
    comp_col1, comp_col2 = st.columns(2)
    approach1 = comp_col1.text_input("Approach 1", placeholder="e.g., deep learning")
    approach2 = comp_col2.text_input("Approach 2", placeholder="e.g., rule-based systems")
    
    if st.button("Compare Approaches") and approach1 and approach2:
        system = get_system()
        with st.spinner("Comparing..."):
            comparison = system.compare(approach1, approach2)
        
        if comparison:
            st.markdown("#### Comparison Results")
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"**{approach1}** use cases:")
                for uc in comparison.get("use_cases_1", []):
                    st.markdown(f"• {uc}")
            with col2:
                st.markdown(f"**{approach2}** use cases:")
                for uc in comparison.get("use_cases_2", []):
                    st.markdown(f"• {uc}")
            
            st.markdown("**Recommendation:**")
            st.info(comparison.get("recommendation", ""))
            
            if comparison.get("hybrid_potential"):
                st.success(f"**Hybrid Potential:** {comparison['hybrid_potential']}")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 5: LITERATURE REVIEW
# ═══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.markdown("## 📖 Auto-Generated Literature Review")
    st.markdown("Generate a comprehensive literature review from your knowledge graph.")
    
    review_topic = st.text_input(
        "Topic for Literature Review",
        value=st.session_state.get("current_topic", ""),
        placeholder="Enter research topic..."
    )
    
    review_style = st.selectbox(
        "Review Style",
        ["academic", "survey", "executive", "technical"],
        help="academic=formal, survey=comprehensive, executive=non-specialist, technical=expert"
    )
    
    if st.button("📝 Generate Literature Review", type="primary"):
        if review_topic:
            system = get_system()
            
            with st.spinner("📖 Writing literature review... (this may take a minute)"):
                review = system.write_literature_review(review_topic, style=review_style)
            
            if review:
                st.markdown("### Generated Literature Review")
                st.markdown(review)
                
                st.download_button(
                    "⬇️ Download Review (Markdown)",
                    review,
                    f"literature_review_{review_topic[:30].replace(' ','_')}.md",
                    "text/markdown"
                )
        else:
            st.warning("Please enter a topic")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 6: ANALYTICS
# ═══════════════════════════════════════════════════════════════════════════════
with tab6:
    st.markdown("## 📊 System Analytics")
    
    system = get_system()
    session_stats = system.get_session_stats()
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("📄 Papers Processed", session_stats.get("papers_processed", 0))
    col2.metric("🔷 Entities Extracted", session_stats.get("entities_extracted", 0))
    col3.metric("🔗 Relations Found", session_stats.get("relations_found", 0))
    col4.metric("💡 Claims Made", session_stats.get("claims_made", 0))
    
    st.divider()
    
    topics = session_stats.get("topics_researched", [])
    if topics:
        st.markdown("### 📚 Topics Researched This Session")
        for t in topics:
            st.markdown(f"• {t}")
    
    st.divider()
    
    # Full stats
    st.markdown("### 🗄️ Knowledge Graph Statistics")
    stats = system.graph.get_stats()
    
    col1, col2 = st.columns(2)
    with col1:
        st.json(stats.get("node_types", {}))
    with col2:
        st.json(stats.get("edge_types", {}))
    
    # System info
    st.divider()
    st.markdown("### 🖥️ System Configuration")
    from config.settings import CONFIG
    st.code(f"""
LLM Backend: Ollama (Local)
Model: {CONFIG.ollama.primary_model}
Embedding: {CONFIG.ollama.embedding_model}
Graph: {stats.get('backend', 'Unknown')}
Data Dir: {CONFIG.data_dir}
    """)
