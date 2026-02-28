"""
Maritime News Intelligence Pipeline - Analyst Dashboard
A formal, defense-grade interface for maritime intelligence operations.
"""

import streamlit as st
import requests
import json
import os
import time

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
API_BASE = "http://localhost:8000"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

st.set_page_config(
    page_title="Maritime Intelligence Command",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Dark Theme CSS (Defense-Grade, Formal, No Pop Colors)
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    /* Global */
    .stApp {
        background-color: #0d1117;
        color: #c9d1d9;
    }
    
    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #161b22;
        border-right: 1px solid #21262d;
    }
    section[data-testid="stSidebar"] .stMarkdown h1,
    section[data-testid="stSidebar"] .stMarkdown h2,
    section[data-testid="stSidebar"] .stMarkdown h3 {
        color: #e6edf3;
    }
    
    /* Headers */
    h1, h2, h3, h4 {
        color: #e6edf3 !important;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    /* Cards */
    .intel-card {
        background-color: #161b22;
        border: 1px solid #21262d;
        border-radius: 6px;
        padding: 16px 20px;
        margin-bottom: 12px;
    }
    .intel-card h4 {
        margin: 0 0 8px 0;
        font-size: 15px;
        color: #e6edf3 !important;
    }
    .intel-card p {
        margin: 4px 0;
        font-size: 13px;
        color: #8b949e;
        line-height: 1.5;
    }
    
    /* Tags */
    .tag {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 12px;
        font-size: 11px;
        font-weight: 600;
        margin-right: 6px;
        margin-bottom: 4px;
    }
    .tag-critical { background-color: #3d1a1a; color: #da7272; border: 1px solid #5a2d2d; }
    .tag-high     { background-color: #3d2e1a; color: #d4a05a; border: 1px solid #5a4a2d; }
    .tag-medium   { background-color: #1a2e3d; color: #5a9ed4; border: 1px solid #2d4a5a; }
    .tag-low      { background-color: #1a3d2e; color: #5ad49e; border: 1px solid #2d5a4a; }
    
    .tag-geo      { background-color: #1c2333; color: #7a8baa; border: 1px solid #2d3a52; }
    .tag-defense  { background-color: #1c2333; color: #7a8baa; border: 1px solid #2d3a52; }
    .tag-sanction { background-color: #2a1c1c; color: #aa7a7a; border: 1px solid #3d2d2d; }
    .tag-category { background-color: #1c1c2a; color: #8b8baa; border: 1px solid #2d2d3d; }
    
    /* Metric boxes */
    .metric-box {
        background-color: #161b22;
        border: 1px solid #21262d;
        border-radius: 6px;
        padding: 18px;
        text-align: center;
    }
    .metric-box .value {
        font-size: 28px;
        font-weight: 700;
        color: #e6edf3;
    }
    .metric-box .label {
        font-size: 11px;
        color: #8b949e;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-top: 4px;
    }
    
    /* Chat */
    .chat-msg-user {
        background-color: #1c2333;
        border: 1px solid #2d3a52;
        border-radius: 8px;
        padding: 10px 14px;
        margin: 8px 0;
        color: #c9d1d9;
        font-size: 13px;
    }
    .chat-msg-ai {
        background-color: #161b22;
        border: 1px solid #21262d;
        border-radius: 8px;
        padding: 10px 14px;
        margin: 8px 0;
        color: #c9d1d9;
        font-size: 13px;
        line-height: 1.6;
    }
    
    /* Status bar */
    .status-bar {
        background-color: #161b22;
        border: 1px solid #21262d;
        border-radius: 4px;
        padding: 8px 14px;
        font-size: 12px;
        color: #8b949e;
        margin-bottom: 16px;
    }
    .status-online  { border-left: 3px solid #3fb950; }
    .status-offline { border-left: 3px solid #da3633; }
    
    /* Buttons */
    .stButton > button {
        background-color: #21262d;
        color: #c9d1d9;
        border: 1px solid #30363d;
        border-radius: 6px;
        font-weight: 600;
    }
    .stButton > button:hover {
        background-color: #30363d;
        border-color: #8b949e;
    }
    
    /* Tables */
    .stDataFrame {
        background-color: #161b22;
    }
    
    /* Divider */
    hr {
        border-color: #21262d;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------
def api_available():
    try:
        r = requests.get(f"{API_BASE}/db/articles", timeout=3)
        return r.status_code == 200
    except Exception:
        return False

def render_risk_tag(risk_level):
    if not risk_level:
        return ""
    level = risk_level.lower()
    return f'<span class="tag tag-{level}">{risk_level.upper()}</span>'

def render_flag_tags(article):
    tags = ""
    if article.get("is_geopolitical"):
        tags += '<span class="tag tag-geo">GEOPOLITICAL</span>'
    if article.get("has_defense_implications"):
        tags += '<span class="tag tag-defense">DEFENSE</span>'
    if article.get("is_sanction_sensitive"):
        tags += '<span class="tag tag-sanction">SANCTIONS</span>'
    return tags


# ---------------------------------------------------------------------------
# Sidebar Navigation
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### Maritime Intelligence")
    st.markdown("##### Command Dashboard")
    st.markdown("---")
    
    page = st.radio(
        "Navigation",
        [
            "Ingestion Control",
            "Intelligence Feed",
            "Macro Analytics",
            "RAG Query Interface"
        ],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    
    # System status
    online = api_available()
    if online:
        st.markdown('<div class="status-bar status-online">API Status: ONLINE</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="status-bar status-offline">API Status: OFFLINE</div>', unsafe_allow_html=True)
    
    st.markdown(f"<p style='font-size:11px; color:#8b949e;'>Session: {time.strftime('%Y-%m-%d %H:%M')}</p>", unsafe_allow_html=True)


# ===========================================================================
# PAGE 1: INGESTION CONTROL
# ===========================================================================
if page == "Ingestion Control":
    st.markdown("## Ingestion Control Center")
    st.markdown("Initiate data acquisition and intelligence extraction operations.")
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Data Acquisition")
        st.markdown("Fetch the latest maritime news articles from the MarineTraffic GraphQL endpoint.")
        if st.button("Execute Scraper", width="stretch"):
            with st.spinner("Querying MarineTraffic GraphQL API..."):
                try:
                    resp = requests.post(f"{API_BASE}/pipeline/scrape", timeout=30)
                    data = resp.json()
                    st.success(f"Acquisition complete. {data.get('articles_fetched', 0)} articles retrieved.")
                except Exception as e:
                    st.error(f"Scraper failed: {e}")
    
    with col2:
        st.markdown("#### NLP + LLM Extraction")
        st.markdown("Run the full pipeline: preprocessing, entity recognition, classification, and LLM-based event extraction.")
        batch_size = st.number_input("Batch Size", min_value=1, max_value=100, value=20)
        if st.button("Execute Extraction Pipeline", width="stretch"):
            with st.spinner("Processing articles through NLP and LLM layers..."):
                try:
                    resp = requests.post(f"{API_BASE}/pipeline/run", json={"batch_size": batch_size}, timeout=300)
                    data = resp.json()
                    st.success(f"Pipeline complete. {data.get('articles_processed', 0)} articles processed.")
                    if data.get("details"):
                        for d in data["details"]:
                            st.markdown(f"""
                            <div class="intel-card">
                                <h4>{d['title']}</h4>
                                <p>Classification: <span class="tag tag-category">{d['classification']}</span> 
                                {render_risk_tag(d.get('risk_level'))} 
                                Events Extracted: {d['events_extracted']}</p>
                            </div>
                            """, unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Pipeline execution failed: {e}")

    st.markdown("---")
    st.markdown("#### Vector Database Ingestion")
    st.markdown("Embed scraped articles and push vector representations into the Qdrant cluster for RAG operations.")
    if st.button("Ingest into Qdrant", width="stretch"):
        with st.spinner("Embedding articles and upserting vectors..."):
            try:
                from RAG.ingester import ingest_articles
                data_path = os.path.join(BASE_DIR, "latest_articles.json")
                ingest_articles(data_path)
                st.success("Vector ingestion complete.")
            except Exception as e:
                st.error(f"Ingestion failed: {e}")


# ===========================================================================
# PAGE 2: INTELLIGENCE FEED
# ===========================================================================
elif page == "Intelligence Feed":
    st.markdown("## Intelligence Feed")
    st.markdown("Classified and enriched maritime event reports.")
    st.markdown("---")
    
    # Load data - try API first, then fall back to local JSON
    articles_data = []
    events_data = []
    
    try:
        resp_articles = requests.get(f"{API_BASE}/db/articles", timeout=5)
        if resp_articles.status_code == 200:
            articles_data = resp_articles.json()
        
        resp_events = requests.get(f"{API_BASE}/db/events", timeout=5)
        if resp_events.status_code == 200:
            events_data = resp_events.json()
    except Exception:
        pass
    
    # If API returned nothing, fall back to processed_test_results.json
    if not articles_data:
        local_path = os.path.join(BASE_DIR, "processed_test_results.json")
        if os.path.exists(local_path):
            with open(local_path, "r") as f:
                raw_data = json.load(f)
            # Transform the JSON into the same format
            for item in raw_data:
                classification = item.get("classification", item.get("zero_shot_classification", {}))
                enrichment = item.get("llm_structured_output", {}).get("enrichment", {})
                articles_data.append({
                    "id": item.get("hash", item.get("id")),
                    "title": item.get("title", "Unknown"),
                    "classification": classification.get("label", "Unknown") if isinstance(classification, dict) else classification,
                    "confidence": classification.get("score", 0) if isinstance(classification, dict) else 0,
                    "risk_level": enrichment.get("risk_level"),
                    "impact_scope": enrichment.get("impact_scope"),
                    "executive_summary": enrichment.get("executive_summary", item.get("structured_events", {}).get("events", [{}])[0].get("summary", "") if item.get("structured_events", {}).get("events") else "No summary available."),
                    "strategic_tags": enrichment.get("strategic_relevance_tags", []),
                    "is_geopolitical": enrichment.get("is_geopolitical", False),
                    "has_defense_implications": enrichment.get("has_defense_implications", False),
                    "is_sanction_sensitive": enrichment.get("is_sanction_sensitive", False),
                    "named_entities": {"entities": item.get("gliner_entities", item.get("named_entities", {}).get("entities", []))},
                    "url": item.get("source_url", item.get("url", ""))
                })
                for ev in item.get("llm_structured_output", item.get("structured_events", {})).get("events", []):
                    loc = ev.get("location", {})
                    events_data.append({
                        "event_id": ev.get("event_id"),
                        "article_title": item.get("title"),
                        "article_hash": item.get("hash", item.get("id")),
                        "event_date": ev.get("event_date"),
                        "port": loc.get("port"),
                        "country": loc.get("country"),
                        "vessels_involved": ev.get("vessels_involved", []),
                        "organizations_involved": ev.get("organizations_involved", []),
                        "incident_type": ev.get("incident_type"),
                        "casualties": ev.get("casualties"),
                        "cargo_type": ev.get("cargo_type"),
                        "summary": ev.get("summary"),
                        "confidence_score": ev.get("confidence_score"),
                    })
    
    if not articles_data:
        st.warning("No processed data available. Run the extraction pipeline first.")
    else:
        # Metrics row
        total = len(articles_data)
        critical_count = sum(1 for a in articles_data if a.get("risk_level") in ["Critical", "CRITICAL"])
        high_count = sum(1 for a in articles_data if a.get("risk_level") in ["High", "HIGH"])
        defense_count = sum(1 for a in articles_data if a.get("has_defense_implications"))
        sanction_count = sum(1 for a in articles_data if a.get("is_sanction_sensitive"))
        
        m1, m2, m3, m4, m5 = st.columns(5)
        for col, val, label in [
            (m1, total, "TOTAL REPORTS"),
            (m2, critical_count, "CRITICAL"),
            (m3, high_count, "HIGH RISK"),
            (m4, defense_count, "DEFENSE FLAGGED"),
            (m5, sanction_count, "SANCTION FLAGGED")
        ]:
            col.markdown(f'<div class="metric-box"><div class="value">{val}</div><div class="label">{label}</div></div>', unsafe_allow_html=True)
        
        st.markdown("---")
        
        # --- EXPANDER VIEW LOGIC ---
        all_classes = sorted(set(a.get("classification", "Unknown") for a in articles_data))
        selected_class = st.selectbox("Filter by Classification", ["All"] + all_classes)
        filtered = articles_data if selected_class == "All" else [a for a in articles_data if a.get("classification") == selected_class]
        
        for article in filtered:
            title = article.get('title', 'Untitled Report')
            risk = article.get('risk_level', 'Unknown')
            impact = article.get('impact_scope', 'Unknown')
            
            # The expander acts as the clickable bar
            with st.expander(f"📑 {title}  |  Risk: {risk}  |  Impact: {impact}"):
                
                risk_tag = render_risk_tag(article.get("risk_level"))
                flag_tags = render_flag_tags(article)
                strategic = ""
                if article.get("strategic_tags"):
                    strategic = " ".join([f'<span class="tag tag-category">{t}</span>' for t in article["strategic_tags"]])
                
                classification = article.get("classification", "Unknown")
                
                st.markdown(f"""
                <div style="margin-bottom: 15px; margin-top: 5px;">
                    <b>Risk:</b> {risk_tag} &nbsp;|&nbsp; 
                    <b>Impact:</b> {impact} &nbsp;|&nbsp; 
                    <b>Classification:</b> <span class='tag tag-category'>{classification}</span><br>
                    <div style="margin-top: 10px;">{flag_tags} {strategic}</div>
                </div>
                """, unsafe_allow_html=True)

                if article.get("url"):
                    st.markdown(f"<a href='{article['url']}' target='_blank'><button style='background-color:#1f6feb;color:white;border:none;padding:5px 12px;border-radius:5px;cursor:pointer;margin-bottom:15px;'>🔗 Source URL</button></a>", unsafe_allow_html=True)

                st.markdown("#### Executive Summary")
                st.write(article.get("executive_summary", "No summary available."))
                
                # NER Entities Section
                st.markdown("---")
                st.markdown("#### Named Entity Recognition (NER)")
                named_entities_data = article.get("named_entities") or {}
                named_entities = named_entities_data.get("entities", [])
                
                if named_entities:
                    vessels, ports, orgs, countries, persons, incidents, dates = set(), set(), set(), set(), set(), set(), set()
                    
                    for ent in named_entities:
                        label = ent.get("label", "").upper()
                        text = ent.get("text", "")
                        if label in ["VESSEL", "VESSEL NAME", "SHIP"]: vessels.add(text)
                        elif label == "PORT": ports.add(text)
                        elif label == "ORGANIZATION": orgs.add(text)
                        elif label in ["LOCATION", "COUNTRY"]: countries.add(text)
                        elif label == "PERSON": persons.add(text)
                        elif label == "INCIDENT TYPE": incidents.add(text)
                        elif label == "DATE": dates.add(text)
                    
                    c1, c2, c3 = st.columns(3)
                    with c1:
                        st.markdown("**Vessel Names**")
                        st.write(", ".join(vessels) if vessels else "None detected")
                        st.markdown("**Organizations**")
                        st.write(", ".join(orgs) if orgs else "None detected")
                    with c2:
                        st.markdown("**Ports**")
                        st.write(", ".join(ports) if ports else "None detected")
                        st.markdown("**Countries**")
                        st.write(", ".join(countries) if countries else "None detected")
                    with c3:
                        st.markdown("**Incident Types**")
                        st.write(", ".join(incidents) if incidents else "None detected")
                        st.markdown("**Dates**")
                        st.write(", ".join(dates) if dates else "None detected")
                else:
                    st.info("No named entities were extracted for this report.")

                # Events Structured Data
                specific_events = [ev for ev in events_data if (ev.get("article_title") and ev.get("article_title") == article.get("title")) or (ev.get("article_hash") and ev.get("article_hash") == article.get("id"))]
                if specific_events:
                    st.markdown("---")
                    st.markdown("#### Extracted Event Details")
                    for ev in specific_events:
                        st.markdown(f'<div class="intel-card" style="border-left: 4px solid #1f6feb;">', unsafe_allow_html=True)
                        st.markdown(f"**Event ID:** `{ev.get('event_id', 'Unknown')}`")
                        ev_cols1, ev_cols2 = st.columns(2)
                        with ev_cols1:
                            st.markdown(f"**Date:** {ev.get('event_date', 'N/A')}")
                            st.markdown(f"**Incident Type:** {ev.get('incident_type', 'N/A')}")
                            port = ev.get('port') or ev.get('location', {}).get('port') or 'Unknown Port'
                            country = ev.get('country') or ev.get('location', {}).get('country') or 'Unknown Country'
                            st.markdown(f"**Location:** {port}, {country}")
                        with ev_cols2:
                            vessels_list = ev.get('vessels_involved', [])
                            st.markdown(f"**Vessels:** {', '.join(vessels_list) if vessels_list else 'N/A'}")
                            orgs_list = ev.get('organizations_involved', [])
                            st.markdown(f"**Organizations:** {', '.join(orgs_list) if orgs_list else 'N/A'}")
                            st.markdown(f"**Casualties:** {ev.get('casualties', 'N/A')} | **Cargo Type:** {ev.get('cargo_type', 'N/A')}")
                        
                        st.markdown(f"**Summary:** {ev.get('summary', '')}")
                        st.markdown(f"<small>Confidence Score: {ev.get('confidence_score', 0)}</small>", unsafe_allow_html=True)
                        st.markdown("</div>", unsafe_allow_html=True)


# ===========================================================================
# PAGE 3: MACRO ANALYTICS
# ===========================================================================
elif page == "Macro Analytics":
    st.markdown("## Macro Analytics")
    st.markdown("Strategic trend analysis and entity relationship mapping.")
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Topic Modeling and Trend Analysis")
        st.markdown("Run the BERTopic unsupervised learning pipeline to identify emerging macro-themes across the article corpus.")
        if st.button("Generate Topic Model", width="stretch"):
            with st.spinner("Training BERTopic model on article corpus..."):
                try:
                    resp = requests.post(f"{API_BASE}/analytics/topic-model", timeout=120)
                    if resp.status_code == 200:
                        st.success("Topic model generated successfully.")
                    else:
                        st.error(f"Failed: {resp.text}")
                except Exception as e:
                    st.error(f"Topic modeling failed: {e}")
        
        # Show existing dashboard if available
        dashboard_path = os.path.join(BASE_DIR, "theme_dashboard.html")
        if os.path.exists(dashboard_path):
            with open(dashboard_path, "r") as f:
                html_content = f.read()
            st.components.v1.html(html_content, height=500, scrolling=True)
        else:
            st.info("No topic model output available. Generate one using the button above.")
    
    with col2:
        st.markdown("#### Knowledge Graph")
        st.markdown("Build the entity relationship graph connecting vessels, organizations, ports, and incidents.")
        if st.button("Generate Knowledge Graph", width="stretch"):
            with st.spinner("Constructing entity relationship graph..."):
                try:
                    resp = requests.post(f"{API_BASE}/analytics/knowledge-graph", timeout=120)
                    if resp.status_code == 200:
                        st.success("Knowledge graph generated successfully.")
                    else:
                        st.error(f"Failed: {resp.text}")
                except Exception as e:
                    st.error(f"Graph generation failed: {e}")
        
        # Show existing graph if available
        graph_path = os.path.join(BASE_DIR, "knowledge_graph.html")
        if os.path.exists(graph_path):
            with open(graph_path, "r", encoding="utf-8") as f:
                html_data = f.read()
            st.components.v1.html(html_data, height=750, scrolling=True)
        else:
            st.info("No knowledge graph available. Generate one using the button above.")


# ===========================================================================
# PAGE 4: RAG QUERY INTERFACE
# ===========================================================================
elif page == "RAG Query Interface":
    st.markdown("## RAG Query Interface")
    st.markdown("Interrogate the intelligence database using natural language. Responses are grounded strictly in ingested article data.")
    st.markdown("---")
    
    # Chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    # Display chat history
    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            st.markdown(f'<div class="chat-msg-user"><strong>ANALYST:</strong> {msg["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="chat-msg-ai"><strong>INTELLIGENCE SYSTEM:</strong><br>{msg["content"]}</div>', unsafe_allow_html=True)
            if msg.get("sources"):
                for src in msg["sources"]:
                    st.markdown(f'<p style="font-size:11px; color:#8b949e; margin-left:14px;">Source: {src["title"]} (Relevance: {src["score"]})</p>', unsafe_allow_html=True)
    
    # Input
    question = st.text_input("Enter your query", placeholder="e.g., Which vessels were detained in the past month?", label_visibility="collapsed")
    
    if st.button("Submit Query", width="stretch") and question:
        st.session_state.chat_history.append({"role": "user", "content": question})
        
        with st.spinner("Retrieving context and generating response..."):
            try:
                resp = requests.post(f"{API_BASE}/rag/chat", json={"question": question}, timeout=60)
                data = resp.json()
                answer = data.get("answer", "No response generated.")
                sources = data.get("sources", [])
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources
                })
                st.rerun()
            except Exception as e:
                st.error(f"Query failed: {e}")
    
    if st.button("Clear History"):
        st.session_state.chat_history = []
        st.rerun()
