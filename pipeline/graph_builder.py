import json
import networkx as nx
import os
from pyvis.network import Network

def build_knowledge_graph(articles):
    """
    Constructs a knowledge graph from a list of articles (either dicts or ORM objects).
    """
    print("building knowledge graph...")
    # Create an empty graph canvas
    G = nx.Graph()
    
    if not articles:
        print("Warning: No articles provided for graph construction.")
        return
        
    print(f"Processing {len(articles)} articles for graph extraction.\n")
    
    for article in articles:
        # Support both dictionary (JSON fallback) and SQLAlchemy model objects
        if hasattr(article, '__table__'):
            # It's an ORM object (database.Article)
            events = getattr(article, 'events', [])
        else:
            # It's a dict (JSON fallback)
            events = article.get("structured_events", {}).get("events", [])
            # Also check for 'llm_structured_output' which is used in some places
            if not events:
                events = article.get("llm_structured_output", {}).get("events", [])
        
        for event in events:
            # Support both dict and ORM object for the event itself
            is_orm_event = hasattr(event, '__table__')
            
            # Extract components for Nodes
            incident = getattr(event, "incident_type", None) if is_orm_event else event.get("incident_type")
            vessels = getattr(event, "vessels_involved", []) if is_orm_event else event.get("vessels_involved", [])
            orgs = getattr(event, "organizations_involved", []) if is_orm_event else event.get("organizations_involved", [])
            
            location = getattr(event, "location", {}) if is_orm_event else event.get("location", {})
            # Special case for ORM which has flat attributes for port/country
            if is_orm_event:
                port = getattr(event, "port", None)
                country = getattr(event, "country", None)
            else:
                port = location.get("port") if isinstance(location, dict) else None
                country = location.get("country") if isinstance(location, dict) else None
            
            # --- NODE CREATION & EDGE LINKING ---
            
            # Add the Incident Node (The center of the event)
            if incident and incident != "null":
                G.add_node(incident, type="Incident")
                
            # Add Port & Country Nodes
            if port:
                G.add_node(port, type="Port")
            if country:
                G.add_node(country, type="Country")
                if port:
                    # Edge: Port -> Country
                    G.add_edge(port, country, relation="LOCATED_IN")

            # Add Vessel Nodes
            for vessel in vessels:
                G.add_node(vessel, type="Vessel")
                
                # Link Vessel to the Incident
                if incident and incident != "null":
                    G.add_edge(vessel, incident, relation="INVOLVED_IN")
                    
                # Link Vessel to its Location
                if port:
                    G.add_edge(vessel, port, relation="LOCATED_AT")
                elif country:
                    G.add_edge(vessel, country, relation="LOCATED_IN")
                    
            # Add Organization Nodes
            for org in orgs:
                G.add_node(org, type="Organization")
                
                # If there are vessels, connect the org to the vessels (Owner/Manager assumption)
                if vessels:
                    for vessel in vessels:
                        G.add_edge(org, vessel, relation="ASSOCIATED_WITH")
                else:
                    # Otherwise, connect the org directly to the incident or location
                    if incident and incident != "null":
                        G.add_edge(org, incident, relation="INVOLVED_IN")

    # --- TERMINAL OUTPUT ---
    print(" graph created succesfully!")
    print(f"Total Unique Nodes (Entities): {G.number_of_nodes()}")
    print(f"Total Edges (Relationships): {G.number_of_edges()}")
    
    print("\n--- 10 connected graph examples ---")
    edge_list = list(G.edges(data=True))
    for i, (node_a, node_b, edge_data) in enumerate(edge_list[:10]):
        # Extract the node types so we can print clearly
        type_a = G.nodes[node_a].get('type', 'Unknown')
        type_b = G.nodes[node_b].get('type', 'Unknown')
        relation = edge_data.get("relation", "LINKED_TO")
        
        print(f"{i+1}. ({node_a} [{type_a}]) ➔ {relation} ➔ ({node_b} [{type_b}])")
        
    # --- VISUALIZATION PYVIS ---
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_html = os.path.join(base_dir, "knowledge_graph.html")
    
    # Map colors to nodes in the NetworkX graph itself so PyVis can read them
    for node, data in G.nodes(data=True):
        node_type = data.get("type", "Unknown")
        if node_type == "Vessel":
            data["color"] = "#add8e6"  # lightblue
        elif node_type == "Incident":
            data["color"] = "#fa8072"  # salmon
        elif node_type == "Organization":
            data["color"] = "#90ee90"  # lightgreen
        elif node_type in ["Port", "Country"]:
            data["color"] = "#ffa500"  # orange
        else:
            data["color"] = "#d3d3d3"  # lightgray
            
        data["title"] = f"{node}\nType: {node_type}" # Hover tooltip
            
    # Map edge labels so they show up in PyVis
    for u, v, data in G.edges(data=True):
        data["title"] = data.get("relation", "LINKED_TO")
        data["label"] = data.get("relation", "LINKED_TO")

    # Initialize PyVis network (matching Streamlit dark theme)
    # Using directed=False because the relations aren't strictly directional, but we could use True for arrows!
    net = Network(height="750px", width="100%", bgcolor="#161b22", font_color="#c9d1d9", directed=True)
    
    # Load the NetworkX graph
    net.from_nx(G)
    
    # Tweak physics layout to prevent excessive overlap
    net.repulsion(node_distance=150, central_gravity=0.1, spring_length=150, spring_strength=0.05, damping=0.95)
    
    # Save Graph
    net.save_graph(output_html)
    print(f"\n Interactive 3D graph saved successfully to: {output_html}")

if __name__ == "__main__":
    # Point to the database to fetch articles for the graph
    from pipeline.database import SessionLocal, Article
    db = SessionLocal()
    articles = db.query(Article).all()
    build_knowledge_graph(articles)
    db.close()
