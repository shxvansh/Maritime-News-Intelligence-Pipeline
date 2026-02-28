import json
import networkx as nx
import matplotlib.pyplot as plt
import os

def build_knowledge_graph(json_filepath):
    print("Building Knowledge Graph...")
    # Create an empty graph canvas
    G = nx.Graph()
    
    if not os.path.exists(json_filepath):
        print(f"Error: Could not find {json_filepath}.")
        print("Make sure you run the pipeline first to generate the structured data.")
        return
        
    with open(json_filepath, 'r', encoding='utf-8') as f:
        articles = json.load(f)
        
    print(f"Loaded {len(articles)} processed articles.\n")
    
    for article in articles:
        # Safely extract the structured events
        events = article.get("structured_events", {}).get("events", [])
        
        for event in events:
            # Extract components for Nodes
            incident = event.get("incident_type")
            vessels = event.get("vessels_involved", [])
            orgs = event.get("organizations_involved", [])
            
            location = event.get("location", {})
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
    print("✅ Graph created successfully!")
    print(f"Total Unique Nodes (Entities): {G.number_of_nodes()}")
    print(f"Total Edges (Relationships): {G.number_of_edges()}")
    
    print("\n--- 10 Connected Graph Examples ---")
    edge_list = list(G.edges(data=True))
    for i, (node_a, node_b, edge_data) in enumerate(edge_list[:10]):
        # Extract the node types so we can print clearly
        type_a = G.nodes[node_a].get('type', 'Unknown')
        type_b = G.nodes[node_b].get('type', 'Unknown')
        relation = edge_data.get("relation", "LINKED_TO")
        
        print(f"{i+1}. ({node_a} [{type_a}]) ➔ {relation} ➔ ({node_b} [{type_b}])")
        
    # --- VISUALIZATION MATPLOTLIB ---
    plt.figure(figsize=(14, 10))
    
    # Assign specific colors to different node types to make it look professional
    color_map = []
    for node, data in G.nodes(data=True):
        node_type = data.get("type", "Unknown")
        if node_type == "Vessel":
            color_map.append("lightblue")
        elif node_type == "Incident":
            color_map.append("salmon")
        elif node_type == "Organization":
            color_map.append("lightgreen")
        elif node_type in ["Port", "Country"]:
            color_map.append("orange")
        else:
            color_map.append("lightgray")
            
    # Spring layout naturally clusters connected nodes together
    pos = nx.spring_layout(G, k=0.5, seed=42)
    
    # Draw the Nodes and Edges
    nx.draw(
        G, pos, 
        with_labels=True, 
        node_color=color_map, 
        node_size=1800, 
        font_size=8, 
        font_weight="bold", 
        edge_color="gray",
        alpha=0.9
    )
    
    # Draw the Text Labels (Verbs) on the Lines itself
    edge_labels = nx.get_edge_attributes(G, 'relation')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=6, font_color="blue")
    
    plt.title("Maritime Intelligence Knowledge Graph", fontsize=16, fontweight='bold')
    plt.axis("off") 
    
    output_img = os.path.join(base_dir, "knowledge_graph.png")
    plt.savefig(output_img, format="png", dpi=300, bbox_inches='tight')
    print(f"\n🚀 Graph visualization saved successfully to: {output_img}")

if __name__ == "__main__":
    # We will point it dynamically to the processed_test_results.json 
    # located at the root of your project directory
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    json_path = os.path.join(base_dir, "processed_test_results.json")
    
    build_knowledge_graph(json_path)
