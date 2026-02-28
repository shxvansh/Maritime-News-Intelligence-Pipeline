import json
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import logging
import transformers

# Suppress HuggingFace "UNEXPECTED" weigh loading warnings
transformers.logging.set_verbosity_error()
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from bertopic import BERTopic
from hdbscan import HDBSCAN
from umap import UMAP
from sklearn.feature_extraction.text import CountVectorizer

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.preprocessor import Preprocessor

def run_topic_modeling(json_filepath):
    print("🚀 Initializing Topic Modeling & Theme Extraction...\n")
    
    if not os.path.exists(json_filepath):
        print(f"❌ Could not find {json_filepath}.")
        return

    with open(json_filepath, 'r', encoding='utf-8') as f:
        articles = json.load(f)

    if len(articles) < 10:
        print("⚠️ Not enough articles to perform meaningful clustering. Need at least 10.")
        return

    print(f"📂 Loaded {len(articles)} articles. Processing text...")

    # Extract text and use Preprocessor
    docs = []
    timestamps = []
    titles = []

    for idx, article in enumerate(articles):
        raw_text = article.get("content", "")
        if not raw_text:
            continue
            
        # Clean the text using the existing Preprocessor logic
        clean_data = Preprocessor.process_article(raw_text)
        analytics_text = clean_data["analytics_text"]
        
        # We need a bit of substance to cluster properly
        if len(analytics_text) > 50:
            docs.append(analytics_text)
            timestamps.append(article.get("publishedAt", ""))
            titles.append(article.get("title", f"Article {idx}"))

    print(f"🧠 Training BERTopic model on {len(docs)} documents... (This may take a moment)")

    # Fine-tuning parameters for small datasets (like 20-100 articles)
    umap_model = UMAP(n_neighbors=min(5, len(docs)-1), n_components=5, min_dist=0.0, metric='cosine', random_state=42)
    hdbscan_model = HDBSCAN(min_cluster_size=min(3, len(docs)//5), metric='euclidean', cluster_selection_method='eom', prediction_data=True)
    vectorizer_model = CountVectorizer(stop_words="english", ngram_range=(1, 2))

    topic_model = BERTopic(
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        language="english",
        calculate_probabilities=True,
        verbose=True
    )

    topics, probs = topic_model.fit_transform(docs)

    print("\n✅ Modeling Complete!")
    print("-" * 50)
    
    # 1. IDENTIFY EMERGING MARITIME RISKS (Top 5 Macro Themes)
    topic_info = topic_model.get_topic_info()
    
    print("\n🚢 TOP 5 MACRO THEMES IDENTIFIED:")
    # Start from index 1 because index 0 is usually Topic -1 (Outliers)
    top_topics = topic_info[topic_info['Topic'] != -1].head(5)
    
    if top_topics.empty:
        print("Model could not form distinct clusters (dataset might be too small or too chaotic).")
        return

    for idx, row in top_topics.iterrows():
        topic_num = row['Topic']
        count = row['Count']
        # The 'Name' usually looks like "0_vessel_port_crew". Let's clean it.
        keywords = topic_model.get_topic(topic_num)
        top_words = ", ".join([word[0] for word in keywords[:5]])
        
        print(f"Theme {topic_num + 1} ({count} Articles): {top_words}")

    # 2. PRODUCE VISUALIZATIONS
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    print("\n📊 Generating Visualizations...")
    try:
        # Barchart for Theme Frequencies
        fig_barchart = topic_model.visualize_barchart(top_n_topics=6)

        # 3. TEMPORAL DISTRIBUTION CHART
        # Convert timestamps into a clean list of dates/strings (or datetime objects) for BERTopic
        # Handle cases where timestamps might be missing or in weird formats
        clean_timestamps = pd.to_datetime(timestamps, errors='coerce')
        # Fill NaT (Not a Time) with a default to avoid breaking the model
        if clean_timestamps.isna().any():
            clean_timestamps = clean_timestamps.fillna(pd.Timestamp('2024-01-01'))
            
        # Convert back to list of strings (BERTopic expects strings for timestamps by default)
        time_strings = clean_timestamps.strftime('%Y-%m-%d').tolist()

        # Generate Topics Over Time
        print("⏳ Calculating topics over time...")
        topics_over_time = topic_model.topics_over_time(docs, time_strings)
        
        # Visualize the temporal distribution
        fig_timeline = topic_model.visualize_topics_over_time(topics_over_time, top_n_topics=6)
        
        # Output combined Interactive HTML
        output_html = os.path.join(base_dir, "theme_dashboard.html")
        
        with open(output_html, 'w', encoding='utf-8') as f:
            f.write("<html><head><title>Maritime Themes</title>")
            # Add some basic styling so it matches the dark app
            f.write("""
            <style>
                body { background-color: #0d1117; color: #c9d1d9; font-family: sans-serif; padding: 20px; }
                .chart-container { margin-bottom: 40px; }
            </style>
            """)
            f.write("</head><body>")
            
            f.write("<h1>Maritime Intelligence - Theme Analysis</h1>")
            f.write("<p>Generated macro-theme frequencies and their temporal distribution over time.</p>")
            
            # --- Render Bar Chart ---
            f.write("<div class='chart-container'>")
            f.write("<h2>1. Theme Frequencies</h2>")
            f.write(fig_barchart.to_html(full_html=False, include_plotlyjs='cdn'))
            f.write("</div>")
            
            # --- Render Timeline Chart ---
            f.write("<div class='chart-container'>")
            f.write("<h2>2. Temporal Distribution (Emerging Trends)</h2>")
            f.write(fig_timeline.to_html(full_html=False, include_plotlyjs='cdn'))
            f.write("</div>")
            
            f.write("</body></html>")
            
        print(f"✅ Interactive Theme Dashboard saved to: {output_html}")
    except Exception as e:
        print(f"⚠️ Could not generate interactive charts: {e}")

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # We run this on the raw articles because BERTopic needs the full corpus to find relationships
    json_path = os.path.join(base_dir, "latest_articles.json")
    run_topic_modeling(json_path)
