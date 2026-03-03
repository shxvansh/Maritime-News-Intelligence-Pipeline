import json
import os
import re
import sys
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import logging
import transformers

# Suppress HuggingFace "UNEXPECTED" weight loading warnings
transformers.logging.set_verbosity_error()
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Pre-import all heavy libraries at module level — loaded once at server startup
from bertopic import BERTopic
from hdbscan import HDBSCAN
from umap import UMAP
from sklearn.feature_extraction.text import CountVectorizer

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pipeline.preprocessor import Preprocessor


# Maritime-domain stopwords to filter out from keyword labels
MARITIME_NOISE_WORDS = {
    # Scraper artifacts
    "mmmbl", "mmbbl", "shadow", "sail", "stay", "blue", "star", "blaze", "break",
    # Time words (irrelevant for themes)
    "january", "february", "march", "april", "june", "july", "august",
    "september", "october", "november", "december", "month", "year", "week",
    # Generic words that add no maritime signal
    "said", "also", "would", "could", "make", "used", "like", "include",
    "according", "following", "reported", "number", "total", "general",
    "meter", "metre", "length", "route", "suffer", "anchor", "build",
    "report", "record", "authority", "aground", "comoros", "kurtarma",
}

# Maritime keyword → human-readable theme label mapping
THEME_LABEL_MAP = [
    ({"inspection", "detention", "deficiency", "mou", "psc"},          "Port State Control & Inspections"),
    ({"piracy", "attack", "armed", "robbery", "hijack", "gulf", "aden"}, "Piracy & Maritime Security"),
    ({"sanction", "iran", "export", "oil", "tanker", "crude"},          "Sanctions & Geopolitical Tensions"),
    ({"crew", "seafarer", "sailor", "aboard", "fatigue", "welfare"},     "Crew Welfare & Labour"),
    ({"rescue", "distress", "salvage", "capsiz", "sinking", "emergency"}, "Vessel Distress & Rescue"),
    ({"container", "cargo", "loss", "damage", "freight"},               " Cargo & Container Operations"),
    ({"port", "terminal", "singapore", "congestion", "delay"},          " Port Operations & Congestion"),
    ({"environment", "emission", "carbon", "fuel", "lng", "sulfur"},    " Environmental & Fuel Compliance"),
    ({"collision", "grounding", "accident", "incident", "damage"},      " Vessel Accidents & Collisions"),
    ({"wind", "storm", "weather", "cyclone", "wave"},                   " Weather & Environmental Risk"),
    ({"vessel", "ship", "fleet", "operator", "owner"},                  " General Vessel Operations"),
]


def _auto_label_topic(keywords: list[str]) -> str:
    """Map a list of topic keywords to a human-readable label."""
    keyword_set = {k.lower() for k in keywords}

    best_label = None
    best_score = 0

    for theme_keywords, label in THEME_LABEL_MAP:
        # Score = how many keywords from this theme appear in the topic
        score = len(theme_keywords & keyword_set)
        if score > best_score:
            best_score = score
            best_label = label

    # Fallback: capitalise the top 3 clean keywords
    if best_label is None or best_score == 0:
        clean = [k.title() for k in keywords if len(k) > 3][:3]
        best_label = f" {' / '.join(clean)}" if clean else " Miscellaneous"

    return best_label


def _clean_keyword(word: str) -> str | None:
    """Return cleaned keyword or None if it should be filtered out."""
    w = word.lower().strip()
    if len(w) < 4:                          # too short
        return None
    if not re.match(r'^[a-z\s\-]+$', w):   # non-ASCII / numbers
        return None
    if w in MARITIME_NOISE_WORDS:           # known noise
        return None
    return w.title()                        # title Case for display


class AnalyticsEngine:
    """
    Pre-initialized analytics engine. Instantiate at server startup via the
    lifespan context manager so all BERTopic sub-model libraries are already
    loaded into memory when the /analytics/topic-model endpoint is called.
    """
    def __init__(self):
        print(" pre-init bertopic sub-models (umap, hdbscan, countvectorizer)...")
        self.BERTopic = BERTopic
        self.UMAP = UMAP
        self.HDBSCAN = HDBSCAN
        self.CountVectorizer = CountVectorizer
        self.Preprocessor = Preprocessor
        print(" bertopic sub-models pre-initialized.")

    def run(self, json_filepath: str) -> dict:
        """Run full topic modeling pipeline."""
        return _run_topic_modeling_impl(json_filepath, self)


def _run_topic_modeling_impl(json_filepath, engine=None):
    # handeling the edge case
    """main logic for topic modleing"""
    print(" init topic modeling & theme extraction...\n")

    if not os.path.exists(json_filepath):
        print(f" Could not find {json_filepath}.")
        return

    with open(json_filepath, 'r', encoding='utf-8') as f:
        articles = json.load(f)

    if len(articles) < 10:
        print(" not enough articles to perform meaningful clustering. need at least 10.")
        return

    print(f" Loaded {len(articles)} articles. Processing text...")

    docs, timestamps, titles = [], [], []
    preprocessor_class = engine.Preprocessor if engine else Preprocessor

    for idx, article in enumerate(articles):
        raw_text = article.get("content", "")
        if not raw_text:
            continue
        clean_data = preprocessor_class.process_article(raw_text)
        analytics_text = clean_data["analytics_text"]
        if len(analytics_text) > 50:
            docs.append(analytics_text)
            timestamps.append(article.get("publishedAt", ""))
            titles.append(article.get("title", f"Article {idx}"))

    print(f" Training BERTopic model on {len(docs)} documents...")

    umap_model    = UMAP(n_neighbors=min(5, len(docs) - 1), n_components=5,
                         min_dist=0.0, metric='cosine', random_state=42)
    hdbscan_model = HDBSCAN(min_cluster_size=min(3, len(docs) // 5),
                             metric='euclidean', cluster_selection_method='eom',
                             prediction_data=True)
    vectorizer_model = CountVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        min_df=2,                   # ignore words that appear in only 1 doc (reduces noise)
        token_pattern=r'\b[a-zA-Z]{4,}\b'  # only words ≥ 4 letters, no numbers
    )

    topic_model = BERTopic(
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        language="english",
        calculate_probabilities=True,
        verbose=True,
    )

    topics, probs = topic_model.fit_transform(docs)

    print("\n modeling complete!")
    print("-" * 50)

    # --- Build human-readable topic summary ---
    topic_info   = topic_model.get_topic_info()
    valid_topics = topic_info[topic_info['Topic'] != -1].head(6)

    if valid_topics.empty:
        print("model could not form distinct clusters.")
        return

    # Map topic_id → {label, keywords, count}, with deduplication
    topic_summaries = []
    used_labels: dict[str, int] = {}   # label → how many times used so far

    print("\n top maritime themes found:\n")
    for _, row in valid_topics.iterrows():
        topic_num    = row['Topic']
        count        = row['Count']
        raw_keywords = [word for word, _ in topic_model.get_topic(topic_num)]
        clean_kws    = [c for k in raw_keywords if (c := _clean_keyword(k))]
        base_label   = _auto_label_topic([k.lower() for k in raw_keywords])

        # Disambiguate duplicate labels
        if base_label in used_labels:
            # append the most distinctive clean keyword that isn't in the label text
            distinguish = next(
                (k for k in clean_kws if k.lower() not in base_label.lower()), None
            )
            label = f"{base_label} — {distinguish}" if distinguish else f"{base_label} ({count} arts.)"
        else:
            label = base_label
        used_labels[base_label] = used_labels.get(base_label, 0) + 1

        top5_display = ", ".join(clean_kws[:5]) if clean_kws else "—"
        topic_summaries.append({
            "id":       topic_num,
            "label":    label,
            "keywords": clean_kws[:8],
            "count":    int(count),
        })
        print(f"  {label}  ({count} articles)")
        print(f"  Keywords: {top5_display}\n")

    # --- Generate visualizations ---
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    print("\n gen visualizations...")
    try:
        # Rename topics in the model object so charts use our labels
        label_map = {s["id"]: f"{s['label']} ({s['count']} articles)"
                     for s in topic_summaries}
        topic_model.set_topic_labels(label_map)

        fig_barchart = topic_model.visualize_barchart(
            top_n_topics=6,
            custom_labels=True,
            title="Maritime Intelligence — Theme Keyword Scores",
            width=1100,
            height=500,
        )

        # Temporal chart
        clean_timestamps = pd.to_datetime(timestamps, errors='coerce')
        if clean_timestamps.isna().any():
            clean_timestamps = clean_timestamps.fillna(pd.Timestamp('2024-01-01'))
        time_strings = clean_timestamps.strftime('%Y-%m-%d').tolist()

        print(" calc topics over time...")
        topics_over_time = topic_model.topics_over_time(docs, time_strings)
        fig_timeline = topic_model.visualize_topics_over_time(
            topics_over_time,
            top_n_topics=6,
            custom_labels=True,
            title="Emerging Maritime Risks — Topic Frequency Over Time",
        )

        # build the readable html dashboard
        output_html = os.path.join(base_dir, "theme_dashboard.html")

        theme_cards_html = ""
        colors = ["#1f6feb", "#388bfd", "#58a6ff", "#79c0ff", "#a5d6ff", "#cae8ff"]
        for i, s in enumerate(topic_summaries):
            kw_tags = "".join(
                f'<span style="background:#21262d;border:1px solid #30363d;'
                f'padding:3px 10px;border-radius:12px;font-size:12px;'
                f'margin:3px;display:inline-block;color:#c9d1d9">{k}</span>'
                for k in s["keywords"]
            )
            color = colors[i % len(colors)]
            theme_cards_html += f"""
            <div style="background:#161b22;border:1px solid #21262d;border-radius:8px;
                        padding:18px 22px;margin-bottom:14px;
                        border-left:4px solid {color};">
                <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:10px;">
                    <h3 style="margin:0;color:#e6edf3;font-size:16px">{s['label']}</h3>
                    <span style="background:{color}22;color:{color};border:1px solid {color}55;
                                 padding:3px 12px;border-radius:20px;font-size:13px;font-weight:600">
                        {s['count']} articles
                    </span>
                </div>
                <div>{kw_tags}</div>
            </div>"""

        with open(output_html, 'w', encoding='utf-8') as f:
            f.write(f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>Maritime Theme Analysis</title>
  <style>
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      background-color: #0d1117;
      color: #c9d1d9;
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif;
      padding: 32px 40px;
      max-width: 1200px;
      margin: 0 auto;
    }}
    .header {{ margin-bottom: 32px; border-bottom: 1px solid #21262d; padding-bottom: 20px; }}
    .header h1 {{ color: #e6edf3; font-size: 24px; margin-bottom: 6px; }}
    .header p  {{ color: #8b949e; font-size: 14px; }}
    .section-title {{
      color: #e6edf3; font-size: 18px; font-weight: 600;
      margin: 32px 0 16px 0;
      padding-bottom: 8px;
      border-bottom: 1px solid #21262d;
    }}
    .plotly-chart {{ border-radius: 8px; overflow: hidden; margin-bottom: 8px; }}
  </style>
</head>
<body>
  <div class="header">
    <h1> Maritime Intelligence — Theme Analysis</h1>
    <p>BERTopic macro-theme extraction across all scraped maritime news articles.
       Themes are auto-labeled using domain keyword matching.</p>
  </div>

  <div class="section-title"> Identified Macro-Themes</div>
  {theme_cards_html}

  <div class="section-title"> Theme Keyword Scores</div>
  <p style="color:#8b949e;font-size:13px;margin-bottom:12px">
    Each bar shows how strongly a keyword defines its theme.
    Longer bar = more representative keyword.
  </p>
  <div class="plotly-chart">
    {fig_barchart.to_html(full_html=False, include_plotlyjs='cdn')}
  </div>

  <div class="section-title"> Emerging Trends Over Time</div>
  <p style="color:#8b949e;font-size:13px;margin-bottom:12px">
    How often each theme appeared in articles over the scraped time range.
    A rising line = an emerging maritime risk or developing situation.
  </p>
  <div class="plotly-chart">
    {fig_timeline.to_html(full_html=False, include_plotlyjs=False)}
  </div>

</body>
</html>""")

        print(f" Interactive Theme Dashboard saved to: {output_html}")

    except Exception as e:
        import traceback
        print(f" Could not generate charts: {e}")
        traceback.print_exc()


def run_topic_modeling(json_filepath):
    # todo check perfrmance here
    """entry point for older scripts"""
    return _run_topic_modeling_impl(json_filepath, engine=None)


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    json_path = os.path.join(base_dir, "latest_articles.json")
    run_topic_modeling(json_path)
