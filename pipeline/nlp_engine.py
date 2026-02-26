from gliner import GLiNER
from transformers import pipeline as hf_pipeline
import torch
import warnings

# Suppress some transformers warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)

class NLPEngine:
    def __init__(self):
        print("Loading GLiNER model (this might take a minute on first run)...")
        # Ensure we use MPS if available (Apple Silicon), else CPU
        device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        self.ner_model = GLiNER.from_pretrained("urchade/gliner_medium-v2.1").to(device)
        self.ner_labels = ["Vessel Name", "Port", "Organization", "Country", "Person", "Incident Type", "Date", "Cargo"]
        
        print(f"Loading Zero-Shot Classifier to {device}...")
        kwargs = {}
        if device.type == 'mps':
            kwargs['device'] = 'mps'
        self.classifier = hf_pipeline("zero-shot-classification", model="facebook/bart-large-mnli", **kwargs)
        self.classification_labels = [
            "Collision", "Piracy", "Sanctions", "Military Activity", 
            "Port Disruption", "Environmental Incident", 
            "Financial or Shipping Markets", "Regulatory Development", "Accident"
        ]

    def extract_entities(self, text: str) -> list[dict]:
        """Extracts maritime-specific entities using GLiNER."""
        # Truncate text marginally to avoid exceeding token limits during standard runs
        safe_text = text[:1500] if len(text) > 1500 else text
        entities = self.ner_model.predict_entities(safe_text, self.ner_labels)
        return entities

    def classify_incident(self, text: str) -> dict:
        """Classifies the article into a predefined maritime incident category."""
        safe_text = text[:1500] if len(text) > 1500 else text
        result = self.classifier(
            safe_text, 
            candidate_labels=self.classification_labels, 
            multi_label=True # This is crucial for non-mutually exclusive categories
        )
        
        # Result format: {'sequence': text, 'labels': ['Piracy', ...], 'scores': [0.9, ...]}
        top_label = result['labels'][0]
        top_score = result['scores'][0]
        
        return {
            "label": top_label,
            "score": top_score,
            "is_confident": top_score > 0.85 # Threshold for relying solely on zero-shot
        }
