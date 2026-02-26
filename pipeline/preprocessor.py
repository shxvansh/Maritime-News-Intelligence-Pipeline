import re
import hashlib
import spacy
import sys

# Load spaCy model for sentence segmentation and stopword removal
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Downloading en_core_web_sm...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm")

class Preprocessor:
    @staticmethod
    def clean_markdown(text: str) -> str:
        """Removes Markdown image tags and extracts text from Markdown links."""
        if not text:
            return ""
        
        # Remove Markdown images: ![alt text](url)
        text = re.sub(r'!\[.*?\]\([^)]+\)', '', text)
        
        # Replace Markdown links with just the text: [link text](url) -> link text
        text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)
        
        # Clean up excessive whitespace padding
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r' {2,}', ' ', text)
        
        return text.strip()

    @staticmethod
    def compute_hash(text: str) -> str:
        """Computes a SHA-256 hash of the cleaned text."""
        return hashlib.sha256(text.encode('utf-8')).hexdigest()

    @staticmethod
    def segment_sentences(text: str) -> list[str]:
        """Segments text into sentences using spaCy."""
        doc = nlp(text)
        return [sent.text.strip() for sent in doc.sents if sent.text.strip()]
        
    @staticmethod
    def normalize_for_analytics(text: str) -> str:
        """Creates a lowercase, punctuation-free, stopword-free version of the text for topic modeling."""
        doc = nlp(text)
        tokens = [token.lemma_.lower() for token in doc if not token.is_stop and not token.is_punct and token.text.strip()]
        return " ".join(tokens)
        
    @classmethod
    def process_article(cls, raw_text: str) -> dict:
        """Runs the full preprocessing pipeline on a single article's content."""
        llm_ready_text = cls.clean_markdown(raw_text)
        article_hash = cls.compute_hash(llm_ready_text)
        sentences = cls.segment_sentences(llm_ready_text)
        analytics_text = cls.normalize_for_analytics(llm_ready_text)
        
        return {
            "llm_ready_text": llm_ready_text,
            "hash": article_hash,
            "sentences": sentences,
            "analytics_text": analytics_text
        }
