import re
import unicodedata
import hashlib
from typing import List
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine


class SecurityManager:
    """
    Production-hardened security layer for RAG systems.
    Backward compatible with existing method signatures.
    """

    DEFAULT_MAX_QUERY_LENGTH = 4000
    DEFAULT_MIN_OVERLAP = 0.2

    def __init__(self, max_query_length: int = DEFAULT_MAX_QUERY_LENGTH):
        self.max_query_length = max_query_length

        # -----------------------------
        # 1. Prompt Injection Patterns
        # -----------------------------
        self.injection_patterns: List[str] = [
            r"ignore\s+(all\s+)?(previous\s+)?instructions",
            r"system\s+prompt",
            r"developer\s+mode",
            r"jailbreak",
            r"disregard\s+(all\s+)?instructions",
            r"bypass\s+(security|restrictions|guardrails)",
            r"you\s+are\s+now",
            r"print\s+(all\s+)?rules",
            r"reveal\s+(the\s+)?(hidden|system)\s+prompt",
        ]

        # Obfuscation-resistant pattern
        self.compiled_patterns = [
            re.compile(pattern, re.IGNORECASE)
            for pattern in self.injection_patterns
        ]

        # -----------------------------
        # 2. Secret / Token Patterns
        # -----------------------------
        self.secret_patterns = [
            r"sk-[a-zA-Z0-9]{20,}",  # OpenAI-style keys
            r"AKIA[0-9A-Z]{16}",     # AWS Access Key
            r"(?i)api[_-]?key\s*[:=]\s*[A-Za-z0-9\-_]{10,}",
            r"(?i)bearer\s+[A-Za-z0-9\-_\.]+",
        ]
        self.compiled_secret_patterns = [
            re.compile(p) for p in self.secret_patterns
        ]

        # -----------------------------
        # 3. PII Engine
        # -----------------------------
        self.analyzer = AnalyzerEngine()
        self.anonymizer = AnonymizerEngine()

    # ==========================================================
    # Utilities
    # ==========================================================

    def _normalize_text(self, text: str) -> str:
        """
        Normalize text to reduce obfuscation attacks.
        """
        if not text:
            return ""

        # Unicode normalization
        text = unicodedata.normalize("NFKC", text)

        # Collapse excessive whitespace
        text = re.sub(r"\s+", " ", text)

        return text.strip()

    def _hash_text(self, text: str) -> str:
        """
        Hash sensitive content for logging without leaking it.
        """
        return hashlib.sha256(text.encode()).hexdigest()

    # ==========================================================
    # 1. Prompt Injection Detection
    # ==========================================================

    def check_prompt_injection(self, query: str) -> bool:
        """
        Returns True if a prompt injection is detected, False otherwise.
        """

        if not query:
            return False

        normalized_query = self._normalize_text(query)

        # Length guard (configurable)
        if len(normalized_query) > self.max_query_length:
            return True

        # Pattern matching
        for pattern in self.compiled_patterns:
            if pattern.search(normalized_query):
                return True

        # Secret detection (prevent prompt leaking attempts)
        for pattern in self.compiled_secret_patterns:
            if pattern.search(normalized_query):
                return True

        return False

    # ==========================================================
    # 2. PII Sanitization
    # ==========================================================

    def sanitize_pii(self, text: str) -> str:
        """
        Masks PII elements like Emails, Phone Numbers, IP Addresses, etc.
        Keeps domain-relevant entities intact.
        """

        if not text:
            return text

        try:
            normalized_text = self._normalize_text(text)

            results = self.analyzer.analyze(
                text=normalized_text,
                entities=[
                    "EMAIL_ADDRESS",
                    "PHONE_NUMBER",
                    "IP_ADDRESS",
                    "CREDIT_CARD",
                    "SOCIAL_SECURITY_NUMBER",
                    "US_BANK_NUMBER",
                    "PERSON",
                    "IBAN_CODE",
                ],
                language="en",
            )

            anonymized_result = self.anonymizer.anonymize(
                text=normalized_text,
                analyzer_results=results,
            )

            # Additional secret masking (defense-in-depth)
            sanitized_text = anonymized_result.text
            for pattern in self.compiled_secret_patterns:
                sanitized_text = pattern.sub("[REDACTED_SECRET]", sanitized_text)

            return sanitized_text

        except Exception:
            # Fail-safe: never crash pipeline due to PII
            return text

    # ==========================================================
    # 3. Grounding Check (Improved Lightweight Version)
    # ==========================================================

    def check_grounding(
        self,
        answer: str,
        context: str,
        min_overlap: float = DEFAULT_MIN_OVERLAP,
    ) -> bool:
        """
        Lightweight hallucination check.

        Uses:
        - Word overlap
        - Named entity overlap
        - Minimum answer length check
        """

        if not answer or not context:
            return True

        rejection_phrases = [
            "cannot be determined",
            "no relevant information",
            "not configured",
            "insufficient data",
        ]

        if any(phrase in answer.lower() for phrase in rejection_phrases):
            return True

        # Normalize
        orig_answer = self._normalize_text(answer)
        orig_context = self._normalize_text(context)
        
        answer_lower = orig_answer.lower()
        context_lower = orig_context.lower()

        answer_words = set(re.findall(r"\b[a-z]{4,}\b", answer_lower))
        context_words = set(re.findall(r"\b[a-z]{4,}\b", context_lower))

        if not answer_words:
            return True

        overlap = answer_words.intersection(context_words)
        overlap_ratio = len(overlap) / max(len(answer_words), 1)

        # Additional heuristic: entity consistency (MUST use original cased text)
        answer_entities = set(re.findall(r"\b[A-Z][a-z]{2,}\b", orig_answer))
        context_entities = set(re.findall(r"\b[A-Z][a-z]{2,}\b", orig_context))

        if answer_entities and not answer_entities.intersection(context_entities):
            return False

        return overlap_ratio >= min_overlap