"""Advanced retrieval algorithms: BM25Okapi, Medical Insurance Synonym Expansion, and Reciprocal Rank Fusion (RRF)."""

import re
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Set
import numpy as np

from app.config import settings


@dataclass
class PolicyChunk:
    """Represents an atomic, searchable section or subsection of the policy."""
    section_id: str
    section_title: str
    category: str
    content: str
    keywords: List[str] = field(default_factory=list)

    @property
    def citation(self) -> str:
        return f"[{self.section_id}] {self.section_title}"


@dataclass
class SearchResult:
    """Represents a retrieval match with relevance scoring."""
    chunk: PolicyChunk
    score: float
    bm25_score: float = 0.0
    semantic_score: float = 0.0
    matched_terms: List[str] = field(default_factory=list)


# Domain-specific medical & insurance synonym taxonomy for query expansion
INSURANCE_TAXONOMY: Dict[str, List[str]] = {
    "maternity": ["childbirth", "delivery", "cesarean", "pregnancy", "prenatal", "labor"],
    "pregnancy": ["maternity", "childbirth", "delivery", "cesarean"],
    "cataract": ["eye", "vision", "surgery", "lens"],
    "rhinoplasty": ["cosmetic", "nose", "aesthetic", "elective"],
    "cosmetic": ["aesthetic", "plastic", "rhinoplasty", "hair transplant", "contouring"],
    "emergency": ["er", "urgent", "life-threatening", "ambulance", "trauma", "accidental"],
    "accident": ["injury", "trauma", "fall", "external", "sudden"],
    "preexisting": ["chronic", "prior", "pre-existing", "history", "antecedent", "diagnosed"],
    "pre-existing": ["chronic", "prior", "preexisting", "history", "antecedent", "diagnosed"],
    "waiting": ["period", "delay", "moratorium", "effective", "consecutive"],
    "appeal": ["dispute", "denial", "rejected", "grievance", "specialist", "escalate", "eob"],
    "reimbursement": ["coinsurance", "deductible", "copay", "coverage", "out-of-pocket", "payout"],
    "imaging": ["mri", "ct", "cat scan", "pet", "x-ray", "radiology"],
    "inpatient": ["hospitalization", "icu", "admitted", "room boarding", "surgery", "ward"],
    "outpatient": ["specialist", "consultation", "clinic", "office visit"]
}


class BM25Okapi:
    """Okapi BM25 ranking algorithm with length normalization and saturation."""

    def __init__(self, corpus: List[List[str]], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus_size = len(corpus)
        self.avgdl = sum(len(doc) for doc in corpus) / max(self.corpus_size, 1)
        self.doc_freqs: Dict[str, int] = {}
        self.idf: Dict[str, float] = {}
        self.doc_lens = [len(doc) for doc in corpus]
        self.corpus = corpus
        self._initialize()

    def _initialize(self) -> None:
        for doc in self.corpus:
            unique_terms = set(doc)
            for term in unique_terms:
                self.doc_freqs[term] = self.doc_freqs.get(term, 0) + 1

        for term, freq in self.doc_freqs.items():
            # BM25 probabilistic IDF formula
            self.idf[term] = math.log((self.corpus_size - freq + 0.5) / (freq + 0.5) + 1.0)

    def get_scores(self, query_tokens: List[str]) -> np.ndarray:
        scores = np.zeros(self.corpus_size)
        for i, doc in enumerate(self.corpus):
            doc_len = self.doc_lens[i]
            len_norm = 1.0 - self.b + self.b * (doc_len / max(self.avgdl, 1.0))
            term_counts: Dict[str, int] = {}
            for t in doc:
                term_counts[t] = term_counts.get(t, 0) + 1

            doc_score = 0.0
            for q in query_tokens:
                if q in term_counts:
                    tf = term_counts[q]
                    idf_val = self.idf.get(q, 0.1)
                    numerator = tf * (self.k1 + 1.0)
                    denominator = tf + self.k1 * len_norm
                    doc_score += idf_val * (numerator / max(denominator, 1e-6))
            scores[i] = doc_score
        return scores


class PolicyRetriever:
    """Enterprise hybrid policy retriever using BM25Okapi + TF-IDF Semantic Dense Similarity + RRF."""

    def __init__(self, policy_path: Optional[Path] = None):
        self.policy_path = policy_path or settings.resolved_policy_path
        self.chunks: List[PolicyChunk] = []
        self.bm25: Optional[BM25Okapi] = None
        self.vocabulary: Dict[str, int] = {}
        self.idf_vec: np.ndarray = np.array([])
        self.tfidf_matrix: np.ndarray = np.array([])
        self._load_and_index()

    def _tokenize(self, text: str) -> List[str]:
        clean = re.sub(r"[^\w\s-]", " ", text.lower())
        tokens = re.findall(r"\b[a-z0-9_-]{2,}\b", clean)
        stopwords = {
            "the", "and", "for", "with", "that", "this", "are", "from",
            "per", "under", "all", "after", "before", "not", "any", "over",
            "have", "has", "had", "been", "was", "were", "shall", "will", "does"
        }
        return [t for t in tokens if t not in stopwords]

    def _expand_query(self, tokens: List[str]) -> List[str]:
        """Expands query with domain synonyms from medical insurance ontology."""
        expanded = list(tokens)
        for t in tokens:
            if t in INSURANCE_TAXONOMY:
                expanded.extend(INSURANCE_TAXONOMY[t][:3])
        return list(dict.fromkeys(expanded))

    def _load_and_index(self) -> None:
        if not self.policy_path.exists():
            raise FileNotFoundError(f"Policy document not found at {self.policy_path}")

        raw_text = self.policy_path.read_text(encoding="utf-8")
        self.chunks = self._parse_markdown(raw_text)

        tokenized_corpus = [
            self._tokenize(f"{c.section_id} {c.section_title} {c.category} {c.content}")
            for c in self.chunks
        ]

        # 1. Initialize BM25Okapi
        self.bm25 = BM25Okapi(tokenized_corpus)

        # 2. Build TF-IDF semantic vector space
        vocab = {}
        for doc in tokenized_corpus:
            for t in doc:
                if t not in vocab:
                    vocab[t] = len(vocab)
        self.vocabulary = vocab

        num_docs = len(self.chunks)
        vocab_size = len(vocab)
        if vocab_size == 0 or num_docs == 0:
            return

        df = np.zeros(vocab_size)
        for doc in tokenized_corpus:
            for t in set(doc):
                df[vocab[t]] += 1

        self.idf_vec = np.log((1 + num_docs) / (1 + df)) + 1.0

        matrix = np.zeros((num_docs, vocab_size))
        for i, tokens in enumerate(tokenized_corpus):
            if not tokens:
                continue
            tf_counts = {}
            for t in tokens:
                tf_counts[t] = tf_counts.get(t, 0) + 1
            for t, count in tf_counts.items():
                col = vocab[t]
                tf = count / len(tokens)
                matrix[i, col] = tf * self.idf_vec[col]
            norm = np.linalg.norm(matrix[i])
            if norm > 0:
                matrix[i] = matrix[i] / norm
        self.tfidf_matrix = matrix

    def _parse_markdown(self, markdown_text: str) -> List[PolicyChunk]:
        chunks: List[PolicyChunk] = []
        lines = markdown_text.splitlines()
        current_major_heading = "General"
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if line.startswith("## "):
                current_major_heading = line.lstrip("#").strip()
                i += 1
                continue

            sub_match = re.match(r"^-\s+\*\*(Section\s+[0-9]+(?:\.[0-9]+)?)\s*(.*?)\*\*:\s*(.*)$", line)
            if sub_match:
                sec_id = sub_match.group(1).strip()
                sec_title = sub_match.group(2).strip().rstrip(":")
                content_start = sub_match.group(3).strip()
                content_lines = [content_start]
                i += 1
                while i < len(lines) and not lines[i].strip().startswith("## ") and not re.match(r"^-\s+\*\*Section", lines[i].strip()):
                    sub_line = lines[i].strip()
                    if sub_line.startswith("---"):
                        break
                    if sub_line:
                        content_lines.append(sub_line)
                    i += 1
                full_content = " ".join(content_lines)
                chunks.append(PolicyChunk(
                    section_id=sec_id,
                    section_title=sec_title,
                    category=current_major_heading,
                    content=full_content,
                    keywords=self._tokenize(f"{sec_title} {full_content}")[:12]
                ))
                continue

            if re.match(r"^\d+\.\s+", line) and "Section" in current_major_heading:
                content_lines = [line]
                i += 1
                while i < len(lines) and not lines[i].strip().startswith("## ") and not lines[i].strip().startswith("---"):
                    if lines[i].strip():
                        content_lines.append(lines[i].strip())
                    i += 1
                chunks.append(PolicyChunk(
                    section_id=f"{current_major_heading.split('.')[0].strip()}",
                    section_title=f"{current_major_heading}",
                    category=current_major_heading,
                    content=" ".join(content_lines)
                ))
                continue

            i += 1

        if not chunks:
            sections = markdown_text.split("## ")
            for sec in sections[1:]:
                parts = sec.split("\n", 1)
                title = parts[0].strip()
                body = parts[1].strip() if len(parts) > 1 else ""
                chunks.append(PolicyChunk(
                    section_id=title.split(" ")[0] if "Section" in title else "Section",
                    section_title=title,
                    category="Policy",
                    content=body
                ))
        return chunks

    def search(self, query: str, top_k: Optional[int] = None, threshold: Optional[float] = None) -> List[SearchResult]:
        """
        Hybrid Reciprocal Rank Fusion (RRF) combining:
        1. BM25Okapi scoring on expanded lexical query
        2. Cosine Vector Similarity on dense TF-IDF space
        3. Exact section ID & title boosting
        """
        k = top_k or settings.TOP_K_CHUNKS
        raw_tokens = self._tokenize(query)
        if not raw_tokens or len(self.chunks) == 0:
            return []

        expanded_tokens = self._expand_query(raw_tokens)

        # 1. BM25 Scores
        bm25_scores = self.bm25.get_scores(expanded_tokens)
        max_bm25 = float(np.max(bm25_scores)) if len(bm25_scores) > 0 and np.max(bm25_scores) > 0 else 1.0
        norm_bm25 = bm25_scores / max_bm25

        # 2. Vector Cosine Similarity
        q_vec = np.zeros(len(self.vocabulary))
        for t in raw_tokens:
            if t in self.vocabulary:
                q_vec[self.vocabulary[t]] += 1.0
        if np.linalg.norm(q_vec) > 0:
            q_vec = (q_vec * self.idf_vec) / np.linalg.norm(q_vec * self.idf_vec)
        cos_scores = np.dot(self.tfidf_matrix, q_vec)

        # 3. Reciprocal Rank Fusion (RRF) with k=60
        rrf_constant = 60.0
        bm25_ranked = np.argsort(-bm25_scores)
        cos_ranked = np.argsort(-cos_scores)

        rrf_scores = np.zeros(len(self.chunks))
        for rank, idx in enumerate(bm25_ranked):
            rrf_scores[idx] += 0.55 / (rrf_constant + rank + 1)
        for rank, idx in enumerate(cos_ranked):
            rrf_scores[idx] += 0.45 / (rrf_constant + rank + 1)

        # Normalize RRF
        max_rrf = float(np.max(rrf_scores)) if np.max(rrf_scores) > 0 else 1.0
        norm_rrf = rrf_scores / max_rrf

        results: List[SearchResult] = []
        lower_q = query.lower()

        for idx, chunk in enumerate(self.chunks):
            score = float(norm_rrf[idx])

            # Exact section ID boost (e.g. "3.2" or "Section 3.2")
            clean_sec = chunk.section_id.lower().replace("section", "").strip()
            if clean_sec and clean_sec in lower_q:
                score += 0.45

            if chunk.section_title.lower() in lower_q:
                score += 0.35

            chunk_text = f"{chunk.section_id} {chunk.section_title} {chunk.content}".lower()
            matched = [t for t in raw_tokens if t in chunk_text]

            final_score = min(1.0, round(score, 4))
            results.append(SearchResult(
                chunk=chunk,
                score=final_score,
                bm25_score=round(float(norm_bm25[idx]), 4),
                semantic_score=round(float(cos_scores[idx]), 4),
                matched_terms=matched
            ))

        results.sort(key=lambda r: r.score, reverse=True)
        return results[:k]

    def get_chunk_by_section(self, section_id: str) -> Optional[PolicyChunk]:
        target = section_id.lower().strip()
        for c in self.chunks:
            if c.section_id.lower().strip() == target or target in c.section_id.lower():
                return c
        return None

    def get_all_chunks(self) -> List[PolicyChunk]:
        return self.chunks
