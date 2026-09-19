"""
Knowledge Area Interpretation Pipeline (Step 11).

Interprets semantic clusters discovered by HDBSCAN (Step 10) into candidate
engineering knowledge areas using deterministic evidence extraction.

Pipeline:
    HDBSCAN Cluster Assignments (cluster_id >= 0, -1 for noise)
                    ↓
    Cluster Document Grouping & Evidence Inspection
    (Commits, CommitFiles, PRs, PRFiles, Contained Commits)
                    ↓
    Evidence Extraction:
    - Representative Documents (ranked by membership probability & richness)
    - Important File Paths (normalized, frequency-ranked file paths)
    - Representative Terms (weighted tokenization & TF-IDF term scoring)
    - PR Author vs Commit Author Separation (preserving distinct identities)
                    ↓
    Candidate Label Generation
    (Deterministic, explainable, evidence-grounded domain labels)
                    ↓
    Structured Interpretation Result with Full Traceability

STRICT SCOPE CONSTRAINTS:
- No LLM (no OpenAI, no Gemini, no OpenRouter).
- No Neo4j modifications (no (:KnowledgeArea) nodes, no Step 9 modifications).
- No Supabase schema changes.
- No developer expertise scoring, no knowledge risk scoring, no developer ranking.
- Purely deterministic, in-memory evidence-based candidate knowledge area identification.
"""
import logging
import math
import re
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np

from knowledge.clustering import (
    KnowledgeDocument,
    load_knowledge_documents,
    build_embeddings,
    cluster_embeddings,
    run_knowledge_clustering,
    _log_progress,
)
from database.supabase_client import SupabaseClient
from vector.embeddings import EmbeddingService

logger = logging.getLogger(__name__)

# Standard English stopwords
ENGLISH_STOPWORDS: Set[str] = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
    "any", "are", "aren't", "as", "at", "be", "because", "been", "before", "being",
    "below", "between", "both", "but", "by", "can", "can't", "cannot", "could",
    "couldn't", "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down",
    "during", "each", "few", "for", "from", "further", "had", "hadn't", "has",
    "hasn't", "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her",
    "here", "here's", "hers", "herself", "him", "himself", "his", "how", "how's",
    "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't", "it",
    "it's", "its", "itself", "let's", "me", "more", "most", "mustn't", "my",
    "myself", "no", "nor", "not", "of", "off", "on", "once", "only", "or",
    "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "same",
    "shan't", "she", "she'd", "she'll", "she's", "should", "shouldn't", "so",
    "some", "such", "than", "that", "that's", "the", "their", "theirs", "them",
    "themselves", "then", "there", "there's", "these", "they", "they'd", "they'll",
    "they're", "they've", "this", "those", "through", "to", "too", "under", "until",
    "up", "very", "was", "wasn't", "we", "we'd", "we'll", "we're", "we've", "were",
    "weren't", "what", "what's", "when", "when's", "where", "where's", "which",
    "while", "who", "who's", "whom", "why", "why's", "with", "won't", "would",
    "wouldn't", "you", "you'd", "you'll", "you're", "you've", "your", "yours",
    "yourself", "yourselves", "just", "will", "now", "also", "using", "used", "use",
}

# VCS, Git, Programming syntax and boilerplate stopwords
CODE_GIT_STOPWORDS: Set[str] = {
    # Git & VCS terms
    "commit", "commits", "committed", "file", "files", "pull", "request", "requests",
    "pr", "prs", "merge", "merges", "merged", "branch", "branches", "master", "main",
    "patch", "patches", "diff", "diffs", "repository", "repo", "sha", "hash",
    "author", "author_login", "unknown", "none", "null", "true", "false", "status",
    "modified", "additions", "deletions", "changes", "change", "changed", "changing",
    "add", "added", "adding", "adds", "update", "updated", "updating", "updates",
    "fix", "fixed", "fixing", "fixes", "delete", "deleted", "remove", "removed",
    "removing", "create", "created", "creating", "initial", "code", "line", "lines",
    "test", "tests", "testing", "tested", "todo", "wip", "rfc", "blob", "url",
    "github", "http", "https", "com", "org", "net", "io", "preview", "truncated",
    "stats", "date", "timestamp", "created_at", "closed_at", "merged_at",
    "title", "description", "body", "state", "open", "closed", "number",
    # Programming language keywords and extensions
    "py", "ts", "js", "json", "md", "txt", "html", "css", "yaml", "yml", "xml", "sh",
    "sql", "import", "from", "def", "class", "return", "self", "package", "const",
    "let", "var", "function", "async", "await", "try", "except", "catch", "throw",
    "new", "public", "private", "protected", "void", "int", "str", "float", "bool",
    "list", "dict", "set", "tuple", "any", "optional", "union",
}

ALL_STOPWORDS: Set[str] = ENGLISH_STOPWORDS | CODE_GIT_STOPWORDS

# Domain taxonomy for explainable candidate labeling
DOMAIN_CONCEPT_PATTERNS: List[Tuple[Set[str], str]] = [
    # Payment & Financial Processing
    (
        {"payment", "retry", "transaction"},
        "Payment Processing & Retry Handling",
    ),
    (
        {"payment", "refund", "charge"},
        "Payment Processing & Refunds",
    ),
    (
        {"payment", "gateway", "stripe"},
        "Payment Gateway Integration",
    ),
    (
        {"payment", "billing", "subscription"},
        "Billing & Subscription Management",
    ),
    (
        {"payment", "transaction"},
        "Payment Processing",
    ),
    (
        {"payment"},
        "Payment Processing",
    ),
    # Caching & State
    (
        {"cache", "redis", "invalidation"},
        "Caching & Cache Invalidation",
    ),
    (
        {"cache", "redis"},
        "Redis Caching",
    ),
    (
        {"cache", "memcached"},
        "Caching Service",
    ),
    (
        {"cache", "ttl"},
        "Caching & Expiration",
    ),
    (
        {"cache"},
        "Caching",
    ),
    (
        {"redis"},
        "Redis Data Services",
    ),
    # Authentication & Authorization
    (
        {"auth", "jwt", "token"},
        "JWT Authentication & Token Management",
    ),
    (
        {"auth", "oauth", "sso"},
        "OAuth & Single Sign-On",
    ),
    (
        {"auth", "permission", "rbac"},
        "Role-Based Access Control",
    ),
    (
        {"auth", "login", "session"},
        "Authentication & Session Management",
    ),
    (
        {"auth", "security"},
        "Authentication & Security",
    ),
    (
        {"auth"},
        "Authentication",
    ),
    # Database & Data Persistence
    (
        {"database", "migration", "schema"},
        "Database Migrations & Schema",
    ),
    (
        {"database", "model", "orm"},
        "Data Models & Persistence",
    ),
    (
        {"database", "query", "index"},
        "Database Queries & Indexing",
    ),
    (
        {"database", "postgres"},
        "PostgreSQL Persistence",
    ),
    (
        {"migration", "schema"},
        "Schema Migrations",
    ),
    (
        {"database"},
        "Database Persistence",
    ),
    # Messaging, Webhooks & Events
    (
        {"webhook", "event", "callback"},
        "Webhook Processing & Callbacks",
    ),
    (
        {"webhook", "retry"},
        "Webhook Delivery & Retries",
    ),
    (
        {"webhook"},
        "Webhook Integration",
    ),
    (
        {"queue", "kafka", "rabbitmq"},
        "Message Queue & Event Streaming",
    ),
    (
        {"event", "pubsub", "listener"},
        "Event-Driven Architecture",
    ),
    # API & Routing
    (
        {"api", "router", "endpoint"},
        "API Routing & Endpoints",
    ),
    (
        {"api", "controller", "handler"},
        "API Request Handling",
    ),
    (
        {"api", "middleware"},
        "API Middleware & Filters",
    ),
    (
        {"api"},
        "API Services",
    ),
    # Notifications & Communications
    (
        {"notification", "email", "sms"},
        "Multi-Channel Notifications",
    ),
    (
        {"notification", "alert"},
        "Alerting & Notifications",
    ),
    (
        {"notification"},
        "Notification Services",
    ),
    (
        {"email"},
        "Email Delivery Service",
    ),
    # Observability & Monitoring
    (
        {"metric", "telemetry", "prometheus"},
        "Metrics & Telemetry",
    ),
    (
        {"logger", "logging", "tracing"},
        "Logging & Distributed Tracing",
    ),
    (
        {"monitoring", "healthcheck"},
        "Health Monitoring & Diagnostics",
    ),
    (
        {"logging"},
        "Logging & Auditing",
    ),
    # Order & E-Commerce
    (
        {"order", "checkout", "cart"},
        "Order Processing & Checkout",
    ),
    (
        {"order", "fulfillment"},
        "Order Fulfillment",
    ),
    (
        {"order"},
        "Order Management",
    ),
    # Inventory & Catalog
    (
        {"inventory", "stock"},
        "Inventory & Stock Management",
    ),
    (
        {"catalog", "product"},
        "Product Catalog Services",
    ),
    # User & Account
    (
        {"user", "profile", "account"},
        "User Account & Profile Management",
    ),
    (
        {"user", "customer"},
        "Customer Data Management",
    ),
    # Search & Query
    (
        {"search", "indexing", "elastic"},
        "Search & Indexing Engine",
    ),
    (
        {"search", "filter"},
        "Search & Filtering",
    ),
    # Fraud & Risk
    (
        {"fraud", "detection", "risk"},
        "Fraud Detection & Risk Management",
    ),
    (
        {"ledger", "accounting"},
        "Ledger & Financial Accounting",
    ),
]


def tokenize_text(text: str) -> List[str]:
    """
    Tokenizes input text into normalized lower-case terms.
    Splits on whitespace, punctuation, snake_case, and camelCase.
    Filters numbers, short words (< 3 chars), and hex hashes.
    """
    if not text:
        return []

    # 1. Split camelCase (e.g. PaymentGateway -> Payment Gateway)
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)

    # 2. Replace non-alphanumeric characters with spaces (splits snake_case, slashes, dashes)
    cleaned = re.sub(r"[^a-zA-Z0-9]", " ", text)

    tokens: List[str] = []
    for raw_word in cleaned.split():
        word = raw_word.strip().lower()
        if len(word) < 3:
            continue
        # Filter pure numeric tokens (e.g. "123", "45")
        if word.isdigit():
            continue
        # Filter hex commit hashes (e.g. 7+ hex chars like abc1234, deadbeef)
        if len(word) >= 7 and re.fullmatch(r"[0-9a-f]+", word):
            continue
        if word in ALL_STOPWORDS:
            continue
        tokens.append(word)

    return tokens


def extract_file_paths(documents: List[KnowledgeDocument], max_files: int = 10) -> List[str]:
    """
    Extracts, normalizes, and ranks file paths from documents belonging to a cluster.
    Evidence sources:
    - metadata["filename"]
    - metadata["changed_files"]
    - document.text parsed changed file blocks
    - contained_commits in PR documents
    """
    file_counts: Dict[str, int] = {}

    for doc in documents:
        meta = doc.metadata or {}

        # Direct filename on CommitFile or PRFile
        direct_fn = meta.get("filename")
        if direct_fn and isinstance(direct_fn, str):
            clean_fn = direct_fn.strip().replace("\\", "/")
            if clean_fn and not clean_fn.startswith("..."):
                file_counts[clean_fn] = file_counts.get(clean_fn, 0) + 3

        # List of changed files in Commit or PR metadata
        changed_files = meta.get("changed_files")
        if isinstance(changed_files, list):
            for fn in changed_files:
                if isinstance(fn, str):
                    clean_fn = fn.strip().replace("\\", "/")
                    if clean_fn and not clean_fn.startswith("..."):
                        file_counts[clean_fn] = file_counts.get(clean_fn, 0) + 2

        # Parse text lines for files: e.g. "- payment/retry.py" or "Changed file:\npayment/retry.py"
        lines = doc.text.split("\n")
        in_files_section = False
        for line in lines:
            line_str = line.strip()
            if not line_str:
                continue
            if line_str.lower().startswith("changed files:") or line_str.lower().startswith("pr changed files:"):
                in_files_section = True
                continue
            if in_files_section:
                if line_str.startswith("- "):
                    candidate = line_str[2:].strip().replace("\\", "/")
                    if candidate and not candidate.startswith("..."):
                        file_counts[candidate] = file_counts.get(candidate, 0) + 1
                elif line_str.startswith("...") or not line_str:
                    in_files_section = False

            if line_str.lower() == "changed file:":
                # Next line might be filename
                continue

    # Sort file paths by frequency descending, then alphabetically for deterministic stability
    ranked_files = sorted(file_counts.keys(), key=lambda fn: (-file_counts[fn], fn))
    return ranked_files[:max_files]


def extract_representative_terms(
    documents: List[KnowledgeDocument],
    all_cluster_documents: Optional[Dict[int, List[KnowledgeDocument]]] = None,
    top_k: int = 10,
) -> List[str]:
    """
    Extracts representative terms from documents within a cluster using weighted frequency
    and corpus distinctiveness (TF-IDF).

    Weighted evidence:
    - PR title, commit messages, filename base names: weight = 3
    - Body, descriptions, path components: weight = 2
    - General text / patch diffs: weight = 1
    """
    cluster_term_weights: Dict[str, float] = {}

    for doc in documents:
        meta = doc.metadata or {}

        # High priority: PR title or commit message
        high_priority_texts = []
        if meta.get("title"):
            high_priority_texts.append(str(meta["title"]))
        if meta.get("message"):
            high_priority_texts.append(str(meta["message"]))
        if meta.get("commit_message"):
            high_priority_texts.append(str(meta["commit_message"]))

        # Check PR contained commits
        contained_commits = meta.get("contained_commits")
        if isinstance(contained_commits, list):
            for c in contained_commits:
                if isinstance(c, dict) and c.get("message"):
                    high_priority_texts.append(str(c["message"]))

        # Filename base tokens get high priority
        direct_fn = meta.get("filename")
        if direct_fn and isinstance(direct_fn, str):
            high_priority_texts.append(direct_fn)

        for text in high_priority_texts:
            for token in tokenize_text(text):
                cluster_term_weights[token] = cluster_term_weights.get(token, 0.0) + 3.0

        # Medium priority: PR description/body or changed files
        med_priority_texts = []
        if meta.get("body"):
            med_priority_texts.append(str(meta["body"]))
        changed_files = meta.get("changed_files")
        if isinstance(changed_files, list):
            for cf in changed_files:
                if isinstance(cf, str):
                    med_priority_texts.append(cf)

        for text in med_priority_texts:
            for token in tokenize_text(text):
                cluster_term_weights[token] = cluster_term_weights.get(token, 0.0) + 2.0

        # General text fallback (first 1000 characters to avoid huge diff noise)
        general_text = doc.text[:1000]
        for token in tokenize_text(general_text):
            cluster_term_weights[token] = cluster_term_weights.get(token, 0.0) + 1.0

    if not cluster_term_weights:
        return []

    # If all_cluster_documents is provided and contains > 1 cluster, compute IDF distinctiveness
    if all_cluster_documents and len(all_cluster_documents) > 1:
        n_clusters = len(all_cluster_documents)
        scored_terms: Dict[str, float] = {}

        for term, tf in cluster_term_weights.items():
            # Document frequency = in how many clusters does this term appear?
            df = sum(
                1 for c_docs in all_cluster_documents.values()
                if any(term in tokenize_text(d.text[:500]) for d in c_docs)
            )
            df = max(1, df)
            idf = math.log(1.0 + (n_clusters / df))
            scored_terms[term] = tf * idf

        sorted_terms = sorted(scored_terms.keys(), key=lambda t: (-scored_terms[t], t))
    else:
        sorted_terms = sorted(cluster_term_weights.keys(), key=lambda t: (-cluster_term_weights[t], t))

    return sorted_terms[:top_k]


def select_representative_documents(
    documents: List[KnowledgeDocument],
    max_docs: int = 4,
) -> List[str]:
    """
    Selects a small representative subset of document IDs for the cluster.
    Preferences:
    1. Higher HDBSCAN membership probability
    2. Richer document type (pull_request, commit with files, commit_file)
    3. Document length / recency
    """
    if not documents:
        return []

    type_scores = {
        "pull_request": 4,
        "commit": 3,
        "commit_file": 2,
        "pr_file": 1,
    }

    def sort_key(doc: KnowledgeDocument) -> Tuple[float, int, int]:
        prob = doc.cluster_probability if doc.cluster_probability is not None else 0.5
        t_score = type_scores.get(doc.document_type, 0)
        text_len = len(doc.text)
        return (prob, t_score, text_len)

    ranked_docs = sorted(documents, key=sort_key, reverse=True)
    return [d.id for d in ranked_docs[:max_docs]]


def generate_candidate_label(
    representative_terms: List[str],
    important_files: List[str],
) -> str:
    """
    Generates an explainable candidate knowledge area label based on evidence.
    CRITICAL RULES:
    - Never hardcodes cluster IDs or mappings (e.g. never check `if cluster_id == 0`).
    - Grounded strictly in extracted representative terms and file evidence.
    - If evidence is empty/insufficient, returns 'Unclassified Engineering Area'.
    """
    if not representative_terms and not important_files:
        return "Unclassified Engineering Area"

    terms_set = set(representative_terms)

    # 1. Match domain concepts against specific multi-term patterns
    for pattern_terms, candidate_label in DOMAIN_CONCEPT_PATTERNS:
        if pattern_terms.issubset(terms_set):
            return candidate_label

    # 2. Match single top-term domain patterns
    top_3_terms_set = set(representative_terms[:3])
    for pattern_terms, candidate_label in DOMAIN_CONCEPT_PATTERNS:
        if len(pattern_terms) == 1 and pattern_terms.issubset(top_3_terms_set):
            return candidate_label

    # 3. Dynamic label synthesis from top terms
    if representative_terms:
        # Take top 1 or 2 most prominent terms
        top_terms = representative_terms[:2]

        # Common abbreviation expansion for readability
        abbrev_map = {
            "auth": "Authentication",
            "db": "Database",
            "msg": "Messaging",
            "cfg": "Configuration",
            "config": "Configuration",
            "repo": "Repository",
            "sync": "Synchronization",
            "async": "Asynchronous Processing",
            "api": "API",
            "rpc": "RPC Services",
            "sdk": "SDK",
            "cli": "CLI Tools",
            "ui": "User Interface",
            "jwt": "JWT Security",
            "sql": "SQL Database",
            "ttl": "Cache Invalidation",
        }

        clean_parts = []
        for t in top_terms:
            if t in abbrev_map:
                clean_parts.append(abbrev_map[t])
            else:
                clean_parts.append(t.capitalize())

        if len(clean_parts) == 1:
            term = clean_parts[0]
            if term.endswith("ing"):
                return term
            return f"{term} Engineering"

        # Two terms: e.g. "Payment", "Retry" -> "Payment & Retry Handling"
        t1, t2 = clean_parts[0], clean_parts[1]
        if t2.endswith("ing"):
            return f"{t1} & {t2}"
        return f"{t1} & {t2} Management"

    # 4. Fallback from important files directory if no terms extracted
    if important_files:
        first_file = important_files[0]
        parts = first_file.split("/")
        if len(parts) > 1:
            dir_name = parts[0].replace("_", " ").replace("-", " ").title()
            return f"{dir_name} Component"

    return "Unclassified Engineering Area"


def extract_evidence_summary(documents: List[KnowledgeDocument]) -> Dict[str, Any]:
    """
    Extracts summary evidence metrics per cluster.
    CRITICAL: Preserves strict separation between PR author (who opened the PR)
    and Commit authors (who authored commits, including PR-contained commits).
    Does NOT calculate developer expertise or risk scores.
    """
    type_counts: Dict[str, int] = {}
    pr_authors: Set[str] = set()
    commit_authors: Set[str] = set()

    for doc in documents:
        d_type = doc.document_type
        type_counts[d_type] = type_counts.get(d_type, 0) + 1
        meta = doc.metadata or {}

        if d_type == "pull_request":
            pr_author = meta.get("author_login")
            if pr_author and pr_author != "Unknown":
                pr_authors.add(pr_author)

            # Contained commits inside this PR
            contained_commits = meta.get("contained_commits")
            if isinstance(contained_commits, list):
                for c in contained_commits:
                    if isinstance(c, dict):
                        c_auth = c.get("author_login") or c.get("author_name") or c.get("author")
                        if c_auth and c_auth != "Unknown":
                            commit_authors.add(c_auth)

        elif d_type in ("commit", "commit_file"):
            c_auth = meta.get("author_login")
            if c_auth and c_auth != "Unknown":
                commit_authors.add(c_auth)

    return {
        "document_type_counts": type_counts,
        "pr_authors": sorted(list(pr_authors)),
        "commit_authors": sorted(list(commit_authors)),
    }


def interpret_clusters(
    repository: str,
    documents: List[KnowledgeDocument],
    labels: np.ndarray,
    probabilities: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Consumes Step 10 clustering outputs (documents, labels, probabilities) and interprets
    each discovered cluster into a candidate knowledge area.

    Handles:
    - Empty datasets (status: no_data)
    - Small / all-noise datasets (status: completed or insufficient_data)
    - HDBSCAN noise (-1): counted as noise_documents, NEVER turned into a knowledge area
    - Full traceability from candidate label -> cluster_id -> representative documents -> all source documents
    """
    if len(documents) != len(labels):
        raise ValueError(
            f"Document count ({len(documents)}) and label count ({len(labels)}) must match."
        )

    # Empty data handling
    if not documents:
        return {
            "status": "no_data",
            "repository": repository,
            "clusters": [],
            "noise_documents": 0,
            "total_documents": 0,
        }

    prob_list = (
        [round(float(p), 4) if p is not None else None for p in probabilities]
        if probabilities is not None
        else [None] * len(labels)
    )

    # Assign cluster_id and cluster_probability to document instances
    clusters_map: Dict[int, List[KnowledgeDocument]] = {}
    noise_docs: List[KnowledgeDocument] = []

    for doc, label, prob in zip(documents, labels, prob_list):
        lbl = int(label)
        doc.cluster_id = lbl
        doc.cluster_probability = prob

        if lbl == -1:
            noise_docs.append(doc)
        else:
            clusters_map.setdefault(lbl, []).append(doc)

    noise_count = len(noise_docs)

    # Sort regular clusters by size descending (largest cluster first)
    regular_cluster_ids = sorted(
        clusters_map.keys(),
        key=lambda c_id: len(clusters_map[c_id]),
        reverse=True,
    )

    # Interpret each regular cluster
    cluster_results: List[Dict[str, Any]] = []
    for c_id in regular_cluster_ids:
        cluster_docs = clusters_map[c_id]

        # 1. Important file paths
        important_files = extract_file_paths(cluster_docs, max_files=8)

        # 2. Representative terms (using all clusters for IDF contrast)
        representative_terms = extract_representative_terms(
            cluster_docs,
            all_cluster_documents=clusters_map,
            top_k=8,
        )

        # 3. Representative documents (prefer higher membership probability)
        representative_docs = select_representative_documents(cluster_docs, max_docs=4)

        # 4. Generate candidate label from evidence
        candidate_label = generate_candidate_label(representative_terms, important_files)

        # 5. Evidence summary (author separation, type breakdown)
        evidence_summary = extract_evidence_summary(cluster_docs)

        # 6. Full traceability
        all_doc_ids = [d.id for d in cluster_docs]

        cluster_results.append({
            "cluster_id": c_id,
            "candidate_label": candidate_label,
            "document_count": len(cluster_docs),
            "representative_documents": representative_docs,
            "representative_terms": representative_terms,
            "important_files": important_files,
            "all_document_ids": all_doc_ids,
            "evidence_summary": evidence_summary,
        })

    return {
        "status": "completed",
        "repository": repository,
        "clusters": cluster_results,
        "noise_documents": noise_count,
        "noise_document_ids": [d.id for d in noise_docs],
        "total_documents": len(documents),
    }


def run_knowledge_interpretation(
    owner: str,
    repo: str,
    min_cluster_size: int = 5,
    min_samples: int = 3,
    metric: str = "euclidean",
    cluster_selection_method: str = "eom",
    supabase_client: Optional[SupabaseClient] = None,
    github_client: Optional[Any] = None,
    pr_commits_map: Optional[Dict[int, List[Dict[str, Any]]]] = None,
    embedding_service: Optional[EmbeddingService] = None,
) -> Dict[str, Any]:
    """
    Executes the complete Step 10 + Step 11 discovery and interpretation pipeline:
    1. Loads GitHub knowledge documents (Supabase + GitHub PR commits)
    2. Generates dense vectors using existing BGE embedding model
    3. Runs HDBSCAN semantic clustering
    4. Interprets discovered clusters into candidate knowledge areas
    """
    owner_clean = owner.strip()
    repo_clean = repo.strip()
    full_name = f"{owner_clean}/{repo_clean}"

    # Step 10: Run clustering
    clustering_result = run_knowledge_clustering(
        owner=owner_clean,
        repo=repo_clean,
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric=metric,
        cluster_selection_method=cluster_selection_method,
        supabase_client=supabase_client,
        github_client=github_client,
        pr_commits_map=pr_commits_map,
        embedding_service=embedding_service,
    )

    status = clustering_result.get("status", "completed")
    if status == "no_data":
        return {
            "status": "no_data",
            "repository": full_name,
            "clusters": [],
            "noise_documents": 0,
            "total_documents": 0,
        }

    # Load reconstructed documents to preserve full text and metadata
    documents = load_knowledge_documents(
        owner=owner_clean,
        repo=repo_clean,
        supabase_client=supabase_client,
        github_client=github_client,
        pr_commits_map=pr_commits_map,
    )

    if not documents:
        return {
            "status": "no_data",
            "repository": full_name,
            "clusters": [],
            "noise_documents": 0,
            "total_documents": 0,
        }

    # Extract labels and probabilities from doc mappings in clustering result
    doc_mapping_dict = {
        m["id"]: (m.get("cluster_id", -1), m.get("cluster_probability"))
        for m in clustering_result.get("documents", [])
    }

    labels_list: List[int] = []
    probs_list: List[Optional[float]] = []
    for d in documents:
        if d.id in doc_mapping_dict:
            c_id, prob = doc_mapping_dict[d.id]
            labels_list.append(c_id)
            probs_list.append(prob)
        else:
            labels_list.append(-1)
            probs_list.append(0.0)

    labels = np.array(labels_list, dtype=int)
    probabilities = np.array(
        [p if p is not None else 0.0 for p in probs_list], dtype=float
    )

    regular_count = len(set(labels) - {-1})
    _log_progress(
        f"[Knowledge Interpretation] Interpreting {regular_count} clusters into candidate knowledge areas..."
    )

    # Step 11: Interpret clusters
    interpreted = interpret_clusters(
        repository=full_name,
        documents=documents,
        labels=labels,
        probabilities=probabilities,
    )
    _log_progress(f"[Knowledge Interpretation] Interpretation complete for '{full_name}'.\n")
    return interpreted


if __name__ == "__main__":
    import argparse
    import json
    import sys
    from dotenv import load_dotenv

    load_dotenv()
    import os
    load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

    parser = argparse.ArgumentParser(
        description="AI Digital Shadow - Run Knowledge Area Interpretation CLI"
    )
    parser.add_argument("owner", help="Repository owner (e.g. sonal-38)")
    parser.add_argument("repo", help="Repository name (e.g. smart_payment_platform)")
    parser.add_argument(
        "--min-cluster-size",
        type=int,
        default=5,
        help="HDBSCAN min_cluster_size (default: 5)",
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=3,
        help="HDBSCAN min_samples (default: 3)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print raw JSON response output",
    )

    args = parser.parse_args()

    try:
        res = run_knowledge_interpretation(
            owner=args.owner,
            repo=args.repo,
            min_cluster_size=args.min_cluster_size,
            min_samples=args.min_samples,
        )
        if args.json:
            print("\n--- INTERPRETATION JSON RESULT ---")
            print(json.dumps(res, indent=2))
        else:
            print("\n================ INTERPRETATION RESULTS ================")
            for cl in res.get("clusters", []):
                print(f"Cluster #{cl['cluster_id']}: '{cl['candidate_label']}' ({cl['document_count']} docs)")
                print(f"  Representative Terms: {', '.join(cl.get('representative_terms', [])[:6])}")
                print(f"  Important Files: {', '.join(cl.get('important_files', [])[:4])}")
            print("========================================================\n")
    except Exception as err:
        print(f"\n[Knowledge Interpretation CLI Error] {err}", file=sys.stderr)
        sys.exit(1)
