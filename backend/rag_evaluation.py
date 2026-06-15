import re
import json
from collections import Counter
from typing import Dict, List, Optional

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.metrics.pairwise import cosine_similarity

from rag_chatbot import ask_chatbot


def preprocess_text(
    text: str,
    remove_stopwords: bool = False,
    custom_stopwords: Optional[set] = None,
) -> List[str]:
    """Lowercase, remove punctuation/digits, and tokenize."""
    text = text.lower().strip()
    text = re.sub(r"[^a-z\s]", " ", text)
    tokens = [token for token in text.split() if token]

    if remove_stopwords:
        stopwords = set(ENGLISH_STOP_WORDS)
        if custom_stopwords:
            stopwords |= custom_stopwords
        tokens = [token for token in tokens if token not in stopwords]

    return tokens


def compute_grounding_score(
    generated_answer: str,
    retrieved_docs: List[str],
    remove_stopwords: bool = False,
) -> Dict[str, float]:
    """
    Grounding Score = common_word_count(answer, context) / total_words_in_answer

    Uses multiset overlap (word frequency aware):
    common_count = sum(min(freq_answer[w], freq_context[w]) for each word)
    """
    if not generated_answer.strip():
        return {
            "grounding_score_percent": 0.0,
            "common_words_count": 0.0,
            "total_answer_words": 0.0,
        }

    combined_context = " ".join(retrieved_docs)

    answer_tokens = preprocess_text(generated_answer, remove_stopwords=remove_stopwords)
    context_tokens = preprocess_text(combined_context, remove_stopwords=remove_stopwords)

    if len(answer_tokens) == 0:
        return {
            "grounding_score_percent": 0.0,
            "common_words_count": 0.0,
            "total_answer_words": 0.0,
        }

    answer_counter = Counter(answer_tokens)
    context_counter = Counter(context_tokens)

    common_count = sum(min(answer_counter[word], context_counter[word]) for word in answer_counter)
    grounding_score = (common_count / len(answer_tokens)) * 100.0

    return {
        "grounding_score_percent": grounding_score,
        "common_words_count": float(common_count),
        "total_answer_words": float(len(answer_tokens)),
    }


def compute_relevance_scores(
    query: str,
    retrieved_docs: List[str],
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> Dict[str, np.ndarray | float]:
    """
    Relevance = cosine_similarity(query_embedding, doc_embedding)
    Returns score per doc and average score.
    """
    if not query.strip() or not retrieved_docs:
        return {
            "relevance_scores": np.array([], dtype=float),
            "average_relevance_score": 0.0,
        }

    model = SentenceTransformer(model_name)

    query_embedding = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    doc_embeddings = model.encode(retrieved_docs, convert_to_numpy=True, normalize_embeddings=True)

    scores = cosine_similarity(query_embedding, doc_embeddings)[0]

    return {
        "relevance_scores": scores,
        "average_relevance_score": float(np.mean(scores)),
    }


def compute_precision_at_k(retrieved_docs: List[str], relevant_flags: List[bool]) -> float:
    """
    Precision@K = relevant_docs / total_docs

    Here, K is len(retrieved_docs), and relevant_flags is a manual True/False
    list aligned to retrieved_docs.
    """
    total_docs = len(retrieved_docs)
    if total_docs == 0:
        return 0.0
    if len(relevant_flags) != total_docs:
        raise ValueError("Length mismatch: relevant_flags must match retrieved_docs.")

    relevant_docs = sum(1 for is_relevant in relevant_flags if is_relevant)
    return relevant_docs / total_docs


def compute_recall_f1_at_k(
    retrieved_docs: List[str],
    relevant_flags: List[bool],
    total_relevant_docs: int,
) -> Dict[str, float]:
    """
    Recall@K = relevant_retrieved_in_top_k / total_relevant_docs
    F1@K = 2 * (Precision@K * Recall@K) / (Precision@K + Recall@K)
    """
    if total_relevant_docs <= 0:
        raise ValueError("total_relevant_docs must be > 0.")

    precision_at_k = compute_precision_at_k(retrieved_docs, relevant_flags)
    relevant_retrieved = sum(1 for is_relevant in relevant_flags if is_relevant)
    recall_at_k = relevant_retrieved / total_relevant_docs

    if precision_at_k + recall_at_k == 0:
        f1_at_k = 0.0
    else:
        f1_at_k = 2 * (precision_at_k * recall_at_k) / (precision_at_k + recall_at_k)

    return {
        "recall_at_k": recall_at_k,
        "f1_at_k": f1_at_k,
    }


def extract_key_terms(
    retrieved_docs: List[str],
    top_k: Optional[int] = 20,
    remove_stopwords: bool = True,
) -> List[str]:
    """
    Extract key terms from retrieved documents using token frequency.
    Returns unique terms sorted by frequency (high to low).
    """
    if not retrieved_docs:
        return []

    all_tokens: List[str] = []
    for doc in retrieved_docs:
        all_tokens.extend(preprocess_text(doc, remove_stopwords=remove_stopwords))

    if not all_tokens:
        return []

    term_counts = Counter(all_tokens)
    ranked_terms = [term for term, _count in term_counts.most_common()]

    if top_k is None:
        return ranked_terms
    return ranked_terms[:top_k]


def compute_keyword_match_score(
    retrieved_docs: List[str],
    generated_answer: str,
    top_k: Optional[int] = 20,
    remove_stopwords: bool = True,
) -> Dict[str, object]:
    """
    keyword_match_score = matched_keywords / total_keywords
    """
    keywords = extract_key_terms(
        retrieved_docs=retrieved_docs,
        top_k=top_k,
        remove_stopwords=remove_stopwords,
    )

    if not keywords:
        return {
            "keywords": [],
            "matched_keywords": [],
            "keyword_match_score": 0.0,
        }

    answer_tokens = set(preprocess_text(generated_answer, remove_stopwords=remove_stopwords))
    matched_keywords = [keyword for keyword in keywords if keyword in answer_tokens]

    score = len(matched_keywords) / len(keywords)
    return {
        "keywords": keywords,
        "matched_keywords": matched_keywords,
        "keyword_match_score": score,
    }


def _response_json_to_text(raw_response: str) -> str:
    """Convert chatbot JSON output into one text blob for embedding similarity."""
    try:
        parsed = json.loads(raw_response)
    except json.JSONDecodeError:
        return raw_response.strip()

    mode = parsed.get("mode", "")
    if mode in {"chat", "invalid"}:
        return str(parsed.get("message", "")).strip()

    if mode == "analysis":
        parts: List[str] = []
        assessment = str(parsed.get("assessment", "")).strip()
        disclaimer = str(parsed.get("disclaimer", "")).strip()
        if assessment:
            parts.append(assessment)

        sections = parsed.get("sections", {})
        if isinstance(sections, dict):
            for section_items in sections.values():
                if isinstance(section_items, list):
                    parts.extend(str(item).strip() for item in section_items if str(item).strip())

        if disclaimer:
            parts.append(disclaimer)
        return " ".join(parts).strip()

    return raw_response.strip()


def compute_output_similarity_for_query(
    query: str,
    num_runs: int = 3,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> Dict[str, object]:
    """
    Run the same query multiple times against chatbot and compute cosine similarity
    between response texts.
    """
    if num_runs < 2:
        raise ValueError("num_runs must be >= 2")

    raw_outputs: List[str] = []
    normalized_texts: List[str] = []

    for _ in range(num_runs):
        raw = ask_chatbot(query)
        raw_outputs.append(raw)
        normalized_texts.append(_response_json_to_text(raw))

    model = SentenceTransformer(model_name)
    output_embeddings = model.encode(normalized_texts, convert_to_numpy=True, normalize_embeddings=True)
    sim_matrix = cosine_similarity(output_embeddings, output_embeddings)

    upper_triangle_indices = np.triu_indices(num_runs, k=1)
    pairwise_scores = sim_matrix[upper_triangle_indices]

    return {
        "query": query,
        "raw_outputs": raw_outputs,
        "normalized_outputs": normalized_texts,
        "similarity_matrix": sim_matrix,
        "pairwise_similarities": pairwise_scores,
        "average_output_similarity": float(np.mean(pairwise_scores)) if pairwise_scores.size else 0.0,
    }


def evaluate_rag_response(
    query: str,
    retrieved_docs: List[str],
    generated_answer: str,
    remove_stopwords: bool = False,
) -> Dict[str, object]:
    grounding = compute_grounding_score(
        generated_answer=generated_answer,
        retrieved_docs=retrieved_docs,
        remove_stopwords=remove_stopwords,
    )
    relevance = compute_relevance_scores(query=query, retrieved_docs=retrieved_docs)

    return {
        "grounding_score_percent": grounding["grounding_score_percent"],
        "common_words_count": grounding["common_words_count"],
        "total_answer_words": grounding["total_answer_words"],
        "relevance_scores": relevance["relevance_scores"],
        "average_relevance_score": relevance["average_relevance_score"],
    }


def print_results(query: str, retrieved_docs: List[str], generated_answer: str, results: Dict[str, object]) -> None:
    print("\n" + "=" * 80)
    print("RAG EVALUATION RESULTS")
    print("=" * 80)
    print(f"Query: {query}")
    print(f"Generated Answer: {generated_answer}")
    print(f"Retrieved Documents: {len(retrieved_docs)}")

    print("\n--- Grounding Score ---")
    print(f"Common words in answer/context : {int(results['common_words_count'])}")
    print(f"Total words in answer          : {int(results['total_answer_words'])}")
    print(f"Grounding Score                : {results['grounding_score_percent']:.2f}%")

    print("\n--- Relevance Scores (Cosine Similarity) ---")
    for index, score in enumerate(results["relevance_scores"], start=1):
        print(f"Doc {index}: {score:.4f}")
    print(f"Average Relevance Score        : {results['average_relevance_score']:.4f}")

    if "precision_at_k" in results:
        print("\n--- Retrieval Precision ---")
        print(f"Precision@K                   : {float(results['precision_at_k']) * 100:.2f}%")

    if "recall_at_k" in results:
        print(f"Recall@K                      : {float(results['recall_at_k']) * 100:.2f}%")

    if "f1_at_k" in results:
        print(f"F1@K                          : {float(results['f1_at_k']) * 100:.2f}%")

    if "keyword_match_score" in results:
        print("\n--- Keyword Match ---")
        print(f"Matched Keywords              : {len(results.get('matched_keywords', []))}")
        print(f"Total Keywords                : {len(results.get('keywords', []))}")
        print(f"Keyword Match Score           : {float(results['keyword_match_score']) * 100:.2f}%")

    print("=" * 80)


if __name__ == "__main__":
    # Example usage
    query = "What are the treatment options for eczema and dry itchy skin?"

    retrieved_docs = [
        "Eczema is a chronic inflammatory skin condition causing dry, itchy, and red patches. Treatment often includes moisturizers, topical corticosteroids, and avoiding triggers.",
        "For dry skin management, regular emollient use and mild cleansers are recommended. Antihistamines may help reduce itching in some cases.",
        "Psoriasis is an autoimmune condition characterized by scaly plaques. Treatments include topical agents, phototherapy, and biologics.",
    ]

    generated_answer = (
        "Eczema treatment usually includes daily moisturizers, topical corticosteroids, "
        "trigger avoidance, and mild cleansers to reduce dryness and itching."
    )

    output = evaluate_rag_response(
        query=query,
        retrieved_docs=retrieved_docs,
        generated_answer=generated_answer,
        remove_stopwords=True,
    )

    # Manual relevance labels aligned with retrieved_docs.
    # Example: first 2 docs are relevant, third is not.
    relevant_flags = [True, True, False]
    output["precision_at_k"] = compute_precision_at_k(retrieved_docs, relevant_flags)
    
    # Total relevant docs in ground truth corpus for this query.
    # Set this from your annotation sheet during experiments.
    total_relevant_docs = 3
    recall_f1 = compute_recall_f1_at_k(retrieved_docs, relevant_flags, total_relevant_docs)
    output["recall_at_k"] = recall_f1["recall_at_k"]
    output["f1_at_k"] = recall_f1["f1_at_k"]

    keyword_metrics = compute_keyword_match_score(
        retrieved_docs=retrieved_docs,
        generated_answer=generated_answer,
        top_k=20,
        remove_stopwords=True,
    )
    output["keywords"] = keyword_metrics["keywords"]
    output["matched_keywords"] = keyword_metrics["matched_keywords"]
    output["keyword_match_score"] = keyword_metrics["keyword_match_score"]

    print_results(query, retrieved_docs, generated_answer, output)

    # Optional raw dict output for logging/paper tables
    print("\nRaw Output Dictionary:")
    print({
        "grounding_score_percent": round(float(output["grounding_score_percent"]), 2),
        "relevance_scores": [round(float(score), 4) for score in output["relevance_scores"]],
        "average_relevance_score": round(float(output["average_relevance_score"]), 4),
        "precision_at_k_percent": round(float(output["precision_at_k"]) * 100, 2),
        "recall_at_k_percent": round(float(output["recall_at_k"]) * 100, 2),
        "f1_at_k_percent": round(float(output["f1_at_k"]) * 100, 2),
        "keyword_match_score_percent": round(float(output["keyword_match_score"]) * 100, 2),
        "matched_keywords": output["matched_keywords"],
        "total_keywords": len(output["keywords"]),
    })

    # Same query repeated 3 times: output consistency via cosine similarity
    repeat_query = "What are the treatment options for eczema and dry itchy skin?"
    repeat_eval = compute_output_similarity_for_query(query=repeat_query, num_runs=3)

    print("\n" + "=" * 80)
    print("REPEATED QUERY OUTPUT SIMILARITY (3 RUNS)")
    print("=" * 80)
    print(f"Query: {repeat_eval['query']}")
    for index, text in enumerate(repeat_eval["normalized_outputs"], start=1):
        print(f"Output {index}: {text[:220]}{'...' if len(text) > 220 else ''}")

    print("\nPairwise Cosine Similarities:")
    pairwise = repeat_eval["pairwise_similarities"]
    for index, score in enumerate(pairwise, start=1):
        print(f"Pair {index}: {float(score):.4f}")

    print("\nSimilarity Matrix:")
    matrix = repeat_eval["similarity_matrix"]
    for row in matrix:
        print("  " + " ".join(f"{float(value):.4f}" for value in row))

    print(f"\nAverage Output Similarity: {float(repeat_eval['average_output_similarity']):.4f}")
    print("=" * 80)
