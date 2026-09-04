"""
Task 5: Autocomplete & Autocorrect Analytics (N-Gram Language Models & Levenshtein Distance)
Track: Data Analytics (Level 2) - OIBSIP
Author: Data Analytics Intern
"""

import os
import re
import time
from collections import Counter, defaultdict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Styling
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['figure.dpi'] = 120
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['axes.labelsize'] = 11

def tokenize_text(text):
    """Clean text, remove punctuation, lowercase, and tokenize."""
    clean = re.sub(r'[^a-zA-Z\s]', ' ', text).lower()
    tokens = clean.split()
    return tokens

def build_ngram_models(tokens):
    """Build Unigram, Bigram, and Trigram frequency models."""
    unigram_counts = Counter(tokens)
    total_tokens = len(tokens)
    
    bigram_counts = defaultdict(Counter)
    trigram_counts = defaultdict(Counter)
    
    for i in range(len(tokens) - 1):
        w1, w2 = tokens[i], tokens[i+1]
        bigram_counts[w1][w2] += 1
        
    for i in range(len(tokens) - 2):
        w1, w2, w3 = tokens[i], tokens[i+1], tokens[i+2]
        trigram_counts[(w1, w2)][w3] += 1
        
    return unigram_counts, bigram_counts, trigram_counts, total_tokens

def autocomplete(prompt, bigram_counts, trigram_counts, unigram_counts, top_k=5):
    """
    Predict top-k next words using Trigram model with Bigram backoff.
    Returns list of (candidate_word, probability).
    """
    tokens = tokenize_text(prompt)
    if not tokens:
        # Return top unigrams
        total = sum(unigram_counts.values())
        return [(w, c / total) for w, c in unigram_counts.most_common(top_k)]
    
    if len(tokens) >= 2:
        context = (tokens[-2], tokens[-1])
        if context in trigram_counts and trigram_counts[context]:
            total_context = sum(trigram_counts[context].values())
            return [(w, c / total_context) for w, c in trigram_counts[context].most_common(top_k)]
            
    # Backoff to bigram
    last_word = tokens[-1]
    if last_word in bigram_counts and bigram_counts[last_word]:
        total_context = sum(bigram_counts[last_word].values())
        return [(w, c / total_context) for w, c in bigram_counts[last_word].most_common(top_k)]
        
    # Backoff to most frequent unigrams
    total = sum(unigram_counts.values())
    return [(w, c / total) for w, c in unigram_counts.most_common(top_k)]

def levenshtein_distance(s1, s2):
    """Compute standard minimum Levenshtein edit distance between s1 and s2."""
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
        
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i-1] == s2[j-1]:
                cost = 0
            else:
                cost = 1
            dp[i][j] = min(
                dp[i-1][j] + 1,      # deletion
                dp[i][j-1] + 1,      # insertion
                dp[i-1][j-1] + cost  # substitution
            )
    return dp[m][n]

def autocorrect(word, vocab_counts, total_vocab_tokens, max_dist=2, top_k=3):
    """
    Noisy Channel Model Autocorrect:
    Score = P(word|candidate) * P(candidate)
    Approximated as: (10^(-edit_dist)) * (count(candidate) / total_vocab_tokens)
    """
    word = word.lower().strip()
    if word in vocab_counts:
        return [(word, 0, 1.0)]
    
    candidates = []
    word_len = len(word)
    
    for candidate, count in vocab_counts.items():
        # Fast length heuristic prune
        if abs(len(candidate) - word_len) > max_dist:
            continue
        dist = levenshtein_distance(word, candidate)
        if dist <= max_dist:
            prior = count / total_vocab_tokens
            likelihood = 10.0 ** (-dist)
            posterior_score = likelihood * prior
            candidates.append((candidate, dist, posterior_score))
            
    # Sort primarily by min edit distance, then by posterior score
    candidates.sort(key=lambda x: (x[1], -x[2]))
    return candidates[:top_k]

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    corpus_path = os.path.join(base_dir, "data", "corpus_text.txt")
    images_dir = os.path.join(base_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    print("=" * 70)
    print("TASK 5 - AUTOCOMPLETE & AUTOCORRECT NLP ANALYTICS")
    print("=" * 70)

    # 1. Load & Tokenize Corpus
    print("\n[1] CORPUS INGESTION & TOKENIZATION...")
    with open(corpus_path, "r", encoding="utf-8") as f:
        corpus_raw = f.read()

    tokens = tokenize_text(corpus_raw)
    unigrams, bigrams, trigrams, total_tokens = build_ngram_models(tokens)
    vocab_size = len(unigrams)

    print(f"- Total Tokens Ingested : {total_tokens:,}")
    print(f"- Unique Vocabulary Size : {vocab_size:,} distinct words")
    print(f"- Unique Bigram Transitions: {sum(len(v) for v in bigrams.values()):,}")
    print(f"- Unique Trigram Contexts  : {sum(len(v) for v in trigrams.values()):,}")

    # Visual 1: Vocabulary Frequency & Zipf's Law
    print("\n[2] GENERATING VOCABULARY DISTRIBUTION & ZIPF'S LAW CHART...")
    top_25 = unigrams.most_common(25)
    words_top, counts_top = zip(*top_25)

    all_counts_sorted = sorted(unigrams.values(), reverse=True)
    ranks = np.arange(1, len(all_counts_sorted) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    fig.patch.set_facecolor('#f8f9fa')

    # Top 25 words bar
    sns.barplot(x=list(counts_top), y=list(words_top), ax=axes[0], palette='Blues_r', hue=list(words_top), legend=False)
    axes[0].set_title("Top 25 Most Frequent Words in Corpus", fontweight='bold', pad=10)
    axes[0].set_xlabel("Frequency (Occurrences)", fontweight='bold')
    axes[0].set_ylabel("Word Token", fontweight='bold')

    # Zipf's law log-log plot
    axes[1].loglog(ranks, all_counts_sorted, marker=".", linestyle="none", color='#2c3e50', alpha=0.6, label='Observed Word Frequencies')
    # Theoretical Zipf line (C / rank)
    c_const = all_counts_sorted[0]
    axes[1].loglog(ranks, c_const / ranks, linestyle="--", color='#e74c3c', linewidth=2, label="Zipf's Theoretical Law (1/r)")
    axes[1].set_title("Word Frequency vs. Rank (Zipf's Law Confirmation)", fontweight='bold', pad=10)
    axes[1].set_xlabel("Log Rank", fontweight='bold')
    axes[1].set_ylabel("Log Frequency", fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, which="both", ls="--", alpha=0.3)

    plt.tight_layout()
    chart1_path = os.path.join(images_dir, "01_vocabulary_frequency_zipf.png")
    plt.savefig(chart1_path, dpi=150)
    plt.close()
    print(f"Saved vocabulary distribution chart to: {chart1_path}")

    # 3. Autocomplete Demonstration & Visualization
    print("\n[3] TESTING AUTOCOMPLETE (NEXT-WORD PREDICTION)...")
    test_prompts = [
        "data",
        "machine",
        "natural",
        "artificial intelligence",
        "operating",
        "clinical"
    ]

    autocomplete_results = []
    print(f"{'Prompt Phrase':<25} | {'Top-3 Predicted Completions (Probability)':<45}")
    print("-" * 75)
    for prompt in test_prompts:
        completions = autocomplete(prompt, bigrams, trigrams, unigrams, top_k=3)
        comp_str = ", ".join([f"{w} ({p*100:.1f}%)" for w, p in completions])
        print(f"{prompt:<25} | {comp_str:<45}")
        for w, p in completions:
            autocomplete_results.append({'Prompt': prompt, 'Completion': w, 'Probability': p})

    # Plot Autocomplete Probabilities
    plt.figure(figsize=(11, 5))
    plt.gcf().patch.set_facecolor('#f8f9fa')
    df_ac = pd.DataFrame(autocomplete_results)
    df_ac['Candidate'] = df_ac['Prompt'] + " -> " + df_ac['Completion']
    sns.barplot(x='Probability', y='Candidate', data=df_ac.head(15), palette='mako', hue='Candidate', legend=False)
    plt.title("N-Gram Next-Word Conditional Probabilities: P(Word_n | Context)", fontsize=13, fontweight='bold', pad=12)
    plt.xlabel("Conditional Probability P(w_n | Context)", fontweight='bold')
    plt.ylabel("Prediction Pair", fontweight='bold')
    plt.tight_layout()
    chart2_path = os.path.join(images_dir, "02_autocomplete_ngram_probabilities.png")
    plt.savefig(chart2_path, dpi=150)
    plt.close()
    print(f"Saved autocomplete probability chart to: {chart2_path}")

    # 4. Autocorrect Evaluation: 20 Misspelled Words Benchmark
    print("\n[4] EVALUATING AUTOCORRECT ON 20-WORD TEST SET...")
    test_words = [
        ("datta", "data"),
        ("machin", "machine"),
        ("learnin", "learning"),
        ("algoritm", "algorithm"),
        ("artifitial", "artificial"),
        ("inteligence", "intelligence"),
        ("systms", "systems"),
        ("computr", "computer"),
        ("langugae", "language"),
        ("modls", "models"),
        ("sciense", "science"),
        ("softwar", "software"),
        ("prosess", "process"),
        ("netwrok", "network"),
        ("healtcare", "healthcare"),
        ("medisin", "medicine"),
        ("cloudd", "cloud"),
        ("finaancial", "financial"),
        ("optimze", "optimize"),
        ("enginer", "engineers")
    ]

    benchmark_records = []
    correct_top1 = 0
    correct_top3 = 0

    print(f"{'Input Typo':<14} | {'Target':<14} | {'Top Suggestion':<16} | {'Dist':<5} | {'Score':<10} | {'Status'}")
    print("-" * 75)

    for typo, target in test_words:
        sugg = autocorrect(typo, unigrams, total_tokens, max_dist=2, top_k=3)
        if sugg:
            top_word, dist, score = sugg[0]
            top_candidates = [s[0] for s in sugg]
        else:
            top_word, dist, score = "None", 99, 0.0
            top_candidates = []

        is_top1 = (top_word == target)
        is_top3 = (target in top_candidates)

        if is_top1:
            correct_top1 += 1
            status = "[MATCH]"
        elif is_top3:
            status = "[IN TOP-3]"
        else:
            status = "[MISMATCH]"

        if is_top3:
            correct_top3 += 1

        print(f"{typo:<14} | {target:<14} | {top_word:<16} | {dist:<5} | {score:<10.4e} | {status}")
        benchmark_records.append({
            'Typo': typo,
            'Target': target,
            'Suggested': top_word,
            'Edit_Distance': dist,
            'Score': score,
            'Top1_Match': is_top1,
            'Top3_Match': is_top3
        })

    acc_top1 = (correct_top1 / len(test_words)) * 100
    acc_top3 = (correct_top3 / len(test_words)) * 100
    print("-" * 75)
    print(f"Top-1 Autocorrect Accuracy: {acc_top1:.1f}% ({correct_top1}/{len(test_words)})")
    print(f"Top-3 Autocorrect Accuracy: {acc_top3:.1f}% ({correct_top3}/{len(test_words)})")

    # Plot Autocorrect Distances & Accuracy
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    fig.patch.set_facecolor('#f8f9fa')
    df_bm = pd.DataFrame(benchmark_records)

    # Edit distance distribution
    sns.countplot(x='Edit_Distance', data=df_bm, ax=axes[0], palette='crest', hue='Edit_Distance', legend=False)
    axes[0].set_title("Edit Distance Distribution of Tested Typos", fontweight='bold')
    axes[0].set_xlabel("Levenshtein Edit Distance (Operations)", fontweight='bold')
    axes[0].set_ylabel("Number of Words", fontweight='bold')

    # Accuracy comparison
    axes[1].bar(['Top-1 Exact Match', 'Top-3 Candidate Match'], [acc_top1, acc_top3], color=['#2980b9', '#27ae60'], edgecolor='black', width=0.5)
    axes[1].set_ylim(0, 110)
    axes[1].set_ylabel("Accuracy Rate (%)", fontweight='bold')
    axes[1].set_title("Autocorrect Retrieval Accuracy Benchmark", fontweight='bold')
    axes[1].text(0, acc_top1 + 2, f"{acc_top1:.1f}%", ha='center', fontweight='bold', fontsize=12)
    axes[1].text(1, acc_top3 + 2, f"{acc_top3:.1f}%", ha='center', fontweight='bold', fontsize=12)

    plt.tight_layout()
    chart3_path = os.path.join(images_dir, "03_autocorrect_benchmark_results.png")
    plt.savefig(chart3_path, dpi=150)
    plt.close()
    print(f"Saved autocorrect benchmark chart to: {chart3_path}")

    # 5. Latency Benchmark Analysis (Autocomplete vs. Autocorrect)
    print("\n[5] INFERENCE LATENCY BENCHMARKING...")
    n_iters = 500

    # Measure Autocomplete Latency
    start_time = time.perf_counter()
    for _ in range(n_iters):
        _ = autocomplete("data", bigrams, trigrams, unigrams, top_k=5)
    time_ac = (time.perf_counter() - start_time) / n_iters * 1000  # ms

    # Measure Autocorrect Latency
    start_time = time.perf_counter()
    for _ in range(n_iters):
        _ = autocorrect("machin", unigrams, total_tokens, max_dist=2, top_k=3)
    time_corr = (time.perf_counter() - start_time) / n_iters * 1000  # ms

    print(f"- N-Gram Autocomplete Average Latency : {time_ac * 1000:.2f} microseconds ({time_ac:.4f} ms)")
    print(f"- Levenshtein Autocorrect Latency    : {time_corr:.3f} ms")

    plt.figure(figsize=(7, 4.5))
    plt.gcf().patch.set_facecolor('#f8f9fa')
    plt.bar(['N-Gram Autocomplete', 'Levenshtein Autocorrect'], [time_ac, time_corr],
            color=['#3498db', '#e67e22'], edgecolor='black', width=0.45)
    plt.ylabel("Inference Latency (Milliseconds - Lower is Better)", fontweight='bold')
    plt.title("NLP Latency Comparison: Autocomplete vs. Autocorrect", fontsize=13, fontweight='bold', pad=10)
    plt.text(0, time_ac + 0.05, f"{time_ac:.4f} ms", ha='center', fontweight='bold')
    plt.text(1, time_corr + 0.05, f"{time_corr:.3f} ms", ha='center', fontweight='bold')
    plt.tight_layout()
    chart4_path = os.path.join(images_dir, "04_latency_comparison_benchmark.png")
    plt.savefig(chart4_path, dpi=150)
    plt.close()
    print(f"Saved latency benchmark chart to: {chart4_path}")

    # 6. Top-k Recall Curve
    print("\n[6] TOP-K RECALL ACCURACY CURVE...")
    k_values = [1, 2, 3, 5]
    recall_scores = []

    for k in k_values:
        matches = 0
        for typo, target in test_words:
            candidates = [c[0] for c in autocorrect(typo, unigrams, total_tokens, max_dist=2, top_k=k)]
            if target in candidates:
                matches += 1
        recall_scores.append((matches / len(test_words)) * 100)

    plt.figure(figsize=(8, 4.5))
    plt.gcf().patch.set_facecolor('#f8f9fa')
    plt.plot(k_values, recall_scores, marker='o', linewidth=2.5, color='#27ae60', markersize=8)
    for k, score in zip(k_values, recall_scores):
        plt.text(k, score + 2, f"{score:.1f}%", ha='center', fontweight='bold')
    plt.title("Autocorrect Recall vs. Number of Suggestions (Top-k)", fontsize=13, fontweight='bold', pad=12)
    plt.xlabel("Number of Returned Suggestions (k)", fontweight='bold')
    plt.ylabel("Recovery Rate / Recall (%)", fontweight='bold')
    plt.xticks(k_values)
    plt.ylim(min(recall_scores) - 10, 110)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    chart5_path = os.path.join(images_dir, "05_top_k_accuracy_curve.png")
    plt.savefig(chart5_path, dpi=150)
    plt.close()
    print(f"Saved top-k accuracy curve to: {chart5_path}")

    # 7. Summary & Limitations
    print("\n" + "=" * 70)
    print("N-GRAM & EDIT DISTANCE vs. MODERN NEURAL ARCHITECTURES")
    print("=" * 70)
    print("""
1. Strengths of Classical N-Gram & Levenshtein Models:
   - Ultra-low latency: sub-millisecond execution suitable for embedded mobile keyboard engines.
   - Zero GPU required: tiny RAM footprint, fully explainable probabilities.
2. Limitations of Statistical N-Grams:
   - Fixed Context Horizon: Markov assumption restricts memory to 2-3 words, completely oblivious to long-range dependencies.
   - Out-of-Vocabulary (OOV) Sparsity: unseen n-grams receive zero probability without smoothing.
3. Modern Neural Advancements (Transformers / RNNs / Byte-Pair Encoding):
   - Dense semantic embeddings (Word2Vec / BERT / GPT) understand synonyms and context.
   - Attention mechanisms capture paragraph-level syntactic context, enabling context-aware autocorrect (e.g., distinguishing 'their' vs. 'there').
    """)

if __name__ == "__main__":
    main()
