"""
Evaluate BOTH:
1. Original embeddings in: models/<pair_id>/*.vec
2. Aligned embeddings in: aligned_models/<pair_id>/*.vec

Outputs all results into evaluation_results.csv
"""

import os
import csv
from gensim.models import KeyedVectors

folders_to_evaluate = [
    ("models", "original"),
    ("aligned_models", "aligned")
]

output_csv = "evaluation_results.csv"
analogy_file = "questions-words.txt"
similarity_file = "wordsim353.tsv"

topk_list = [1, 5, 10]

csv_header = [
    "pair_id",
    "model_name",
    "model_type",   # original / aligned
] + [
    f"analogy_top{k}_accuracy" for k in topk_list
] + [
    "word_similarity_pearson",
    "word_similarity_spearman",
    "word_similarity_oov_ratio",
]


def evaluate_top_k_analogies(model, analogy_file, topk):
    total, correct = 0, 0
    with open(analogy_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith(":"):
                continue
            parts = line.strip().lower().split()
            if len(parts) != 4:
                continue

            a, b, c, expected = parts
            if all(w in model for w in [a, b, c]):
                predicted = model.most_similar(
                    positive=[b, c], negative=[a], topn=topk
                )
                predicted_words = [w for w, _ in predicted]
                if expected in predicted_words:
                    correct += 1
                total += 1

    return correct / total if total > 0 else 0.0


results = []

for folder, model_type in folders_to_evaluate:
    if not os.path.exists(folder):
        print(f"Skipping missing folder: {folder}")
        continue

    print(f"\n=== Evaluating folder: {folder} ({model_type}) ===")

    # iterate over pair folders like 0-1, 2-3
    for pair_id in os.listdir(folder):
        pair_path = os.path.join(folder, pair_id)
        if not os.path.isdir(pair_path):
            continue

        print(f"\nEvaluating pair: {pair_id}")

        # evaluate each .vec inside the pair folder
        for filename in os.listdir(pair_path):
            if not filename.endswith(".vec"):
                continue

            model_path = os.path.join(pair_path, filename)
            print(f"  Loading model: {filename}")

            try:
                model = KeyedVectors.load_word2vec_format(model_path, binary=False)

                # analogy results
                analogy_scores = {
                    k: evaluate_top_k_analogies(model, analogy_file, topk=k)
                    for k in topk_list
                }

                # wordsim evaluation
                pearson, spearman, oov_ratio = model.evaluate_word_pairs(similarity_file)

                # store result
                results.append(
                    [pair_id, filename, model_type]
                    + [round(analogy_scores[k], 4) for k in topk_list]
                    + [
                        round(pearson[0], 4),
                        round(spearman[0], 4),
                        round(oov_ratio, 4),
                    ]
                )

            except Exception as e:
                print(f"  Error evaluating {filename}: {e}")


# write CSV
with open(output_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(csv_header)
    writer.writerows(results)

print(f"\nEvaluation completed. Results written to {output_csv}")
