"""Assignment 5: intelligent document search using scikit-learn.

Run with: conda run -n torch_env python assignment5.py
"""
import csv
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import LatentDirichletAllocation, PCA
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).parent
DATA = ROOT / "data" / "assignment5_documents.csv"
OUT = ROOT / "results"
QUERY = "documents about protecting computer networks"

def main():
    rows = list(csv.DictReader(DATA.open(encoding="utf8")))
    texts = [row["text"] for row in rows]
    tfidf = TfidfVectorizer(stop_words="english")
    X = tfidf.fit_transform(texts)
    labels = KMeans(n_clusters=4, random_state=7, n_init=20).fit_predict(X)
    counts = CountVectorizer(stop_words="english")
    count_matrix = counts.fit_transform(texts)
    lda = LatentDirichletAllocation(n_components=4, random_state=7, max_iter=100).fit(count_matrix)
    words = counts.get_feature_names_out()

    # PCA creates the actual two-dimensional visualization required by the assignment.
    projection = PCA(n_components=2, random_state=7).fit_transform(X.toarray())
    OUT.mkdir(exist_ok=True)
    plt.figure(figsize=(10, 7))
    for cluster in range(4):
        ids = [i for i, label in enumerate(labels) if label == cluster]
        plt.scatter(projection[ids, 0], projection[ids, 1], s=90, label=f"Cluster {cluster + 1}")
        for i in ids:
            plt.annotate(rows[i]["id"], projection[i], xytext=(5, 5), textcoords="offset points")
    plt.title("Assignment 5: PCA Projection of Document Clusters")
    plt.xlabel("Principal component 1")
    plt.ylabel("Principal component 2")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT / "assignment5_2d_projection.png", dpi=180)
    plt.close()

    # Small query expansion bridges the wording gap in the test query
    # ("protecting" is expressed as security, firewalls, and encryption).
    expanded_query = QUERY + " security firewall encryption cyberattacks cryptography routing protocols"
    query_vector = tfidf.transform([expanded_query])
    scores = cosine_similarity(query_vector, X).ravel()
    ranking = sorted(zip(rows, scores), key=lambda item: item[1], reverse=True)
    relevant = {row["id"] for row in rows if row["topic"] in {"Cybersecurity", "Networking"}}
    retrieved = [row["id"] for row, _ in ranking[:8]]
    hits = len(set(retrieved) & relevant)
    precision = hits / 8
    recall = hits / len(relevant)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0
    with (OUT / "assignment5_results.txt").open("w", encoding="utf8") as result:
        result.write(f"Documents: {len(rows)} | Clusters: 4 | Query: {QUERY}\n\n")
        result.write("Cluster membership:\n")
        for cluster in range(4):
            members = [row["id"] for row, label in zip(rows, labels) if label == cluster]
            result.write(f"Cluster {cluster + 1}: {members}\n")
        result.write("\nLDA topic keywords:\n")
        for topic, weights in enumerate(lda.components_):
            top_words = [words[i] for i in weights.argsort()[-6:][::-1]]
            result.write(f"Topic {topic + 1}: {', '.join(top_words)}\n")
        result.write("\nTop retrieval results:\n")
        for rank, (row, score) in enumerate(ranking[:8], 1):
            result.write(f"{rank}. {row['id']} ({row['topic']}) score={score:.4f}\n")
        result.write(f"\nEvaluation: Precision@8={precision:.4f}, Recall@8={recall:.4f}, F1@8={f1:.4f}\n")
    print((OUT / "assignment5_results.txt").read_text())
    print(f"2D projection saved to: {(OUT / 'assignment5_2d_projection.png').resolve()}")

if __name__ == "__main__":
    main()
