import csv
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA, LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).parent
DATA = ROOT / "data" / "assignment5_documents.csv"
OUT = ROOT / "results"
QUERY = "documents about protecting computer networks"

with DATA.open(encoding="utf8") as f:
    rows = list(csv.DictReader(f))

texts = [r["text"] for r in rows]

# convert documents into tf-idf vectors
tfidf = TfidfVectorizer(stop_words="english")
X = tfidf.fit_transform(texts)

# group documents into four clusters
# 20 times centroid lidho as best answer
labels = KMeans(n_clusters=4, random_state=7, n_init=20).fit_predict(X)

# discover four topics using lda
count = CountVectorizer(stop_words="english")
count_matrix = count.fit_transform(texts)
lda = LatentDirichletAllocation(
    n_components=4, random_state=7, max_iter=100
).fit(count_matrix)

words = count.get_feature_names_out()

# reduce document vectors to two dimensions using pca
projection = PCA(n_components=2, random_state=7).fit_transform(X.toarray())

OUT.mkdir(exist_ok=True)
plt.figure(figsize=(10, 7))

for cluster in range(4):
    ids = [i for i, label in enumerate(labels) if label == cluster]

    plt.scatter(
        projection[ids, 0],
        projection[ids, 1],
        s=90,
        label=f"cluster {cluster + 1}"
    )

    for i in ids:
        plt.annotate(
            rows[i]["id"],
            projection[i],
            xytext=(5, 5),
            textcoords="offset points"
        )

plt.title("pca projection of document clusters")
plt.xlabel("principal component 1")
plt.ylabel("principal component 2")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(OUT / "assignment5_2d_projection.png", dpi=180)
plt.close()

# expand the query with related security terms
query = QUERY + " security firewall encryption cyberattacks cryptography routing protocols"

# rank documents using cosine similarity
qv = tfidf.transform([query])
scores = cosine_similarity(qv, X).ravel()
ranking = sorted(zip(rows, scores), key=lambda x: x[1], reverse=True)

# identify relevant documents
relevant = {
    r["id"] for r in rows
    if r["topic"] in {"Cybersecurity", "Networking"}
}

# calculate precision, recall and f1 for the top eight
retrieved = [r["id"] for r, score in ranking[:8]]
hits = len(set(retrieved) & relevant)

precision = hits / 8
recall = hits / len(relevant)
f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0

# save results to a text file
with (OUT / "assignment5_results.txt").open("w", encoding="utf8") as f:
    f.write(f"documents: {len(rows)} | clusters: 4 | query: {QUERY}\n\n")

    # display cluster members
    f.write("cluster membership:\n")
    for cluster in range(4):
        members = [
            r["id"] for r, label in zip(rows, labels)
            if label == cluster
        ]
        f.write(f"cluster {cluster + 1}: {members}\n")

    # display the top six words of each topic
    f.write("\nlda topic keywords:\n")
    for i, weights in enumerate(lda.components_):
        top_words = [words[j] for j in weights.argsort()[-6:][::-1]]
        f.write(f"topic {i + 1}: {', '.join(top_words)}\n")

    # display the top eight search results
    f.write("\ntop retrieval results:\n")
    for i, (r, score) in enumerate(ranking[:8], 1):
        f.write(f"{i}. {r['id']} ({r['topic']}) score={score:.4f}\n")

    # display evaluation scores
    f.write(
        f"\nevaluation:\n"
        f"precision@8 = {precision:.4f}\n"
        f"recall@8 = {recall:.4f}\n"
        f"f1@8 = {f1:.4f}\n"
    )

# print results
print((OUT / "assignment5_results.txt").read_text(encoding="utf8"))
print(f"2d projection saved to: {OUT / 'assignment5_2d_projection.png'}")
