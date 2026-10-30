import csv
import math
import re
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).parent
DATA = ROOT / "data/assignment4_documents.csv"
OUT = ROOT / "results"
QUERY = "machine learning applications in healthcare"

DOCS = [
    (
        "D1",
        "Machine learning diagnosis",
        "Machine learning helps doctors detect disease from medical images and patient data.",
        1,
    ),
    (
        "D2",
        "Healthcare AI analysis",
        "Artificial intelligence analyzes healthcare records and improves treatment.",
        1,
    ),
    (
        "D3",
        "Computer networks history",
        "History of network protocols packet switching and the Internet.",
        0,
    ),
    (
        "D4",
        "Deep learning disease prediction",
        "Deep neural networks predict disease risk from clinical data.",
        1,
    ),
    (
        "D5",
        "Database management",
        "Relational databases provide SQL queries and structured storage.",
        0,
    ),
    (
        "D6",
        "Machine learning applications",
        "Supervised learning includes classification prediction and recommendation.",
        1,
    ),
    (
        "D7",
        "Healthcare information systems",
        "Systems store patient records and support hospitals.",
        1,
    ),
    (
        "D8",
        "Computer graphics",
        "Rendering geometric transformations and image synthesis.",
        0,
    ),
    (
        "D9",
        "Statistics in medicine",
        "Statistical analysis supports clinical trials diagnosis and healthcare.",
        1,
    ),
    (
        "D10",
        "Operating systems",
        "Processes memory files and scheduling are operating system topics.",
        0,
    ),
]


def tok(s): # count until number or space
    return re.findall(r"[a-z]+", s.lower())


def cosine(a, b):
    dot_product = sum(x * y for x, y in zip(a, b))

    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))

    return dot_product / (norm_a * norm_b) if norm_a and norm_b else 0


def metrics(rank, rel):
    z = [int(x in rel) for x in rank]

    p = sum(z) / len(z)
    r = sum(z) / len(rel)

    f = 2 * p * r / (p + r) if p + r else 0

    # how early the first relevant document appears, then enu reciprocal
    rr = next(
        (1 / (i + 1) for i, x in enumerate(z) if x),
        0,
    )

    # MAP che so when z = 1 average precisions
    ap = (
        sum(
            sum(z[:i + 1]) / (i + 1)
            for i, x in enumerate(z)
            if x
        )
        / len(rel)
    )

    # more importance to higher ranked relevant documents
    dcg = sum(
        x / math.log2(i + 2)
        for i, x in enumerate(z)
    )

    ideal = sum(
        1 / math.log2(i + 2)
        for i in range(min(len(rel), len(z)))
    )

    ndcg = dcg / ideal if ideal else 0

    return p, r, f, rr, ap, ndcg


def main():
    # read documents
    with DATA.open(encoding="utf8") as f:
        rows = list(csv.DictReader(f))

    texts = [r["title"] + " " + r["content"] for r in rows]
    ts = [tok(t) for t in texts]
    q = tok(QUERY)

    # calculate document frequency
    df = Counter(t for doc in ts for t in set(doc))
    terms = sorted(df)

    # calculate tf-idf vectors
    def v(x):
        c = Counter(x)
        return [
            (c[t] / len(x)) * math.log((len(rows) + 1) / (df[t] + 1)) + 1
            for t in terms
        ]

    X = [v(doc) for doc in ts]
    qv = v(q)

    # rank using tf-idf and cosine similarity
    tf = sorted(
        zip(rows, [cosine(x, qv) for x in X]),
        key=lambda x: x[1],
        reverse=True
    )

    # rank using probabilistic/bim scoring
    # weigh each word with query presence and not in document, then sum log odds
    prob = []
    for r, doc in zip(rows, ts):
        score = sum(
            math.log((len(rows) - df[t] + 0.5) / (df[t] + 0.5))
            for t in set(q) if t in set(doc)
        )
        prob.append((r, score))

    probabilistic = sorted(prob, key=lambda x: x[1], reverse=True)

    # find relevant documents
    rel = {r["id"] for r in rows if r["relevant"] == "1"}

    # save results
    OUT.mkdir(exist_ok=True)
    with (OUT / "assignment4_results.txt").open("w", encoding="utf8") as f:
        f.write("Query: " + QUERY + "\n\n")

        for name, results in [("TF-IDF/VSM", tf), ("Probabilistic/BIM", probabilistic)]:
            f.write(name + " top five:\n")
            for i, (r, score) in enumerate(results[:5]):
                f.write(f'{i+1}. {r["id"]} {r["title"]} score={score:.4f}\n')
            f.write("\n")

        f.write("Evaluation (P R F MRR MAP NDCG):\n")
        for name, results in [("TF-IDF", tf), ("Probabilistic", probabilistic)]:
            scores = metrics([r["id"] for r, _ in results], rel)
            f.write(name + " " + " ".join(f"{x:.4f}" for x in scores) + "\n")

    print((OUT / "assignment4_results.txt").read_text(encoding="utf8"))


if __name__ == "__main__":
    main()
