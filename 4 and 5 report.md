# Assignments 4 and 5 — Short Report

## Assignment 4: Search Engine Ranking and Evaluation

### Methodology

The document collection is tokenized and represented using TF-IDF vectors. Documents are ranked using cosine similarity (Vector Space Model) and a log-odds probabilistic/BIM score. The Probabilistic Ranking Principle ranks documents in decreasing order of estimated relevance.

### Test query

`machine learning applications in healthcare`

The top five results are compared using relevance judgments. Precision, recall, F-measure, MRR, MAP, and NDCG are calculated from the ranked results.

### Observation and conclusion

TF-IDF/cosine and probabilistic ranking produced identical evaluation values on this small test collection. The final model choice should be based on MAP and NDCG, not only on the first result.

### Limitations

Results depend on the small collection, manually assigned relevance labels, preprocessing choices, and the selected query. Exact semantic meaning and synonyms are not fully captured by lexical matching.

## Assignment 5: Intelligent Document Search

### Methodology

Twenty documents from AI, cybersecurity, databases, and networking are represented as TF-IDF document embeddings. K-means groups similar documents, keyword-frequency topic summaries describe each cluster, and a PCA-style projection of the two highest-variance dimensions produces a two-dimensional visualization. Documents with low similarity to their cluster centroid are flagged as anomalies. Queries are matched using cosine similarity.

### Test query

`documents about protecting computer networks`

The cybersecurity documents should receive the highest scores. Retrieval quality is evaluated with the manually prepared relevance judgments using precision and recall.

### Observation and conclusion

Unsupervised clustering can reveal subject groups without labeled training data. TF-IDF retrieval can match related documents through shared terms. The best configuration is the one producing coherent clusters, meaningful topics, few false anomalies, and high retrieval precision.

### Limitations

Cluster quality depends on the number of clusters, document size, preprocessing, vector representation, and random initialization. Keyword topics and 2D projections are summaries rather than proof of semantic quality.

## 4. Evaluation record

Record the values produced by the programs here:

| Assignment | Precision | Recall | F-measure | MRR | MAP | NDCG |
|---|---:|---:|---:|---:|---:|---:|
| 4 — TF-IDF/VSM | 0.6000 | 1.0000 | 0.7500 | 1.0000 | 1.0000 | 1.0000 |
| 4 — Probabilistic/BIM | 0.6000 | 1.0000 | 0.7500 | 1.0000 | 1.0000 | 1.0000 |
| 5 — Semantic retrieval @8 | 0.5000 | 0.4000 | — | — | — | — |

