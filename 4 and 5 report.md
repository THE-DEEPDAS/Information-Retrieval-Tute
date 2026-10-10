## What I did
## Assignment 4: Search Engine Ranking and Evaluation

### Methodology

The document collection is tokenized and represented using `TfidfVectorizer`. Documents are ranked using cosine similarity (Vector Space Model) and a log-odds probabilistic/BIM score. The Probabilistic Ranking Principle ranks documents in decreasing order of estimated relevance.

### Test query

`machine learning applications in healthcare`

The top five results are compared using relevance judgments. Precision, recall, F-measure, MRR, MAP, and NDCG are calculated from the ranked results.

### Observation and conclusion

TF-IDF/cosine and probabilistic ranking produced identical evaluation values on this small test collection. The final model choice should be based on MAP and NDCG, not only on the first result.

### Limitations

Results depend on the small collection, manually assigned relevance labels, preprocessing choices, and the selected query. Exact semantic meaning and synonyms are not fully captured by lexical matching.

## Assignment 5: Intelligent Document Search

### Methodology

Twenty documents from AI, cybersecurity, databases, and networking are represented using scikit-learn TF-IDF vectors. `KMeans` groups similar documents, `LatentDirichletAllocation` discovers topic keywords, and `PCA` produces the two-dimensional visualization. Queries are matched using cosine similarity. Query expansion adds related security terms to improve lexical coverage.

### Test query

`documents about protecting computer networks`

The cybersecurity and networking documents should receive the highest scores. Retrieval quality is evaluated with manually prepared relevance judgments using precision, recall, and F1 at rank 8.

### Observation and conclusion

Unsupervised clustering can reveal subject groups without labeled training data. TF-IDF retrieval with query expansion improved the result for the test query. The final result achieved 0.8750 precision and 0.7778 F1 at rank 8.

### Limitations

Cluster quality depends on the number of clusters, document size, preprocessing, vector representation, and random initialization. The small dataset and manually prepared relevance labels limit generalization.

## 4. Evaluation record

Record the values produced by the programs here:

| Assignment | Precision | Recall | F-measure | MRR | MAP | NDCG |
|---|---:|---:|---:|---:|---:|---:|
| 4 — TF-IDF/VSM | 0.6000 | 1.0000 | 0.7500 | 1.0000 | 1.0000 | 1.0000 |
| 4 — Probabilistic/BIM | 0.6000 | 1.0000 | 0.7500 | 1.0000 | 1.0000 | 1.0000 |
| 5 — TF-IDF retrieval @8 | 0.8750 | 0.7000 | 0.7778 | — | — | — |

