import os
import re
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np

folder = "documents"

all_tokens = []

for filename in os.listdir(folder):
    if filename.endswith(".txt"):

        path = os.path.join(folder, filename)

        with open(path, "r", encoding="utf-8") as file:
            text = file.read()

        text = text.lower()
        # \b is word boundary
        words = re.findall(r"\b[a-z]+\b", text)
        all_tokens.extend(words)

print("TOTAL COLLECTION")
print("Total words:", len(all_tokens))
print("Unique words:", len(set(all_tokens)))
print("PART A - ZIPF'S LAW")

frequency = Counter(all_tokens)
# most_common() sorts the words from highest frequency to lowest frequency.
sorted_words = frequency.most_common()
print("\nTop 20 most frequent words:")
# left-align the rank and word, right-align the frequency
print(f"{'Rank':<10}{'Word':<15}{'Frequency':>10}")

for rank, (word, count) in enumerate(sorted_words[:20], start=1):
    print(
        f"{rank:<10}"
        f"{word:<15}"
        f"{count:>10}"
    )

ranks = np.arange(1, len(sorted_words) + 1)

frequencies = np.array(
    [count for word, count in sorted_words]
)

plt.figure()
plt.plot(ranks, frequencies)
plt.xlabel("Word Rank")
plt.ylabel("Word Frequency")
plt.title("Zipf's Law: Rank vs Word Frequency")
plt.grid()
plt.show()

plt.figure()

plt.loglog(
    ranks,
    frequencies,
    marker="."
)

plt.xlabel("Log(Rank)")
plt.ylabel("Log(Frequency)")
plt.title("Zipf's Law: Log-Log Plot")

plt.grid()

plt.show()


print("PART B - HEAP'S LAW")
# the number of distinct words in a document as a function of the document length
# vocab size = K * B^N

total_words = len(all_tokens)

word_counts = []
vocabulary_sizes = []

for percentage in range(10, 101, 10):
    n = int(total_words * percentage / 100)

    # Take first n words
    current_tokens = all_tokens[:n]
    N = len(current_tokens)
    V = len(set(current_tokens))

    word_counts.append(N)
    vocabulary_sizes.append(V)

    print(
        f"Words = {N:<8}"
        f"Vocabulary = {V}"
    )

plt.figure()

plt.plot(
    word_counts,
    vocabulary_sizes,
    marker="o"
)

plt.xlabel("Total Number of Words (N)")
plt.ylabel("Vocabulary Size (V)")
plt.title("Heap's Law: Total Words vs Vocabulary Size")

plt.grid()

plt.show()
plt.figure()

plt.loglog(
    word_counts,
    vocabulary_sizes,
    marker="o"
)

plt.xlabel("Log(Total Words)")
plt.ylabel("Log(Vocabulary Size)")
plt.title("Heap's Law: Log-Log Plot")

plt.grid()

plt.show()

log_N = np.log(word_counts)
log_V = np.log(vocabulary_sizes)

beta, log_K = np.polyfit(
    log_N,
    log_V,
    1
)


K = np.exp(log_K)


print("HEAP'S LAW PARAMETERS")


print(f"K    = {K:.4f}")
print(f"Beta = {beta:.4f}")

print(
    f"\nEstimated Heap's Law:"
    f"\nV(N) = {K:.4f} * N^{beta:.4f}"
)
