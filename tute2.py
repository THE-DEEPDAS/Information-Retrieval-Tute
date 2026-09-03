import re
from nltk.stem import PorterStemmer

text = "The students are studying different subjects and they are enjoying their studies. The teachers are teaching the subjects very effectively."

print("Original Text:")
print(text)

tokens = re.findall(r'\b[a-zA-Z]+\b', text.lower())

print("\nTokens:")
print(tokens)

stop_words = {
    "the", "are", "and", "they", "their",
    "is", "am", "a", "an", "in", "on",
    "of", "to", "for", "with", "very"
}

filtered_tokens = [
    word for word in tokens
    if word not in stop_words
]

print("\nText after Stop-word Removal:")
print(filtered_tokens)

stemmer = PorterStemmer()

stemmed_words = [
    stemmer.stem(word)
    for word in filtered_tokens
]

print("\nStemmed Words:")
print(stemmed_words)


# Simple dictionary-based lemmatization for this example
lemma_dict = {
    "students": "student",
    "studying": "study",
    "subjects": "subject",
    "enjoying": "enjoy",
    "studies": "study",
    "teachers": "teacher",
    "teaching": "teach",
    "effectively": "effective"
}

lemmatized_words = [
    lemma_dict.get(word, word)
    for word in filtered_tokens
]

print("\nLemmatized Words:")
print(lemmatized_words)


print("\nComparison between Stemming and Lemmatization:")

print("""
1. Stemming removes prefixes or suffixes from a word to obtain
   a root-like form.

2. Lemmatization converts a word into its meaningful dictionary
   form.

3. Real-world applications:
   - Sentiment Analysis
   - Search Engines
""")