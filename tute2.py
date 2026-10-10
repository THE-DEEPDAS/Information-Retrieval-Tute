import re
import spacy
from nltk.stem import PorterStemmer

nlp = spacy.load("en_core_web_sm")
stemmer = PorterStemmer()
text = "The students are studying different subjects and they are enjoying their studies. The teachers are teaching the subjects very effectively."

print("Original Text:")
print(text)

clean = re.sub(r'[^a-zA-Z\s]', '', text.lower())

doc = nlp(clean)

tokens = [t.text for t in doc]

print("\nTokens:")
print(tokens)

filtered_tokens = [
    t.text for t in doc
    if not t.is_stop
]

print("\nText after Stop-word Removal:")
print(filtered_tokens)


stemmed_words = [
    stemmer.stem(t.text)
    for t in doc
    if not t.is_stop
]

print("\nStemmed Words:")
print(stemmed_words)

lemmatized_words = [
    t.lemma_
    for t in doc
    if not t.is_stop
]

print("\nLemmatized Words:")
print(lemmatized_words)

print("\nComparison between Stemming and Lemmatization:")

for original, stem, lemma in zip(filtered_tokens, stemmed_words, lemmatized_words):
    if stem != lemma:
        print(original, "-> Stemming:", stem, "| Lemmatization:", lemma)