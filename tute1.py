import spacy
import re
import matplotlib.pyplot as plt
from textblob import TextBlob
from nltk.stem import PorterStemmer
from sklearn.metrics import confusion_matrix,precision_score,recall_score,f1_score

nlp=spacy.load("en_core_web_sm")
stemmer=PorterStemmer()

text=input("Enter a paragraph: ")

clean=re.sub(r'[^a-zA-Z\s]','',text.lower())
print("\nCleaned text:",clean)

doc=nlp(clean)
tokens=[t.text for t in doc]
print("\nTokens:",tokens)

tokens=[t.text for t in doc if not t.is_stop]
print("\nAfter stop word removal:",tokens)

lemmas=[t.lemma_ for t in doc if not t.is_stop]
print("\nAfter lemmatization:",lemmas)

stems=[stemmer.stem(t.text) for t in doc if not t.is_stop]
print("\nAfter stemming:",stems)

score=TextBlob(text).sentiment.polarity
result="Positive" if score>=0 else "Negative"

print("\nSentiment:",result)

# what a confusion matrix in 1 sample so i did this
test=[
    "I love this product",
    "This is amazing",
    "I hate this product",
    "This is horrible"
]

actual=["Positive","Positive","Negative","Negative"]

pred=[]
for x in test:
    s=TextBlob(x).sentiment.polarity
    pred.append("Positive" if s>=0 else "Negative")

cm=confusion_matrix(actual,pred,labels=["Positive","Negative"])

print("\nConfusion Matrix:")
print(cm)

print("\nPrecision:",precision_score(actual,pred,pos_label="Positive"))
print("\nRecall:",recall_score(actual,pred,pos_label="Positive"))
print("\nF1 Score:",f1_score(actual,pred,pos_label="Positive"))

plt.imshow(cm)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.colorbar()
plt.show()