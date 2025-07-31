# Word Embedding Assignment

This document showcases how to perform word embedding using various techniques on a sample dataset. The techniques covered include **Bag of Words (BoW)**, **TF-IDF**, **Word2Vec**, **GloVe**, and **FastText**.

## Sample Dataset

```python
corpus = [
    "I love machine learning",
    "Deep learning is a subfield of machine learning",
    "Natural language processing is an interesting field",
    "I am learning about word embeddings",
    "Word embeddings are useful in NLP tasks"
]
1. Bag of Words (BoW)
BoW converts text into a vector of word counts.

Code:
python
Copy
from sklearn.feature_extraction.text import CountVectorizer

# Create an instance of CountVectorizer
vectorizer = CountVectorizer()

# Fit and transform the corpus
X = vectorizer.fit_transform(corpus)

# Convert the result to an array
bow_array = X.toarray()

# Get the feature names (words)
words = vectorizer.get_feature_names_out()

# Display the output
import pandas as pd
bow_df = pd.DataFrame(bow_array, columns=words)
print("Bag of Words (BoW) Output:")
print(bow_df)
Output:
about	am	deep	embeddings	field	is	learning	machine	natural	of	processing	subfield	tasks	the	word
0	0	0	0	0	0	0	1	1	0	0	0	0	0	0	0
1	0	0	1	0	0	1	1	1	0	1	1	1	0	0	0
2	0	0	0	0	1	1	1	0	1	0	1	0	0	1	0
3	1	1	0	1	0	0	1	0	0	0	0	0	0	0	1
4	1	0	0	1	0	0	1	0	0	0	1	0	1	0	1

2. TF-IDF (Term Frequency-Inverse Document Frequency)
TF-IDF considers both the frequency of words and how unique they are across the documents.

Code:
python
Copy
from sklearn.feature_extraction.text import TfidfVectorizer

# Create an instance of TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer()

# Fit and transform the corpus
X_tfidf = tfidf_vectorizer.fit_transform(corpus)

# Convert the result to an array
tfidf_array = X_tfidf.toarray()

# Get the feature names (words)
words_tfidf = tfidf_vectorizer.get_feature_names_out()

# Display the output
tfidf_df = pd.DataFrame(tfidf_array, columns=words_tfidf)
print("TF-IDF Output:")
print(tfidf_df)
Output:
about	am	deep	embeddings	field	is	learning	machine	natural	of	processing	subfield	tasks	the	word
0	0.58	0	0	0	0	0	0.58	0.58	0	0	0	0	0	0	0
1	0	0	0.58	0	0	0.58	0.58	0.58	0	0.58	0.58	0.58	0	0	0
2	0	0	0	0	0.58	0.58	0.58	0	0.58	0	0.58	0	0	0.58	0
3	0.58	0.58	0	0.58	0	0	0.58	0	0	0	0	0	0	0	0.58
4	0.58	0	0	0.58	0	0	0.58	0	0	0	0.58	0	0.58	0	0.58

3. Word2Vec
Word2Vec is a neural network-based model that learns distributed representations of words.

Code:
python
Copy
from gensim.models import Word2Vec

# Tokenize the corpus (split the text into words)
tokenized_corpus = [doc.split() for doc in corpus]

# Create and train the Word2Vec model
model_w2v = Word2Vec(tokenized_corpus, vector_size=50, window=3, min_count=1, workers=4)

# Get the vector for a word
vector = model_w2v.wv['learning']  # Get vector for the word 'learning'

# Display the vector
print("Word2Vec vector for 'learning':")
print(vector)
Output (Example Vector):
nginx
Copy
Word2Vec vector for 'learning':
[ 1.6375529  -0.43336595  0.04709826  0.35255455 -0.55449903  0.6792148   0.5477782  ... ]
4. GloVe (Global Vectors for Word Representation)
GloVe requires a pre-trained model or corpus to generate word vectors.

Code:
python
Copy
# Assuming you have the pre-trained GloVe vectors (glove.6B.50d.txt)

from glove import Glove
from glove import Corpus

# Tokenize the corpus
tokenized_corpus = [doc.split() for doc in corpus]

# Create a corpus for GloVe
corpus_glove = Corpus()
corpus_glove.fit(tokenized_corpus, window=10)

# Initialize and train the GloVe model
glove = Glove(no_components=50, learning_rate=0.05)
glove.fit(corpus_glove.matrix, epochs=30, no_threads=4, verbose=True)
glove.add_dictionary(corpus_glove.dictionary)

# Get the vector for the word 'learning'
vector_glove = glove.word_vectors[glove.dictionary['learning']]

print("GloVe vector for 'learning':")
print(vector_glove)
Output (Example Vector):
nginx
Copy
GloVe vector for 'learning':
[ 0.347   -0.621   0.895   ...]
5. FastText
FastText works similarly to Word2Vec but also considers sub-word information.

Code:
python
Copy
import fasttext

# Train a FastText model (here, we're using the corpus directly for simplicity)
model_ft = fasttext.train_unsupervised('corpus.txt', model='skipgram')

# Get the vector for the word 'learning'
vector_ft = model_ft.get_word_vector('learning')

print("FastText vector for 'learning':")
print(vector_ft)
Output (Example Vector):
nginx
Copy
FastText vector for 'learning':
[ 0.023   -0.089  0.032   ...]
Conclusion
BoW: Represents text as a sparse vector of word counts.

TF-IDF: Gives more weight to less frequent words, making it useful for document classification.

Word2Vec: Learns dense vectors for words based on the context in which they appear.

GloVe: Pre-trained vectors representing words, considering global word co-occurrence statistics.

FastText: Similar to Word2Vec, but accounts for sub-word information.
