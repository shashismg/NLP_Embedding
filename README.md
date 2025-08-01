# Embedding Techniques Explained

This repository explores and demonstrates five popular embedding techniques widely used in Natural Language Processing (NLP):

- **Bag of Words (BoW)**
- **TF-IDF (Term Frequency–Inverse Document Frequency)**
- **Word2Vec**
- **GloVe (Global Vectors for Word Representation)**
- **FastText**

Each technique is implemented and explained with examples to help you understand their principles, use cases, and limitations.

## 📦 1. Bag of Words (BoW)

**Overview**:  
Bag of Words is a simple method to represent text data by converting it into fixed-length vectors. Each feature corresponds to a vocabulary word, and values denote word occurrence.

- **Pro**: Simple, quick, and effective for small tasks.
- **Con**: Ignores context, word order, and semantics.

## 🧮 2. TF-IDF

**Overview**:  
TF-IDF improves BoW by reflecting how important a word is to a document in a collection. It penalizes common words and highlights unique ones.

- **Pro**: Good for text classification, information retrieval.
- **Con**: Still ignores word semantics and context.

## 🤖 3. Word2Vec

**Overview**:  
Word2Vec learns dense, continuous word representations using shallow neural networks (CBOW & Skip-gram models), capturing semantic and syntactic relationships between words.

- **Pro**: Captures word similarity and context.
- **Con**: Out-of-vocabulary words can't be represented after training.

## 🌏 4. GloVe

**Overview**:  
GloVe combines matrix factorization and context-based learning to create word vectors. It's trained on global word-word co-occurrences in a corpus.

- **Pro**: Efficient, strong performance for many NLP tasks.
- **Con**: Also suffers from out-of-vocabulary issues.

## ⚡ 5. FastText

**Overview**:  
FastText extends Word2Vec by representing words as n-grams of characters, so it can generate representations for out-of-vocabulary words by combining subword vectors.

- **Pro**: Handles rare and out-of-vocabulary words better.
- **Con**: Slightly larger models, slower than Word2Vec for inference.

## 🚀 Getting Started

Explore the example notebooks/scripts for each technique. You can run them to see how different methods convert raw text to vectors.

## 📁 Structure

```
.
├── bow/
├── tfidf/
├── word2vec/
├── glove/
├── fasttext/
└── README.md
```

Each folder contains implementation, usage examples, and notes.

## 📝 References

- Mikolov, T., et al. "Efficient Estimation of Word Representations in Vector Space." (Word2Vec)
- Pennington, J., Socher, R., & Manning, C.D. "GloVe: Global Vectors for Word Representation."
- Bojanowski, P., et al. "Enriching Word Vectors with Subword Information." (FastText)

## 📧 Contact

For questions or suggestions, please open an issue or contact the maintainer.

**Happy Embedding!** 🎉
