# Citation Link Prediction
 
This project was developed for the **NLP 053 - CSE UOI 2025 Kaggle Challenge**.
 
The goal of the project is to predict whether one scientific paper cites another, using information from paper abstracts, authors, and the citation graph.
 
## Kaggle Competition
 
Competition link:  
https://www.kaggle.com/competitions/nlp-cse-uoi-2025
 
## Problem Description
 
The task is formulated as a binary classification / link prediction problem.  
Given a pair of papers `(source, target)`, the model predicts the probability that the source paper cites the target paper.
 
## Dataset
 
The provided dataset consists of:
 
- `abstracts.txt`: paper IDs and their abstracts
- `authors.txt`: paper IDs and their authors
- `edgelist.txt`: existing citation links
- `test.txt`: paper pairs for prediction
 
The dataset files are not included in this repository due to competition restrictions.
 
## Methodology
 
The final solution combines:
 
- Sentence-BERT embeddings for abstract representation
- Graph-based features from the citation network
- Author overlap features
- Negative sampling for non-citation examples
- A feed-forward neural network implemented in PyTorch
 
## Features Used
 
For each paper pair, the model uses:
 
- Embeddings of both papers
- Cosine similarity between abstracts
- Number of common authors
- Number of common graph neighbors
- Product of node degrees
 
## Model
 
The final model is a PyTorch feed-forward neural network trained with binary cross-entropy loss.
 
The best validation log loss achieved was approximately:
0.13999
