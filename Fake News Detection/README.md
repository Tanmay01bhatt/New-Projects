### Fake News Detection | BERT + GRU Hybrid Classifier
 
> Hybrid NLP architecture combining BERT contextual embeddings with GRU sequence modelling to detect fake news with high accuracy.
 
**Tech Stack:** `Python` `PyTorch` `HuggingFace Transformers` `BERT` `GRU`
 
#### Overview
 
An NLP-based fake news classifier built on a hybrid BERT–GRU architecture, trained and evaluated on the [ISOT Fake News Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset).
 
The model leverages BERT's contextual token embeddings as input to a GRU (Gated Recurrent Unit) sequence model — combining **global semantic understanding** from the transformer with **sequential pattern detection** from the recurrent layer.
 
#### Key Results
 
| Metric | Score |
|--------|-------|
| Test Accuracy | **95.76%** |
| Dataset | ISOT Fake News |

#### Features
 
- **Hybrid Architecture:** BERT embeddings feed directly into a GRU, combining transformer-level semantic richness with recurrent temporal modelling.
- **Differential Learning Rates:** Applied separate learning rates across transformer and recurrent layers to reduce overfitting and improve generalisation on unseen articles.
- **Strong Generalisation:** Achieved 95.76% accuracy on held-out test data, demonstrating robustness beyond the training distribution.
 
---
