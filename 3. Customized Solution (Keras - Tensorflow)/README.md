Our best model is the **pre-trained GloVe embeddings (*200-dimensions vector*), associated with LSTM (Long-Short Term Memory)** recurrent neural network.

GloVe Tweeter embeddings : http://nlp.stanford.edu/data/glove.twitter.27B.zip

## Learning curves of each model

<img src='/pictures\advanced_models_embeddings_RNNs.png'>

The following websites were very useful to understand the logic behind words embeddings and Tensorflow tools:
- https://www.kaggle.com/bertcarremans/using-word-embeddings-for-sentiment-analysis
- https://www.kaggle.com/tanulsingh077/deep-learning-for-nlp-zero-to-transformers-bert
- https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/text/Tokenizer
- https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/sequence/pad_sequences