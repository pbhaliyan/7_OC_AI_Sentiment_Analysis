# :dart: Sentiment Analysis
Perform binary sentiment analysis (positive / negative) with tweets.

We benchmark 3 approaches:
- Turn-Key Solutions with Microsoft Cognitive Services (Sentiment Analysis);
- Low-code Solutions with Microsoft Azure Machine Learning Studio (Designer);
- Advanced and Custom Solutions with Keras / Tensorflow (RNN/LSTM and Word Embeddings)

# :card_index_dividers: Dataset
[Sentiment140 dataset with 1.6 million tweets](https://www.kaggle.com/kazanova/sentiment140)

# :scroll: Tasks
- :heavy_check_mark: Pre-process Tweets;
- :heavy_check_mark: Use Azure Text Analytics - Sentiment Analysis;
- :heavy_check_mark: Use AMLS Designer, Logistic Regression model, and 2 text vectorizations;
- :heavy_check_mark: Use Tensorflow / Keras with RNN / LTSM and word embeddings (from scratch and pre-trained);
- :heavy_check_mark: Compare all approaches and evaluate performance (AUC, Accuracy);
- :heavy_check_mark: Deploy the best model for real-time inferencing and publish endpoint;
- :heavy_check_mark: Write a blog article.

<img src='.\pictures\summary_of_models_performance.png'>

# :computer: Dependencies
NLTK, Spacy, WordCloud, Azure ML/AI SDK, scikit-learn, Azure Portal, Azure Machine Learning Studio, Tensorflow/Keras, Google Colab (with GPU), Tensorboard, pretrained word embeddings (Word2Vec, GloVE, USE)

# :pushpin: References 
- [Text analytics Overview](https://azure.microsoft.com/fr-fr/services/cognitive-services/text-analytics/);
- [Azure Machine Learning Studio](https://azure.microsoft.com/fr-fr/services/machine-learning/#product-overview), with dedicated [Portal](https://ml.azure.com/) and its low-code [Designer](https://azure.microsoft.com/fr-fr/services/machine-learning/designer/#features);
- Tensorflow/Keras : [Recurrent Neural Network](https://www.tensorflow.org/guide/keras/rnn) and [Word embeddings](https://www.tensorflow.org/text/guide/word_embeddings); [Text Classification with RNN](https://www.tensorflow.org/text/tutorials/text_classification_rnn);
- Pre-traind word embeddings: [Word2Vec](https://www.tensorflow.org/tutorials/text/word2vec), [GloVe - Global Vectors for Word Representation](https://nlp.stanford.edu/projects/glove/), [Universal Sentence Encoder](https://tfhub.dev/google/universal-sentence-encoder/4).

# Further readings
- [Neural Machine Translation with attention](https://www.tensorflow.org/text/tutorials/nmt_with_attention);
- [Sentiment Analysis with BERT - *Bidirectional Encoder Representations from Transformers*](https://www.tensorflow.org/text/tutorials/classify_text_with_bert).
