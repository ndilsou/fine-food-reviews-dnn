# Amazon Fine Food Reviews Sentiment Classification
Original data available at:
https://www.kaggle.com/snap/amazon-fine-food-reviews

This repo contains an NLP project centered around the amazon fine food review dataset.
The task in a multi class classification on the score [0-5] given by a reviewer for a product sold on Amazon.

The central part of the project is the main [notebook](./notebooks/fine_food_reviews.ipynb) that contains a simple EDA, parsing of the corpus performed with Spacy and finally various models.

We use a Logistic regression applied on LSA embedded reviews as our baseline.

Word embeddings are obtained by training FastText (Gensim version) on the corpus. FastText is selected for its ability
to deal with oov tokens as the corpus contains a fair amount of spelling mistakes.

The main neural architecture uses the word embeddings concatenated with POS tags embeddings passed through a 1D convolutional layers, before a dense block.

