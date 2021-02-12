Note: model based on https://github.com/anantzoid/VQA-Keras-Visual-Question-Answering

Download GloVe vectors: http://nlp.stanford.edu/data/glove.6B.zip


1. Perform embedding on encodings made by word2vec or something similar.

    I don't know if we can just grab the encodings from somewhere or we should train word2vec over the dictionary we have.

2. Weigh the misspredictions differently depending on the type of question.

    For example predicting "no" when the answer is "red" is a bigger mistake than predicting "blue".
    How to do this:
    https://github.com/keras-team/keras/issues/2115#issuecomment-490079116

3. K-fold cross validation and predict with an ensemble.

    Make different validation splits, train a model with each one and then predict by using the average prediction across all the models.
