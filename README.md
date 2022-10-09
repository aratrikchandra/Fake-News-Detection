# Fake-News-Detection-Using-NLTK
The project is divided in tow steps:
1. Train machine learning model
2. Deploy the model using Flask APP

**Prerequisite:**
First we have to install Flask, nltk, python ,sklearn and all necessary libraires

**Hyperparameters with GridSearchCV :**

Hyperparameters are the variables that the user specify usually while building the Machine Learning model. Thus, hyperparameters are specified before specifying the parameters or we can say that hyperparameters are used to evaluate optimal parameters of the model.

**GridSearchCV**

We know what hyperparameters are,so now our goal should be to find the best hyperparametric values to get the perfect prediction results from our model.  But the question arises, how to find these best sets of hyperparameters? One can try the Manual Search method, by using the hit and trial process and can find the best hyperparameters which would take huge time to build a single model.

For this reason, methods like Random Search, GridSearch were introduced. Here, we will discuss how Grid Seach is performed and how it is executed with cross-validation in GridSearchCV.

Grid Search uses a different combination of all the specified hyperparameters and their values and calculates the performance for each combination and selects the best value for the hyperparameters. This makes the processing time-consuming and expensive based on the number of hyperparameters involved.

**K-Fold Cross Validation :**

K-fold Cross-Validation is an iterative process that divides the train data into k partitions. Each iteration keeps one partition for testing and the remaining k-1 partitions for training the model. The next iteration will set the next partition as test data and the remaining k-1 as train data and so on. In each iteration, it will record the performance of the model and at the end give the average of all the performance.

**How to apply GridSearchCV :**

                                                clf = GridSearchCv(estimator, param_grid, cv, scoring)

Primarily, it takes 4 arguments i.e. estimator, param_grid, cv, and scoring. The description of the arguments is as follows:

1. estimator – A scikit-learn model

2. param_grid – A dictionary with parameter names as keys and lists of parameter values.

3. scoring – The performance measure. For example, ‘r2’ for regression models, ‘precision’ for classification models.

4. cv – An integer that is the number of folds for K-fold cross-validation.

GridSearchCV can be used on several hyperparameters to get the best values for the specified hyperparameters.

**Drawback of GridSearchCV :**

GridSearchCV is a model selection step and this should be done after Data Processing tasks.GridSearchCV will go through all the intermediate combinations of hyperparameters which makes grid search computationally very expensive.

**Data Preprocessing ::**

* Lowercase the text document
* Remove the words counting just one letter
* Remove the words that contain numbers
* Tokenize the text and remove punctuation
* Remove all stop words
* Remove tokens that are empty
* pos tag the text
* Lemmatize the text

**Vectorizing dataset :**

For any text to be fed to a model, the text has to be transformed into numerical values. This process is called vectorizing and will be redone everytime a new feature is added.

**Feature Engineering :**

* Explicit POS tagging
* TF-IDF weighting
* Count Vectorizer

**POS Tagging :**

 It is a process of converting a sentence to forms – list of words, list of tuples (where each tuple is having a form (word, tag)). The tag in case of is a part-of-speech tag, and signifies whether the word is a noun, adjective, verb, and so on.
 Steps Involved in the POS tagging example:
* Tokenize text (word_tokenize)
* Apply pos_tag to above step that is nltk.pos_tag(tokenize_text)

**TF-IDF weighting :**

Term frequency-inverse document frequency is a text vectorizer that transforms the text into a usable vector. It combines 2 concepts, Term Frequency (TF) and Document Frequency (DF).
Steps for TF-IDF Vectorizer :
* Create a term frequency matrix where rows are documents and columns are distinct terms throughout all documents. Count word occurrences in every text
* Compute inverse document frequency (IDF) using the previously explained formula.
* Multiply TF matrix with IDF respectively

**Count Vectorizer :**

Bag of Words: It is a representation of text that describes the occurrence of words within a document. It involves : A. Vocabulary of known words. B. A measure of the presence of known words. It is called a ‘bag’ of words, because any information about the order or structure of words in the document is discarded. The model is only concerned with whether known words occur in the document, not where in the document. CountVectorizer package is used when implementing the Bag of Words model.

Problem with Bag of Words:

CountVectorizer simply gives equal weight to each of the word present in the document. If we have two different documents talking about the same topic but with different lengths, we observe that the average count values in the longer document will be higher than the shorter document. Hence we can avoid this problem by using the TfIDF Vectortizer- it evaluated the frequency of the words than the occurrence of the same word in the document, giving the importance of the words to that particular document.

**Predicting on test data :**

Run Models on preprocessed + pos-tagged + TF-IDF weighted  + Trigram vectorized text (CountVectorizer).Then ar last deploy the model usinf Flask.


