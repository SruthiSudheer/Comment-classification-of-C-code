# Comment classification of C code
This project is a submission of subtask named Information Retreival in Software Engineering [(IRSE)](https://sites.google.com/view/ir-se/home) given by Forum for Information Retrieval Evaluation [(FIRE)](http://fire.irsi.res.in/fire/2022/home) 2022. It aims to present different text mining frameworks and analyze their performance for classification of C codes as useful or non-useful. The frameworks involve various classifiers and feature engineering schemes following bag of words (BOW) model.
Classical machine learning models like random forest, logistic regression and support vector machine and transformer based models like BERT, RoBERT and ALBERT have been explored. 
## Pre-requisites
NumPy, Scikit-Learn, [NLTK](https://www.nltk.org/install.html), [Torch](https://pypi.org/project/torch/), [Transformers](https://pypi.org/project/transformers/)
## To run the framework
Create a folder named `saved_models` in the main project path during training phase to store the trained models, and thus the models can be reused without training. 
In the `testing_irse.py`

the argument `model` can be 

    'bert' for transformer models

    'entropy' for Entropy based term weighting scheme

    'tfidf' for TF-IDF based term weighting scheme 
and the argument `clf_opt` can be

    'lr' for Logistic Regression 

    'rf' for Random Forest

    'svm' for Support Vector Machine 
    
 The desired number of terms can be selected by `no_of_selected_features`.    
 For running BERT, RoBERT and ALBERT models change the `model_name` in the `irse2022.py` and `model_source` in `testing_irse.py` from [Hugging Face](https://huggingface.co/models?search=bert-base-uncased)
