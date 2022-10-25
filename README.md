# Comment classification of C code
This project is a submission of subtask named Information Retreival in Software Engineering [(IRSE)](https://sites.google.com/view/ir-se/home) given by Forum for Information Retrieval Evaluation [(FIRE)](http://fire.irsi.res.in/fire/2022/home) 2022. It aims to present different text mining frameworks and analyze their performance for classification of C codes as useful or non-useful. The frameworks involve various classifiers and feature engineering schemes following bag of words (BOW) model.
Classical machine learning models like random forest, logistic regression and support vector machine and transformer based models like BERT, RoBERT and ALBERT have been explored. 
## Pre-requisites
 [NLTK](https://www.nltk.org/install.html), NumPy, Scikit-Learn, [Torch](https://pypi.org/project/torch/), [Transformers](https://pypi.org/project/transformers/)
## To run the framework
Create a folder named `output` in the main project path to store the outputs of individual test documents. Create another directory saved_models in the main project path to store the trained models, so that they can be repeatedly used.
