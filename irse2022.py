# -*- coding: utf-8 -*-
"""
Created on Mon Jul 25 15:45:09 2022

@author: Sruthi
"""

import csv,json,os,re,sys
import nltk
import numpy as np
import pandas as pd
import statistics
import torch
from sklearn import preprocessing
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import GridSearchCV 
from sklearn.pipeline import Pipeline
import joblib
from sklearn import svm 
from sklearn.linear_model import LogisticRegression 
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import SelectKBest,chi2,mutual_info_classif
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold, train_test_split 
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics import classification_report
from gensim.models import LogEntropyModel
from gensim.corpora import Dictionary
from gensim.models.doc2vec import Doc2Vec,TaggedDocument 
from collections import Counter
from transformers import AutoModelForSequenceClassification,BertTokenizerFast,BertForSequenceClassification,Trainer,TrainingArguments,AutoTokenizer,AutoModel,RobertaTokenizer, RobertaModel

from transformers import PLBartTokenizer, PLBartForSequenceClassification
from torch.utils.data import DataLoader

working_dir = os.getcwd()
checkpoint_dir = "saved_models/transformer_checkpoints"
if not os.path.exists(os.path.join(os.getcwd(),checkpoint_dir)):
    os.mkdir(checkpoint_dir)

# Class for Torch Model
class get_torch_data_format(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor([self.labels[idx]])
        return item

    def __len__(self):
        return len(self.labels)
    
class get_validation_data_format(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings
        self.device = torch.device('cuda')

    def __getitem__(self, idx):
        item = {k: v[idx].clone().detach().to(device=self.device) for k, v in self.encodings.items()}
        return item

    def __len__(self):
        return len(self.encodings['input_ids'])


class comment_classif():
    def __init__(self,path='/home/interns/irse2022/',model='entropy',model_source='monologg/biobert_v1.1_pubmed',model_path='saved_models/entropy_svm/',vec_len=20,clf_opt='s',no_of_selected_features=None,threshold=0.5):
        self.path = path
        self.model = model
        self.model_source=model_source
        self.model_path=model_path
        self.vec_len=int(vec_len)        
        self.clf_opt=clf_opt
        self.no_of_selected_features=no_of_selected_features
        if self.no_of_selected_features!=None:
            self.no_of_selected_features=int(self.no_of_selected_features) 
        self.threshold=float(threshold)
     
    def comment_processing(self, text):
        text = re.sub(r'[^!"#$%&\'()*+,-./:;<=>?@[\]^`{|}~\n\w]+',' ', text)     # Remove special characters e.g., emoticons-😄. The ^ in the beginning ensures that all the natural characters will be kept. 
        text = re.sub(r'[^a-zA-Z?<=>!\n]', ' ', text)                          # Remove special characters
        text=re.sub(r'[?]', '\.', text)                                           # Replace '?' with '.' to properly identify floating point numbers 
        text=re.sub(r'([a-zA-Z0-9])([\),.!?;-]+)([a-zA-Z])', r'\1\2 \3', text )    # Space between delimmiter and letter   
        text=re.sub(r'([a-z])([\.])([\s]*)([a-z])', r'\1 \3\4', text)              # Reomove '.' between two lowercase letters e.g., et al. xxx
        text=re.sub(r'([0-9]+)([\.]+)([0-9]+)([\.]+)([0-9]+)', r'\1-\3-\5', text)  # Reomove '.' between three decimal numbers e.g., et 000.55.66
        text=re.sub(r'([a-z])([\.]*)([0-9])', r'\1\2 \3', text)                    # Space between letter and no.    
        text=re.sub(r'(\s)([a-z0-9]+)([A-Z])([\w]+)', r'\1\2. \3\4', text)         # Put a '.' after a lowercase letter/number followed by Uppercase e.g., drains removed by day threeHe continued to 
        text=re.sub(r'([a-z0-9])([\n]+)([A-Z])', r'\1\. \3', text)                 # Put a between lowercase letter/number, \n and uppercase letter e.g., xxx5 \n Yyy
        text=re.sub(r'(\.)([\s]*)([\.]+)', r'\1', text)                            # Removing extra '.'s, if any 
        text=re.sub(r'[^a-zA-Z?:!\n]' ,'', text, count=4, flags= re.M)
        return text
                  
 #uncomment if both the code and the comments are required as the train data    
    def train_data(self):      
        data = pd.read_csv('irse_2022_training_data.csv')
        
        # label_encoder 
        label_encoder = preprocessing.LabelEncoder()
        data['Class']= label_encoder.fit_transform(data['Class'])
        data['Class'].unique()
        
        #removing the comments from the codes
        #data['Code_only'] = data.apply(lambda row : row['Surrounding Code Context'].replace(str(row['Comments']), ''), axis=1)
        trn_cat = data['Class'].values.tolist()
        trn_data = data['Surrounding Code Context'].values.tolist()
        return trn_data, trn_cat 
 
##uncomment if 'comment only' train data is required    
#    def train_data(self):                             
#        data = pd.read_csv('irse_2022_training_data.csv')
#        
#        # label_encoder 
#        label_encoder = preprocessing.LabelEncoder()
#        data['Class']= label_encoder.fit_transform(data['Class'])
#        data['Class'].unique()
#        
#        #Converting pandas dataframe to list
#        comments = data['Comments'].values.tolist()
#        trn_cat = data['Class'].values.tolist()
#
#        
#        count = 0
#        trn_data = []
#        for i in range(len(comments)):    
#            c = self.comment_processing(comments[i])
#            trn_data.append(c)
#            count+=1
#        return trn_data, trn_cat 
                 
    
    def classification_pipeline(self):    
        # AdaBoost 
        if self.clf_opt=='ab':
            print('\n\t### Training AdaBoost Classifier ### \n')
            be1 = svm.SVC(kernel='linear', class_weight='balanced',probability=True)              
            be2 = LogisticRegression(solver='liblinear',class_weight='balanced') 
            be3 = DecisionTreeClassifier(max_depth=50)
#            clf = AdaBoostClassifier(algorithm='SAMME',n_estimators=100)            
            clf = AdaBoostClassifier(algorithm='SAMME.R',n_estimators=100)
            clf_parameters = {
            'clf__base_estimator':(be1,be2,be3),
            'clf__random_state':(0,10),
            }          
        # Logistic Regression 
        elif self.clf_opt=='lr':
            print('\n\t### Training Logistic Regression Classifier ### \n')
            ext2 = 'logistic_regression'
            clf = LogisticRegression(solver='liblinear',class_weight='balanced') 
            clf_parameters = {
            'clf__random_state':(0,10),
            } 
        # Linear SVC 
        elif self.clf_opt=='ls':   
            print('\n\t### Training Linear SVC Classifier ### \n')
            clf = svm.LinearSVC(class_weight='balanced')  
            clf_parameters = {
            'clf__C':(0.1,1,2,10,50,100),
            }         
        # Multinomial Naive Bayes
        elif self.clf_opt=='nb':
            print('\n\t### Training Multinomial Naive Bayes Classifier ### \n')
            clf = MultinomialNB(fit_prior=True, class_prior=None)  
            clf_parameters = {
            'clf__alpha':(0,1),
            }            
        # Random Forest 
        elif self.clf_opt=='rf':
            print('\n\t ### Training Random Forest Classifier ### \n')
            ext2='random_forest'
            clf = RandomForestClassifier(max_features=None,class_weight='balanced')
            clf_parameters = {
            'clf__criterion':('entropy','gini'),       
            'clf__n_estimators':(30,50,100),
            'clf__max_depth':(10,20,30,50,100,200),
            }          
        # Support Vector Machine  
        elif self.clf_opt=='svm': 
            print('\n\t### Training Linear SVM Classifier ### \n')
            ext2='svm'
            clf = svm.SVC(kernel='linear', class_weight='balanced',probability=True)  
            clf_parameters = {
            'clf__C':(0.1,1,5,10,50,100),
            }
        else:
            print('Select a valid classifier \n')
            sys.exit(0)        
        return clf,clf_parameters,ext2        
    
# TFIDF model    
    def tfidf_training_model(self,trn_data,trn_cat):
        print('\n ***** Building TFIDF Based Training Model ***** \n')         
        clf,clf_parameters,ext2=self.classification_pipeline() 
        if self.no_of_selected_features==None:                                  # To use all the terms of the vocabulary
            pipeline = Pipeline([
                ('vect', CountVectorizer(token_pattern=r'\b\w+\b')),
                ('tfidf', TfidfTransformer(use_idf=True,smooth_idf=True)),     
                ('clf', clf),]) 
        else:
            try:                                        # To use selected terms of the vocabulary
                print('No of Selected Terms \t'+str(self.no_of_selected_features)) 
                pipeline = Pipeline([
                    ('vect', CountVectorizer(token_pattern=r'\b\w+\b')),
                    ('tfidf', TfidfTransformer(use_idf=True,smooth_idf=True)),
                    ('feature_selection', SelectKBest(chi2, k=self.no_of_selected_features)),                         # k=1000 is recommended 
                #    ('feature_selection', SelectKBest(mutual_info_classif, k=self.no_of_selected_features)),        
                    ('clf', clf),]) 
            except:                                  # If the input is wrong
                print('Wrong Input. Enter number of terms correctly. \n')
                sys.exit()
    # Fix the values of the parameters using Grid Search and cross validation on the training samples 
        feature_parameters = {
        'vect__min_df': (2,3),
        'vect__ngram_range': ((1, 2),(1,3)),  # Unigrams, Bigrams or Trigrams
        }
        parameters={**feature_parameters,**clf_parameters} 
        grid = GridSearchCV(pipeline,parameters,scoring='f1_micro',cv=10)          
        grid.fit(trn_data,trn_cat)     
        clf= grid.best_estimator_  
#        print(clf)     
        flname=self.path+self.model_path+self.model+'_'+self.clf_opt+'_'+str(self.no_of_selected_features)
        joblib.dump(clf, flname+'_clf.joblib') 
        return clf,ext2

# Doc2Vec model    
    def doc2vec_training_model(self,trn_data,trn_cat):
        print('\n ***** Building Doc2Vec Based Training Model ***** \n')
        print('No of Features \t'+str(self.no_of_selected_features)) 
        tagged_data = [TaggedDocument(words=nltk.word_tokenize(_d.lower()), tags=[str(i)]) for i, _d in enumerate(trn_data)]
        max_epochs = 10       
        trn_model = Doc2Vec(vector_size=self.no_of_selected_features,alpha=0.025,min_alpha=0.00025,min_count=1,dm =1)
        trn_model.build_vocab(tagged_data)  
        print('Number of Training Samples {0}'.format(trn_model.corpus_count))   
        for epoch in range(max_epochs):
            print('Doc2Vec Iteration {0}'.format(epoch))
            trn_model.train(tagged_data,
                       total_examples=trn_model.corpus_count,
                       epochs=100) 
           # decrease the learning rate
            trn_model.alpha -= 0.0002
        trn_vec=[]
        for i in range(0,len(trn_data)):
            vec=[] 
            for v in trn_model.docvecs[i]:
                  vec.append(v)
            trn_vec.append(vec)
    # Classificiation and feature selection pipelines
        clf,clf_parameters,ext2=self.classification_pipeline() 
        pipeline = Pipeline([('clf', clf),])       
        grid = GridSearchCV(pipeline,clf_parameters,scoring='f1_micro',cv=10) 
        grid.fit(trn_vec,trn_cat)     
        clf= grid.best_estimator_
        print(clf)  
        flname=self.path+self.model_path+self.model+'_'+self.clf_opt+'_'+str(self.no_of_selected_features)
        joblib.dump(clf, flname+'_clf.joblib')
        joblib.dump(trn_model, flname+'_model.joblib')
                
        return clf,ext2,trn_model
     
# LogEntropy model    
    def entropy_training_model(self,trn_data,trn_cat): 
        print('\n ***** Building Entropy Based Training Model ***** \n')
        print('No of Selected Terms \t'+str(self.no_of_selected_features)) 
        trn_vec=[]; trn_docs=[]; 
        for doc in trn_data:
            doc=nltk.word_tokenize(doc.lower())
            trn_docs.append(doc)                       # Training docs broken into words
        trn_dct = Dictionary(trn_docs)
        corpus = [trn_dct.doc2bow(row) for row in trn_docs]
        trn_model = LogEntropyModel(corpus)
        no_of_terms=len(trn_dct.keys())
        print('\n Number of Terms in the Vocabulary\t'+str(no_of_terms)+'\n')
        for item in corpus:
            vec=[0]*no_of_terms                                 # Empty vector of terms for a document
            vector = trn_model[item]                            # LogEntropy Vectors
            for elm in vector:
                vec[elm[0]]=elm[1]
            trn_vec.append(vec)
    # Classificiation and feature selection pipelines
        clf,clf_parameters,ext2=self.classification_pipeline() 
        if self.no_of_selected_features==None:                                  # To use all the terms of the vocabulary
            pipeline = Pipeline([('clf', clf),])    
        else:
            try: 
                pipeline = Pipeline([ ('feature_selection', SelectKBest(chi2, k=self.no_of_selected_features)), 
                                     #('feature_selection', SelectKBest(mutual_info_classif, k=self.no_of_selected_features)),  
                    ('clf', clf),])  
            except:                                  # If the input is wrong
                print('Wrong Input. Enter number of terms correctly. \n')
                sys.exit()
        grid = GridSearchCV(pipeline,clf_parameters,scoring='f1_micro',cv=10) 
        grid.fit(trn_vec,trn_cat)     
        clf= grid.best_estimator_
#        print(clf)
        flname=self.path+self.model_path+self.model+'_'+self.clf_opt+'_'+str(self.no_of_selected_features)
        joblib.dump(clf, flname+'_clf.joblib')
        joblib.dump(trn_model, flname+'_model.joblib')
        joblib.dump(trn_dct, flname+'_dict.joblib')
        
        return clf,ext2,trn_dct,trn_model
# BERT model accuracy function
    def compute_metrics(self,pred):
         labels = pred.label_ids
         preds = pred.predictions.argmax(-1)
         acc = accuracy_score(labels, preds)
         return {
             'accuracy': acc,
         }     
    

# BERT model     
    def bert_training_model(self,trn_data,trn_cat,test_size=0.1,max_length=432,model_name='albert-base-v2'): 
        print('\n ***** Running BERT Model ***** \n')       
        
        #Decouple Model name
        
        checkpoint_path = "TransformerResults/{}/checkpoint-1500".format('_'.join(model_name.split('/')))
        

        tokenizer = AutoTokenizer.from_pretrained(model_name, do_lower_case=True) 
        labels=np.asarray(trn_cat)     # Class labels in nparray format     
        (train_texts, valid_texts, train_labels, valid_labels)= train_test_split(trn_data, labels, test_size=test_size)
        train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=max_length)
        valid_encodings = tokenizer(valid_texts, truncation=True, padding=True, max_length=max_length)
        train_dataset = get_torch_data_format(train_encodings, train_labels)
        valid_dataset = get_torch_data_format(valid_encodings, valid_labels)
        if os.path.exists(checkpoint_path):
            print('Checkpoint Exists : Starting from the last used one')
            model = AutoModelForSequenceClassification.from_pretrained(checkpoint_path, num_labels=2)
        else:
            print('Checkpoint Does Not exists!')
            model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
        training_args = TrainingArguments(
            output_dir='saved_models/transformer_checkpoints/{}_model'.format('_'.join(model_name.split('/'))),          # output directory
            num_train_epochs=18,              # total number of training epochs
            per_device_train_batch_size=4,  # batch size per device during training
            per_device_eval_batch_size=4,   # batch size for evaluation
            warmup_steps=500,                # number of warmup steps for learning rate scheduler
            weight_decay=0.01,               # strength of weight decay
            logging_dir='./transformerLogs',            # directory for storing logs
            load_best_model_at_end=True,     # load the best model when finished training (default metric is loss)
            logging_steps=100,               # log & save weights each logging_steps
            evaluation_strategy="steps",     # evaluate each `logging_steps`
            )    
        trainer = Trainer(
            model=model,                         # the instantiated Transformers model to be trained
            args=training_args,                  # training arguments, defined above
            train_dataset=train_dataset,         # training dataset
            eval_dataset=valid_dataset,          # evaluation dataset
            compute_metrics=self.compute_metrics,     # the callback that computes metrics of interest
            )
        print('\n Trainer done \n')
        # if os.path.exists(checkpoint_path):
        #     print('Resuming from Checkpoint !')
        #     trainer.train(resume_from_checkpoint = True)
        # else:
        #     trainer.train()
        print('\n Trainer train done \n')        
        print('\n save model \n')

        parts = model_name.split('/')
        model_name = '_'.join(parts)    
        model_path = os.path.join("saved_models/transformer_checkpoints","{}_model".format(model_name))
        
        if not os.path.exists(model_path):
            os.mkdir(model_path)
        os.chdir(model_path)
        model.save_pretrained(model_path)
        tokenizer.save_pretrained(model_path)
        os.chdir(working_dir)
        class_names = [0,1]
        return model,tokenizer,class_names

# Classification of Documents
    def classification(self,trn_data,trn_cat,tst_data):  
        tst_vec=[]; tst_docs=[]             
        predicted=[0 for i in range(0,len(tst_data))]
        if self.model=='tfidf':
            #clf,ext2=self.tfidf_training_model(trn_data,trn_cat)
            flname=self.path+self.model_path+self.model+'_'+self.clf_opt+'_'+str(self.no_of_selected_features)
            clf=joblib.load(flname+'_clf.joblib')
            predicted = clf.predict(tst_data)
            predicted_probability = clf.predict_proba(tst_data)
        elif self.model=='entropy':
            clf,ext2,trn_dct,trn_model=self.entropy_training_model(trn_data,trn_cat)
            flname=self.path+self.model_path+self.model+'_'+self.clf_opt+'_'+str(self.no_of_selected_features)
            clf=joblib.load(flname+'_clf.joblib')
            trn_dct=joblib.load(flname+'_dict.joblib')
            trn_model=joblib.load(flname+'_model.joblib')
            for doc in tst_data:
                doc=nltk.word_tokenize(doc.lower()) 
                tst_docs.append(doc)                                
            corpus = [trn_dct.doc2bow(row) for row in tst_docs]     
            no_of_terms=len(trn_dct.keys())
            for itm in corpus:
                vec=[0]*no_of_terms                          # Empty vector of terms for a document
                vector = trn_model[itm]                      # Entropy Vectors 
                for elm in vector:
                       vec[elm[0]]=elm[1]
                tst_vec.append(vec) 
            predicted = clf.predict(tst_vec)
            predicted_probability = clf.predict_proba(tst_vec)
        elif self.model=='doc2vec':
            clf,ext2,trn_model=self.doc2vec_training_model(trn_data,trn_cat)
            flname=self.path+self.model_path+self.model+'_'+self.clf_opt+'_'+str(self.no_of_selected_features)
            clf=joblib.load(flname+'_clf.joblib')
            trn_model=joblib.load(flname+'_model.joblib')
            for doc in tst_data:
                doc=nltk.word_tokenize(doc.lower())
                inf_vec = trn_model.infer_vector(doc,epochs=100)
                tst_vec.append(inf_vec)
            predicted = clf.predict(tst_vec)     
            predicted_probability = clf.predict_proba(tst_vec) 
        elif self.model=='bert':                            # A given BERT model from Higgingface. Default is BioBERT.
            trn_model,trn_tokenizer,class_names=self.bert_training_model(trn_data,trn_cat) 
            predicted=[]; predicted_probability=[]
            for doc in tst_data:
                val_encodings = trn_tokenizer(doc, padding=True, truncation=True, max_length=432, return_tensors="pt") 
                
                valid_dataset = get_validation_data_format(val_encodings)
                val_loader = DataLoader(valid_dataset,batch_size=1)
                
                with torch.no_grad():
                    for batch in val_loader:
                        outputs = trn_model(**batch)
                        probs = outputs[0].softmax(1)
                        cl=class_names[probs.argmax()]
                        predicted.append(cl)      
                        predicted_probability.append(probs) 
 
        else:
            print('Error!!! Please select a valid model \n')
            sys.exit(0)            
        return predicted, predicted_probability   
        
# Main function   
    def comment_classif(self):
        print('\n ***** Getting Training Data ***** \n')          
        trn_data,trn_cat=self.train_data() 
        print(len(trn_data)) 

# Experiments using training data only during training phase (dividing it into training and validation set)
        skf = StratifiedKFold(n_splits=10)
        predicted_class_labels=[]; actual_class_labels=[]; 
        count=0; probs=[];
        for train_index, test_index in skf.split(trn_data,trn_cat):
            X_train=[]; y_train=[]; X_test=[]; y_test=[]
            for item in train_index:
                X_train.append(trn_data[item])
                y_train.append(trn_cat[item])
            for item in test_index:
                X_test.append(trn_data[item])
                y_test.append(trn_cat[item])
            count+=1                
            print('Training Phase '+str(count))
            predicted,predicted_probability=self.classification(X_train,y_train,X_test) 
            if self.model == 'bert':
                for item in predicted_probability:
                    probs.append(float(max(item[0])))
            else:
                for item in predicted_probability:
                    probs.append(float(max(item)))
            for item in y_test:
                actual_class_labels.append(item)
            for item in predicted:
                predicted_class_labels.append(item)           
        confidence_score=statistics.mean(probs)-statistics.variance(probs)
        confidence_score=round(confidence_score, 3)
        print ('\n The Probablity of Confidence of the Classifier: \t'+str(confidence_score)+'\n')        

    # Evaluation
        print(' *************** Confusion Matrix ***************  \n')
        print (confusion_matrix(actual_class_labels, predicted_class_labels))
        cm=confusion_matrix(actual_class_labels, predicted_class_labels)
        TP = cm[1,1] 
        TN = cm[0,0] 
        FP = cm[0,1] 
        FN = cm[1,0] 

        re=TP/(TP+FN)
        pr=TP/(TP+FP)
        print ('\n Recall:'+str(re))
        print ('\n Precision:'+str(pr))
        print ('\n Specificity:'+str(TN/(TN+FP))) 
        print ('\n Fmeasure:'+str(2*pr*re/(pr+re)))
        acc = accuracy_score(actual_class_labels, predicted_class_labels)
        print('\n Accuracy: '+str(acc))
       
        class_names=list(Counter(actual_class_labels).keys())
        class_names = [str(x) for x in class_names]
        print('\n ***************  Scores  *************** \n ')
        print(classification_report(actual_class_labels, predicted_class_labels, target_names=class_names))        

# Experiments on Given Test Data during Test Phase
        if confidence_score>0.55:
            print('\n ***** Getting Test Data ***** \n')  
            test_data = pd.read_csv('irse_2022_test_data.csv')
            label_encoder = preprocessing.LabelEncoder()
            test_data['Class']= label_encoder.fit_transform(test_data['Class'])
            actual_class_labels=test_data['Class'].values.tolist()
            tst_data = test_data['Surrounding Code Context'].values.tolist()
              
            if tst_data==[]:
                print('There is no test data in the directory \n')
            else:
                print('\n ***** Classifying Test Data ***** \n')   
                predicted_class_labels=[];
                predicted_class_labels,predicted_probability=self.classification(trn_data,trn_cat,tst_data)
                print(predicted_class_labels)
                print(len(predicted_class_labels))
                print(len(actual_class_labels))
                
                print(' *************** Confusion Matrix of Test Data ***************  \n')
                print (confusion_matrix(actual_class_labels, predicted_class_labels))
                cm=confusion_matrix(actual_class_labels, predicted_class_labels)
                TP = cm[1,1] 
                TN = cm[0,0] 
                FP = cm[0,1] 
                FN = cm[1,0] 
        
                re=TP/(TP+FN)
                pr=TP/(TP+FP)
                print ('\n Recall:'+str(re))
                print ('\n Precision:'+str(pr))
                print ('\n Specificity:'+str(TN/(TN+FP))) 
                print ('\n Fmeasure:'+str(2*pr*re/(pr+re)))
                acc = accuracy_score(actual_class_labels, predicted_class_labels)
                print('\n Accuracy: '+str(acc))
                print(classification_report(actual_class_labels, predicted_class_labels, target_names=class_names)) 
                test_data['Class']= label_encoder.inverse_transform(test_data['Class'])
                test_data['Predicted_class_labels']= label_encoder.inverse_transform(predicted_class_labels)
                test_data.to_csv('results.csv')

                
                print('\n !!!!! Submission file with the test data class labels is ready !!!!! \n')   


#
