# -*- coding: utf-8 -*-
"""
Created on Fri Jul 22 21:14:47 2022

@author: Sruthi
"""
from irse2022 import comment_classif

#clf=comment_classif('/home/interns/irse2022/',model='entropy',clf_opt='svm',model_path='saved_models/',no_of_selected_features=3000)
#clf=comment_classif('/home/interns/irse2022/',model='tfidf',clf_opt='rf',model_path='saved_models/',no_of_selected_features=3000)
#clf=comment_classif('/home/interns/irse2022/',model='doc2vec',clf_opt='svm',model_path='saved_models/',no_of_selected_features=30)

#clf=comment_classif('C/home/interns/irse2022/',model='bert',model_source='bert-base-uncased')
#clf=comment_classif('C/home/interns/irse2022/',model='bert',model_source='roberta-base')
clf=comment_classif('C/home/interns/irse2022/',model='bert',model_source='albert-base-v2')
#clf=comment_classif('C/home/interns/irse2022/',model='bert',model_source='uclanlp/plbart-base')
clf.comment_classif()
