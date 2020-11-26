import hard as hard
import sklearn.datasets as dt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn import svm
from sklearn import metrics
from sklearn.metrics import accuracy_score, roc_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import plot_confusion_matrix
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn import neighbors
import os
import time
import numpy as np
from sklearn import metrics
start = time.time()

start = time.time()
os.chdir(r'C:\Users\mr.geek\Desktop')
print(os.getcwd())
categories = ['Philonthropists', 'Politcians', 'Showbiz', 'sportsmen', 'Writers']
train = dt.load_files(r'C:\Users\mr.geek\Desktop\Learning\Train', categories=categories, encoding='ISO-8859-1')
test = dt.load_files(r'C:\Users\mr.geek\Desktop\Learning\Test', categories=categories, encoding='ISO-8859-1')
print(train.keys())
print(train['target_names'])
# term frequency counting
count_vector = CountVectorizer()
x_trian_tf = count_vector.fit_transform(train.data)
# print(x_trian_tf.shape)
# print(x_trian_tf.toarray())
# TFIDF
tfidf_transformer = TfidfTransformer()
x_train_tfidf = tfidf_transformer.fit_transform(x_trian_tf)
# print(x_train_tfidf.shape)
# Training our Model
learn = MultinomialNB().fit(x_train_tfidf, train.target)
# testing data
x_test_tf = count_vector.transform(test.data)
x_test_tfidf = tfidf_transformer.transform(x_test_tf)
proba = learn.predict_proba(x_test_tfidf)
prediction = learn.predict(x_test_tfidf)
# reporting accuracy
print("Accuracy is of Multinomial Naive Bayes Classifier", accuracy_score(test.target, prediction) * 100)
print(metrics.classification_report(test.target, prediction, target_names=test.target_names))
print(metrics.confusion_matrix(test.target, prediction))
disp = plot_confusion_matrix(learn, x_test_tfidf, test.target, display_labels=test.target_names, cmap=plt.cm.Blues,)
title = "Confusion Matrix Plot for Naive Bayes Classifier"
disp.ax_.set_title(title)
plt.show()
fig1 = plt.figure()
chart = 0
fig1.suptitle("ROC Curve for Different Class Labels")
for i in range(0, (len(categories))):
    print(i)
    y_test_bin = np.int32(test.target == i)
    y_score = proba[:, i]
    fpr, tpr, _ = roc_curve(y_test_bin, y_score, pos_label=1)
    roc_auc = metrics.auc(fpr, tpr)
    chart += 1
    ax = fig1.add_subplot(3, 2, chart)
    ax.set_title(categories[i])
    fig1.subplots_adjust(hspace=1)
    roc_display = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name=categories[i])
    roc_display.plot(ax=ax)
plt.show()
# 2- Text Classification Using SVM
SVM = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto', probability=True)
SVM.fit(x_train_tfidf, train.target)
proba = SVM.predict_proba(x_test_tfidf)
prediction_svm = SVM.predict(x_test_tfidf)
print("SVM Accuracy Score is ", accuracy_score(prediction_svm, test.target) * 100)
print(metrics.classification_report(test.target, prediction_svm, target_names=test.target_names))
print(metrics.confusion_matrix(test.target, prediction_svm))
disp = plot_confusion_matrix(SVM, x_test_tfidf, test.target, display_labels=test.target_names, cmap=plt.cm.Blues,)
title = "Confusion Matrix Plot for SVM Classifier"
disp.ax_.set_title(title)
plt.show()
fig1 = plt.figure()
chart = 0
fig1.suptitle("ROC Curve for Different Class Labels")
for i in range(len(categories)):
    y_test_bin = np.int32(test.target == i)
    y_score = proba[:, i]
    fpr, tpr, _ = roc_curve(y_test_bin, y_score, pos_label=1)
    roc_auc = metrics.auc(fpr, tpr)
    chart += 1
    ax = fig1.add_subplot(3, 2, chart)
    ax.set_title(categories[i])
    fig1.subplots_adjust(hspace=1)
    roc_display = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name=categories[i])
    roc_display.plot(ax=ax)
plt.show()
# 3- Text Classification Using Random Forest
Random_Forest = RandomForestClassifier(n_estimators=100)
Random_Forest.fit(x_train_tfidf, train.target)
proba = Random_Forest.predict_proba(x_test_tfidf)
prediction_rf = Random_Forest.predict(x_test_tfidf)
print("Random Forest Accuracy Score is ", accuracy_score(prediction_rf, test.target) * 100)
print(metrics.classification_report(test.target, prediction_rf, target_names=test.target_names))
print(metrics.confusion_matrix(test.target, prediction_rf))
disp = plot_confusion_matrix(Random_Forest, x_test_tfidf, test.target, display_labels=test.target_names, cmap=plt.cm.Blues,)
title = "Confusion Matrix Plot for Random Forest Classifier"
disp.ax_.set_title(title)
plt.show()
fig1 = plt.figure()
chart = 0
fig1.suptitle("ROC Curve for Different Class Labels")
for i in range(len(categories)):
    y_test_bin = np.int32(test.target == i)
    y_score = proba[:, i]
    fpr, tpr, _ = roc_curve(y_test_bin, y_score, pos_label=1)
    roc_auc = metrics.auc(fpr, tpr)
    chart += 1
    ax = fig1.add_subplot(3, 2, chart)
    ax.set_title(categories[i])
    fig1.subplots_adjust(hspace=1)
    roc_display = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name=categories[i])
    roc_display.plot(ax=ax)
plt.show()
# 4- Text Classification Using KNN
learn = neighbors.KNeighborsClassifier()
learn.fit(x_train_tfidf, train.target)
proba = learn.predict_proba(x_test_tfidf)
prediction = learn.predict(x_test_tfidf)
print("KNN Score is ", accuracy_score(prediction, test.target) * 100)
print(metrics.classification_report(test.target, prediction, target_names=test.target_names))
print(metrics.confusion_matrix(test.target, prediction))
disp = plot_confusion_matrix(learn, x_test_tfidf, test.target, display_labels=test.target_names, cmap=plt.cm.Blues,)
title = "Confusion Matrix Plot for KNN"
disp.ax_.set_title(title)
plt.show()
fig1 = plt.figure()
chart = 0
fig1.suptitle("ROC Curve for Different Class Labels")
for i in range(len(categories)):
    y_test_bin = np.int32(test.target == i)
    y_score = proba[:, i]
    fpr, tpr, _ = roc_curve(y_test_bin, y_score, pos_label=1)
    roc_auc = metrics.auc(fpr, tpr)
    chart += 1
    ax = fig1.add_subplot(3, 2, chart)
    ax.set_title(categories[i])
    fig1.subplots_adjust(hspace=1)
    roc_display = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name=categories[i])
    roc_display.plot(ax=ax)
plt.show()
# 5- Ensemble Learning (fusion of classifiers) of NB, SVM  base classifiers.
# create a dictionary of our models
estimators1 = [("Naive_Bayes", MultinomialNB()), ("svm", svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto'))]
# create our voting classifier, inputting our models
ensemble = VotingClassifier(estimators1, voting="hard")
# fit model to training data
ensemble.fit(x_train_tfidf, train.target)
# proba = ensemble.predict_proba(x_test_tfidf)
# test our model on the test data
ensem_predict = ensemble.predict(x_test_tfidf)
print("The accuracy of Naive Bayes and SVM is ", accuracy_score(ensem_predict, test.target) * 100)
print(metrics.classification_report(test.target, ensem_predict, target_names=test.target_names))
print(metrics.confusion_matrix(test.target, ensem_predict))
disp = plot_confusion_matrix(ensemble, x_test_tfidf, test.target, display_labels=test.target_names, cmap=plt.cm.Blues,)
title = "Confusion Matrix Plot for Ensemble Model of NB and SVM Classifiers"
disp.ax_.set_title(title)
plt.show()

# 6 - Ensemble of NB and Random Forest Base classifier.
estimators2 = [("Naive_Bayes", MultinomialNB()), ("Random_Forest", RandomForestClassifier(n_estimators=100))]
ensemble = VotingClassifier(estimators2, voting="hard")
# fit model to training data
ensemble.fit(x_train_tfidf, train.target)
# proba = ensemble.predict_proba(x_test_tfidf)
ensem_predict = ensemble.predict(x_test_tfidf)
# test our model on the test data
print("The accuracy of Naive Bayes and Random Forest is ", accuracy_score(ensem_predict, test.target) * 100)
print(metrics.classification_report(test.target, ensem_predict, target_names=test.target_names))
print(metrics.confusion_matrix(test.target, ensem_predict))
disp = plot_confusion_matrix(ensemble, x_test_tfidf, test.target, display_labels=test.target_names, cmap=plt.cm.Blues,)
title = "Confusion Matrix Plot for Ensemble Model of NB and RF Classifiers"
disp.ax_.set_title(title)

# 7 - Ensemble of NB and KNN Base classifier.
estimators3 = [("Naive_Bayes", MultinomialNB()), ("KNN", neighbors.KNeighborsClassifier())]
ensemble = VotingClassifier(estimators2, voting="hard")
# fit model to training data
ensemble.fit(x_train_tfidf, train.target)
# proba = ensemble.predict_proba(x_test_tfidf)
ensem_predict = ensemble.predict(x_test_tfidf)
# test our model on the test data
print("The accuracy of Naive Bayes and KNN is ", accuracy_score(ensem_predict, test.target) * 100)
print(metrics.classification_report(test.target, ensem_predict, target_names=test.target_names))
print(metrics.confusion_matrix(test.target, ensem_predict))
disp = plot_confusion_matrix(ensemble, x_test_tfidf, test.target, display_labels=test.target_names, cmap=plt.cm.Blues,)
title = "Confusion Matrix Plot for Ensemble Model of NB and KNN Classifiers"
disp.ax_.set_title(title)
plt.show()
# 8 - Ensemble of SVM and Random forest  classifier.
estimators4 = [("SVM", svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')), ("Random_Forest", RandomForestClassifier(n_estimators=100))]
ensemble = VotingClassifier(estimators4, voting="hard")
# fit model to training data
ensemble.fit(x_train_tfidf, train.target)
# proba = ensemble.predict_proba(x_test_tfidf)
ensem_predict = ensemble.predict(x_test_tfidf)
# test our model on the test data
print("The accuracy of SVM and Random Forest is ", accuracy_score(ensem_predict, test.target) * 100)
print(metrics.classification_report(test.target, ensem_predict, target_names=test.target_names))
print(metrics.confusion_matrix(test.target, ensem_predict))
disp = plot_confusion_matrix(ensemble, x_test_tfidf, test.target, display_labels=test.target_names, cmap=plt.cm.Blues,)
title = "Confusion Matrix Plot for Ensemble Model of SVM and RF Classifiers"
disp.ax_.set_title(title)
plt.show()

# 9 - Ensemble of SVM and KNN  classifier.
estimators5 = [("SVM", svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')), ("KNN", neighbors.KNeighborsClassifier())]
ensemble = VotingClassifier(estimators5, voting="hard")
# fit model to training data
ensemble.fit(x_train_tfidf, train.target)
# proba = ensemble.predict_proba(x_test_tfidf)
ensem_predict = ensemble.predict(x_test_tfidf)
# test our model on the test data
print("The accuracy of SVM and KNN is ", accuracy_score(ensem_predict, test.target) * 100)
print(metrics.classification_report(test.target, ensem_predict, target_names=test.target_names))
print(metrics.confusion_matrix(test.target, ensem_predict))
disp = plot_confusion_matrix(ensemble, x_test_tfidf, test.target, display_labels=test.target_names, cmap=plt.cm.Blues,)
title = "Confusion Matrix Plot for Ensemble Model of SVM and KNN Classifiers"
disp.ax_.set_title(title)
plt.show()

# 10 - Ensemble of Random Forest and KNN classifier.
estimators6 = [("Random_Forest", RandomForestClassifier(n_estimators=100)), ("KNN", neighbors.KNeighborsClassifier())]
ensemble = VotingClassifier(estimators6, voting="hard")
# fit model to training data
ensemble.fit(x_train_tfidf, train.target)
# proba = ensemble.predict_proba(x_test_tfidf)
ensem_predict = ensemble.predict(x_test_tfidf)
# test our model on the test data
print("The accuracy of Random Forest and KNN is ", accuracy_score(ensem_predict, test.target) * 100)
print(metrics.classification_report(test.target, ensem_predict, target_names=test.target_names))
print(metrics.confusion_matrix(test.target, ensem_predict))
disp = plot_confusion_matrix(ensemble, x_test_tfidf, test.target, display_labels=test.target_names, cmap=plt.cm.Blues,)
title = "Confusion Matrix Plot for Ensemble Model of RF and KNN Classifiers"
disp.ax_.set_title(title)
plt.show()

# 11 - Ensemble of Naive Bayes and SVM and Random Forest classifier.
estimators7 = [("Naive_Bayes", MultinomialNB()), ("SVM", svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')), ("Random_Forest", RandomForestClassifier(n_estimators=100))]
ensemble = VotingClassifier(estimators7, voting="hard")
# fit model to training data
ensemble.fit(x_train_tfidf, train.target)
# proba = ensemble.predict_proba(x_test_tfidf)
ensem_predict = ensemble.predict(x_test_tfidf)
# test our model on the test data
print("The accuracy of NBC, Random Forest and SVM is ", accuracy_score(ensem_predict, test.target) * 100)
print(metrics.classification_report(test.target, ensem_predict, target_names=test.target_names))
print(metrics.confusion_matrix(test.target, ensem_predict))
disp = plot_confusion_matrix(ensemble, x_test_tfidf, test.target, display_labels=test.target_names, cmap=plt.cm.Blues,)
title = "Confusion Matrix Plot for Ensemble Model of NB, SVM and RF Classifiers"
disp.ax_.set_title(title)
plt.show()

# 12 - Ensemble of Naive Bayes and SVM and KNN classifier.
estimators8 = [("Naive_Bayes", MultinomialNB()), ("SVM", svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')), ("KNN", neighbors.KNeighborsClassifier())]
ensemble = VotingClassifier(estimators8, voting="hard")
# fit model to training data
ensemble.fit(x_train_tfidf, train.target)
# proba = ensemble.predict_proba(x_test_tfidf)
ensem_predict = ensemble.predict(x_test_tfidf)
# test our model on the test data
print("The accuracy of Naive Bayes, SVM and KNN is ", accuracy_score(ensem_predict, test.target) * 100)
print(metrics.classification_report(test.target, ensem_predict, target_names=test.target_names))
print(metrics.confusion_matrix(test.target, ensem_predict))
disp = plot_confusion_matrix(ensemble, x_test_tfidf, test.target, display_labels=test.target_names, cmap=plt.cm.Blues,)
title = "Confusion Matrix Plot for Ensemble Model of NB, SVM and KNN Classifiers"
disp.ax_.set_title(title)
plt.show()

# 13 - Ensemble of SVM and Random Forest and KNN classifier.
estimators9 = [("SVM", svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')), ("Random_Forest", RandomForestClassifier(n_estimators=100)), ("KNN", neighbors.KNeighborsClassifier())]
ensemble = VotingClassifier(estimators9, voting="hard")
# fit model to training data
ensemble.fit(x_train_tfidf, train.target)
# proba = ensemble.predict_proba(x_test_tfidf)
ensem_predict = ensemble.predict(x_test_tfidf)
# test our model on the test data
print("The accuracy of SVM , Random Forest and KNN is ", accuracy_score(ensem_predict, test.target) * 100)
print(metrics.classification_report(test.target, ensem_predict, target_names=test.target_names))
print(metrics.confusion_matrix(test.target, ensem_predict))
disp = plot_confusion_matrix(ensemble, x_test_tfidf, test.target, display_labels=test.target_names, cmap=plt.cm.Blues,)
title = "Confusion Matrix Plot for Ensemble Model of SVM, RF and KNN Classifiers"
disp.ax_.set_title(title)
plt.show()

# 14 - Ensemble of Naive Bayes and Random Forest and KNN classifier.
estimators10 = [("Naive_Bayes", MultinomialNB()), ("Random_Forest", RandomForestClassifier(n_estimators=100)), ("KNN", neighbors.KNeighborsClassifier())]
ensemble = VotingClassifier(estimators10, voting="hard")
# fit model to training data
ensemble.fit(x_train_tfidf, train.target)
# proba = ensemble.predict_proba(x_test_tfidf)
ensem_predict = ensemble.predict(x_test_tfidf)
# test our model on the test data
print("The accuracy of Naive Bayes , Random Forest and KNN is ", accuracy_score(ensem_predict, test.target) * 100)
print(metrics.classification_report(test.target, ensem_predict, target_names=test.target_names))
print(metrics.confusion_matrix(test.target, ensem_predict))
disp = plot_confusion_matrix(ensemble, x_test_tfidf, test.target, display_labels=test.target_names, cmap=plt.cm.Blues,)
title = "Confusion Matrix Plot for Ensemble Model of NB, RF and KNN Classifiers"
disp.ax_.set_title(title)
plt.show()

# 15 - Ensemble of Naive Bayes, SVM, Random Forest and KNN.
estimators11 = [("Naive_Bayes", MultinomialNB()), ("SVM", svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')), ("Random_Forest", RandomForestClassifier(n_estimators=100)), ("KNN", neighbors.KNeighborsClassifier())]
ensemble = VotingClassifier(estimators11, voting="hard")
# fit model to training data
ensemble.fit(x_train_tfidf, train.target)
# proba = ensemble.predict_proba(x_test_tfidf)
ensem_predict = ensemble.predict(x_test_tfidf)
# test our model on the test data
print("The accuracy of Naive Bayes, SVM, KNN and Random Forest is ", accuracy_score(ensem_predict, test.target) * 100)
print(metrics.classification_report(test.target, ensem_predict, target_names=test.target_names))
print(metrics.confusion_matrix(test.target, ensem_predict))
disp = plot_confusion_matrix(ensemble, x_test_tfidf, test.target, display_labels=test.target_names, cmap=plt.cm.Blues,)
title = "Confusion Matrix Plot for Ensemble Model of NB, SVM,  RF and KNN Classifiers"
disp.ax_.set_title(title)
plt.show()

# Calculating Total take by this script to RUN
end = time.time()
print("the total time taken this script to run is", end - start, "seconds.")
