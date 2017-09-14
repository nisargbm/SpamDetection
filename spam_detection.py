import pandas as pd
#import os
import numpy as np
#dataset = pd.read_csv('SMSSpamCollection', delimiter = '\t', header = None)
#X = dataset.iloc[:, 1].values
#y = dataset.iloc[:, 0].values
    
#import re
#import nltk
#nltk.download('stopwords')
#from nltk.corpus import stopwords
#from nltk.stem.porter import PorterStemmer
#X_temp=[]
#corpus = []
#y_temp = []
#for i in range(0,len(X)):
#    review = re.sub('[^a-zA-Z0-9]', ' ', X[i])
#    review = review.lower()
#    review = review.split()
#    ps = PorterStemmer()
#    review = [ps.stem(word) for word in review if not (word in set(stopwords.words('english')) or len(word) <= 2)]
#    review = ' '.join(review)
#    if len(review)>0:
#        X_temp.append(review)
#        if y[i] == 'ham':
#            y_temp.append(0)
#        else:
#            y_temp.append(1)    
#    if(i%100 == 0):
#        print(i)

#df = pd.DataFrame({'X_temp':X_temp,'y':y_temp})
#df.to_csv('stemmed_data.csv', index=False)


dataset = pd.read_csv('stemmed_data.csv')

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 2000)
X = cv.fit_transform(dataset.iloc[:,0]).toarray()
y = dataset.iloc[:, 1].values
#y_temp = cv.transform('movie was good')

from sklearn.feature_extraction.text import TfidfVectorizer

#sklearn_tfidf = TfidfVectorizer(norm='l2',min_df=0, use_idf=True, smooth_idf=False, sublinear_tf=True, tokenizer=tokenize)
sklearn_tfidf = TfidfVectorizer(norm='l2',max_features = 1000)
X = sklearn_tfidf.fit_transform(dataset.iloc[:,0]).toarray()
y = dataset.iloc[:, 1].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# Fitting Naive Bayes to the Training set
"""
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)
"""
"""
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)
"""

from sklearn.svm import SVC
classifier = SVC(kernel = 'linear', random_state = 0)
classifier.fit(X_train, y_train)

"""
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
#,penalty='l2',C=1,max_iter = 600)
classifier.fit(X_train, y_train)
"""
"""
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 2, metric = 'minkowski')
classifier.fit(X_train, y_train)
"""
# Predicting train result
y_train_pred = classifier.predict(X_train)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_train, y_train_pred)
score =classifier.score(X_train, y_train)
print("Accuracy: {}".format(score))
#print ((cm[0][0]+cm[1][1])/4427)

# Predicting the Test set results
y_pred = classifier.predict(X_test)
#classifier.predict(cv.fit_transform(["It was really good"]).toarray())
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
score =classifier.score(X_test, y_test)
print("Accuracy: {}".format(score))
#print ((cm[0][0]+cm[1][1])/1107)


from sklearn.model_selection import  cross_val_score
print("Cross Validation Accuracy:")
scores = cross_val_score(classifier, X_test, y_test, cv=5)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))


from sklearn.model_selection import GridSearchCV
parameters = [{'penalty': ['l2'], 'C': [1,2,5],'max_iter':[500,700,1000]}]
grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10,
                           n_jobs = -1)
grid_search = grid_search.fit(X_train, y_train)
best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_
