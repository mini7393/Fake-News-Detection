#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from boruta import BorutaPy
from sklearn.ensemble import RandomForestRegressor


# In[13]:


#the final file is having pos tagging and emotion tagging
final1=pd.read_csv("final.csv")
final1 = final1.loc[:, ~final1.columns.str.contains('^Unnamed')]


# In[14]:


#the final file is having pos tagging and emotion tagging
final=final1[["anger","anticipation","disgust","fear","joy","sadness","surprise","trust","negative","positive","noun_count","pronoun_count",
              "verb_count","adj_count","adverb_count","TYPE"]]
final.head()


# In[15]:



RES1=pd.read_csv("export1_dataframe.csv")
#has the text after preprocessing
RES1=RES1[["text"]]
for x in range(0, len(RES1['text'])):
    RES1["text"].iloc[x]= RES1['text'].iloc[x].strip('[]')
RES1['words'] = RES1.text.apply(lambda l : len(l.split()) )
RES1['characters'] = RES1.text.apply(lambda l : len(l))
words=RES1.text.apply(lambda l : len(l.split()) )
characters=RES1.text.apply(lambda l : len(l))

RES1.head()


# In[16]:


RES1.isnull().sum()


# In[17]:


RESULT = pd.concat([RES1, final], axis=1)

RESULT.rename(columns = {'anger':'anger_emotion', 'anticipation':'anticipation_emotion','digust':'disgust_emotion',
                        'fear':'fear_emotion','joy':'joy_emotion','sadness':'sadness_emotion','trust':'trust_emotion',
                        'negative':'negative_emotion','positive':'positive_emotion',"noun_count":"noun_count_pos",
                        "pronoun_count":'pronoun_count_pos', "verb_count":'verb_count_pos',"adj_count":'adj_count_pos',
                        "adverb_count":'averb_count',"TYPE":"TYPE_label","words":"words_text","characters":"characters_text"}, inplace = True) 
RESULT.head()


# In[18]:


X=RESULT.loc[:, RESULT.columns != 'TYPE_label']
y=RESULT.loc[:, RESULT.columns == 'TYPE_label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[20]:


print('Generating TFIDF matrix - Uni-grams and Bi-grams for train data')
tfidf_vectorizer = TfidfVectorizer(max_df=0.90, min_df=2, 
                                           max_features=10000, stop_words='english',
                                           ngram_range=(1,2))
tfidf_matrix = tfidf_vectorizer.fit_transform(X_train['text'].values)
tfidf_matrix = tfidf_matrix.todense()
feature_names = tfidf_vectorizer.get_feature_names()
TFIDF_df_train = pd.DataFrame(tfidf_matrix,columns=feature_names)
print(TFIDF_df_train.head())

print('Generating TFIDF matrix - Uni-grams and Bi-grams for test data')
tfidf_vectorizer = TfidfVectorizer(max_df=0.90, min_df=2, 
                                           max_features=10000, stop_words='english',
                                           ngram_range=(1,2))
tfidf_matrix = tfidf_vectorizer.fit_transform(X_test['text'].values)
tfidf_matrix = tfidf_matrix.todense()
feature_names = tfidf_vectorizer.get_feature_names()
TFIDF_df_test = pd.DataFrame(tfidf_matrix,columns=feature_names)
print(TFIDF_df_train.head())


# In[ ]:


X_train.pop('text')
X_test.pop('text')
X_train=X_train.reset_index()
X_train.pop('index')
y_train=y_train.reset_index()
y_train.pop('index')
X_test=X_test.reset_index()
X_test.pop('index')
y_test=y_test.reset_index()
y_test.pop('index')
X_train.head()


# In[10]:


rf = RandomForestClassifier(n_jobs=-1, class_weight='balanced', max_depth=5)
boruta_feature_selector = BorutaPy(rf, n_estimators='auto', verbose=2, random_state=4242, max_iter = 50, perc = 90)
boruta_feature_selector.fit(TFIDF_df_train.values, y_train.values)


# In[11]:


X_filtered = boruta_feature_selector.transform(TFIDF_df_train.values)
print(X_filtered)
final_features = list()
indexes = np.where(boruta_feature_selector.support_ == True)
for x in np.nditer(indexes):
    final_features.append(TFIDF_df_train.columns[x])
print(final_features)
final_features=pd.Index(final_features)
a = list(final_features & TFIDF_df_test.columns)
b = list(set(final_features) - set(TFIDF_df_test.columns))
du = TFIDF_df_test[a]
for i in b:
    du[i] = 0
print(du)
TFIDF_df_train= TFIDF_df_train[final_features]
#train with features
TFIDF_df_train.shape
#test with features
du.shape
X_train1 = pd.concat([TFIDF_df_train,X_train], axis=1)
X_test1 = pd.concat([du,X_test], axis=1)
print(X_train1.shape)
print(X_test1.shape)


# In[12]:


def getDuplicateColumns(df):

    duplicateColumnNames = set()
    # Iterate over all the columns in dataframe
    for x in range(df.shape[1]):
        # Select column at xth index.
        col = df.iloc[:, x]
        # Iterate over all the columns in DataFrame from (x+1)th index till end
        for y in range(x + 1, df.shape[1]):
            # Select column at yth index.
            otherCol = df.iloc[:, y]
            # Check if two columns at x 7 y index are equal
            if col.equals(otherCol):
                duplicateColumnNames.add(df.columns.values[y])
 
    return list(duplicateColumnNames)


# In[ ]:


duplicateColumnNames = getDuplicateColumns(X_train1)
 
print('Duplicate Columns are as follows')
for col in duplicateColumnNames:
    print('Column name : ', col)


# In[ ]:


X_train1=X_train1.drop(duplicateColumnNames, axis=1)
X_test1=X_test1.drop(duplicateColumnNames, axis=1)


# In[ ]:


X_train1.head()


# In[ ]:


X_test1.head()


# In[ ]:


from sklearn.metrics import confusion_matrix, r2_score,accuracy_score , classification_report,roc_curve, auc
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict


# ## SVM

# In[ ]:


from sklearn.svm import LinearSVC
model4 = LinearSVC()
model4.fit(X_train1, y_train)
y_pred = model4.predict(X_test1)


# In[ ]:


y_predtrain = model4.predict(X_train1)
print('accuracy',round(accuracy_score(y_train, y_predtrain)*100,2),'%')
print('classification report \n',classification_report(y_train, y_predtrain))


# # from sklearn.model_selection import cross_val_predict
# from sklearn.metrics import confusion_matrix
# from xgboost import XGBClassifier
# from sklearn.svm import LinearSVC
# from sklearn.metrics import confusion_matrix, r2_score,accuracy_score , classification_report,roc_curve, auc

# y_train = y_train.astype(object)
# y_test = y_test.astype(object)

# In[ ]:





# In[ ]:


print('accuracy',round(accuracy_score(y_test, y_pred)*100,2),'%')
print('classification report \n',classification_report(y_test, y_pred))
print('confusion matrix \n',confusion_matrix(y_test, y_pred))
false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)
roc_auc = auc(false_positive_rate, true_positive_rate)
print('roc auc',round(roc_auc*100,2),'%')
scores2 = cross_val_score(model4, X_train1,y_train, cv=5)


# In[ ]:


print ("Accuracy after cross-validation for SVM Model:/n" ,scores2*100)
y_predict2 = cross_val_predict(model4, X_train1, y_train, cv=5)
conf_mat2 = confusion_matrix(y_train, y_predict2)
print(conf_mat2)


# ## Naive Bayes

# In[ ]:


from sklearn.naive_bayes import GaussianNB,BernoulliNB
#naive_bayes model
Nb = BernoulliNB()
Nb.fit(X_train1, y_train)
y_pred = Nb.predict(X_test1)
print('accuracy',round(accuracy_score(y_test, y_pred)*100,2),'%')
print('classification report\n',classification_report(y_test, y_pred))
print('confusion matrix \n',confusion_matrix(y_test, y_pred))
false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)
roc_auc = auc(false_positive_rate, true_positive_rate)
print('roc auc',round(roc_auc*100,2),'%')
scores1 = cross_val_score(Nb, X_train1,y_train, cv=5)


# In[ ]:


print ("Accuracy after cross-validation for Naive Bayes Model:" ,scores1*100)
y_predict1 = cross_val_predict(Nb, X_train1, y_train, cv=5)
conf_mat1 = confusion_matrix(y_train, y_predict1)
print(conf_mat1)


# In[ ]:


y_predtrain = Nb.predict(X_train1)
print('accuracy',round(accuracy_score(y_train, y_predtrain)*100,2),'%')
print('classification report \n',classification_report(y_train, y_predtrain))


# In[ ]:


#y_test


# ## XG boost

# In[ ]:


import xgboost as xgb
model=xgb.XGBClassifier(random_state=1,learning_rate=0.01)
 
# fit the model with the training data
model.fit(X_test1 ,y_train)
predict_train = model.predict(X_train1)


# In[ ]:





# model = XGBClassifier()
# model.fit(X_train1, y_train)
# print(model)
# y_pred = model.predict(X_test1)
# print('accuracy',round(accuracy_score(y_test, y_pred)*100,2),'%')
# print('classification report \n',classification_report(y_test, y_pred))
# print('confusion matrix \n',confusion_matrix(y_test, y_pred))    
# false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)
# roc_auc = auc(false_positive_rate, true_positive_rate)
# print('roc auc',round(roc_auc*100,2),'%')
# #scores5 = cross_val_score(model, X, y, cv=10)

# ## Random Forest

# In[ ]:


from sklearn.ensemble import RandomForestRegressor

regressor = RandomForestRegressor(n_estimators=20, random_state=0)
regressor.fit(X_train1, y_train)
predictions = regressor.predict(X_test1)


# In[ ]:


for i in range(len(predictions)):
    if(predictions[i] > 0.5):
        predictions[i] =1
    else:
        predictions[i]=0


# In[ ]:


print('accuracy',round(accuracy_score(y_test,predictions)*100,2),'%')
print('classification report\n',classification_report(y_test, predictions))
print('confusion matrix \n',confusion_matrix(y_test, predictions))
false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, predictions)
roc_auc = auc(false_positive_rate, true_positive_rate)
print('roc auc',round(roc_auc*100,2),'%')

scores1 = cross_val_score(regressor, X_train1,y_train, cv=5)
print ("Accuracy after cross-validation for Random Forest Model:" ,scores1*100)
y_predict1 = cross_val_predict(regressor, X_train1,y_train, cv=5)
conf_mat1 = confusion_matrix(y_train, y_predict1)
print(conf_mat1)
confusion_matrix(y_test, predictions)


# In[ ]:




