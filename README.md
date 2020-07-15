# Introduction

Social media is a very fast-growing thing from the last decade. Most of the information generating today come from social media. In some cases, social media can have the capability of spreading the news more quickly than newspaper Media, TV media.And fake news can be spread just like a bush fire.\
Dataset contains categorical data, we need to apply some transformations before applying ML algorithms. As fake news detection dataset involves textual data, A special processing should be
done.ML provides Natural Language Processing techniques for handling textual datasets. 
![Image](C:\Users\Nidhi\Downloads\fake.jpg)

## How to detect fake news?

People can read the news and cross-check by googling it.Later it can be credited as real or fake depending on the results that we find.\
OR\
We can show an algorithm huge number of fake and real news articles so that it learns to differenciate between them automatically, and then it will give a probability score or percentage of confidence as an output for a given news article, that it is real or fake.

## Let's get started
### Importing libraries
```import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt 
import seaborn as sns 
%matplotlib inline
```
### Loading Dataset
```
df=pd.read_csv('news.csv')
df.head()
```

```
#DataFlair - Get the labels
labels=df.label
labels.head()
```
### Checking class distributions
```
df.groupby("label")['title'].count().plot.bar()
```
### Creating stopwords for cleaning the text
```
import nltk
nltk.download('stopwords')

from nltk.corpus import stopwords
from wordcloud import WordCloud, STOPWORDS 
nltk_stopwords =stopwords.words('english')[0:500:25]
wordcloud_stopwords = STOPWORDS

nltk_stopwords.extend(wordcloud_stopwords)

stopwords = set(nltk_stopwords)
print(stopwords)
```
### Cleaning the title feature
```
import re
def clean(text):
    text = re.sub("http\S+", '', str(text))  #removes urls(if any)
    
    text = re.sub(r"he's", "he is", str(text))
    text = re.sub(r"there's", "there is", str(text))
    text = re.sub(r"We're", "We are", str(text))
    text = re.sub(r"That's", "That is", str(text))
    text = re.sub(r"won't", "will not", str(text))
    text = re.sub(r"they're", "they are", str(text))
    text = re.sub(r"Can't", "Cannot", str(text))
    text = re.sub(r"wasn't", "was not", str(text))
    text = re.sub(r"aren't", "are not", str(text))
    text = re.sub(r"isn't", "is not", str(text))
    text = re.sub(r"What's", "What is", str(text))
    text = re.sub(r"haven't", "have not", str(text))
    text = re.sub(r"hasn't", "has not", str(text))
    text = re.sub(r"There's", "There is", str(text))
    text = re.sub(r"He's", "He is", str(text))
    text = re.sub(r"It's", "It is", str(text))
    text = re.sub(r"You're", "You are", str(text))
    text = re.sub(r"I'M", "I am", str(text))
    text = re.sub(r"shouldn't", "should not", str(text))
    text = re.sub(r"wouldn't", "would not", str(text))
    text = re.sub(r"i'm", "I am", str(text))
    text = re.sub(r"I'm", "I am", str(text))
    text = re.sub(r"Isn't", "is not", str(text))
    text = re.sub(r"Here's", "Here is", str(text))
    text = re.sub(r"you've", "you have", str(text))
    text = re.sub(r"we're", "we are", str(text))
    text = re.sub(r"what's", "what is", str(text))
    text = re.sub(r"couldn't", "could not", str(text))
    text = re.sub(r"we've", "we have", str(text))
    text = re.sub(r"who's", "who is", str(text))
    text = re.sub(r"y'all", "you all", str(text))
    text = re.sub(r"would've", "would have", str(text))
    text = re.sub(r"it'll", "it will", str(text))
    text = re.sub(r"we'll", "we will", str(text))
    text = re.sub(r"We've", "We have", str(text))
    text = re.sub(r"he'll", "he will", str(text))
    text = re.sub(r"Y'all", "You all", str(text))
    text = re.sub(r"Weren't", "Were not", str(text))
    text = re.sub(r"Didn't", "Did not", str(text))
    text = re.sub(r"they'll", "they will", str(text))
    text = re.sub(r"they'd", "they would", str(text))
    text = re.sub(r"DON'T", "DO NOT", str(text))
    text = re.sub(r"they've", "they have", str(text))
    text = re.sub(r"i'd", "I would", str(text))
    text = re.sub(r"should've", "should have", str(text))
    text = re.sub(r"where's", "where is", str(text))
    text = re.sub(r"we'd", "we would", str(text))
    text = re.sub(r"Here's", "Here is", str(text))
    text = re.sub(r"you've", "you have", str(text))
    text = re.sub(r"we're", "we are", str(text))
    text = re.sub(r"what's", "what is", str(text))
    text = re.sub(r"couldn't", "could not", str(text))
    text = re.sub(r"we've", "we have", str(text))
    text = re.sub(r"who's", "who is", str(text))
    text = re.sub(r"y'all", "you all", str(text))
    text = re.sub(r"would've", "would have", str(text))
    text = re.sub(r"it'll", "it will", str(text))
    text = re.sub(r"we'll", "we will", str(text))
    text = re.sub(r"We've", "We have", str(text))
    text = re.sub(r"he'll", "he will", str(text))
    text = re.sub(r"Y'all", "You all", str(text))
    text = re.sub(r"Weren't", "Were not", str(text))
    text = re.sub(r"Didn't", "Did not", str(text))
    text = re.sub(r"they'll", "they will", str(text))
    text = re.sub(r"they'd", "they would", str(text))
    text = re.sub(r"DON'T", "DO NOT", str(text))
    text = re.sub(r"they've", "they have", str(text))
    text = re.sub(r"i'd", "I would", str(text))
    text = re.sub(r"should've", "should have", str(text))
    text = re.sub(r"where's", "where is", str(text))
    text = re.sub(r"we'd", "we would", str(text))
    text = re.sub(r"i'll", "I will", str(text))
    text = re.sub(r"weren't", "were not", str(text))
    text = re.sub(r"They're", "They are", str(text))
    text = re.sub(r"let's", "let us", str(text))
    text = re.sub(r"it's", "it is", str(text))
    text = re.sub(r"can't", "cannot", str(text))
    text = re.sub(r"don't", "do not", str(text))
    text = re.sub(r"you're", "you are", str(text))
    text = re.sub(r"i've", "I have", str(text))
    text = re.sub(r"that's", "that is", str(text))
    text = re.sub(r"i'll", "I will", str(text))
    text = re.sub(r"doesn't", "does not", str(text))
    text = re.sub(r"i'd", "I would", str(text))
    text = re.sub(r"didn't", "did not", str(text))
    text = re.sub(r"ain't", "am not", str(text))
    text = re.sub(r"you'll", "you will", str(text))
    text = re.sub(r"I've", "I have", str(text))
    text = re.sub(r"Don't", "do not", str(text))
    text = re.sub(r"I'll", "I will", str(text))
    text = re.sub(r"I'd", "I would", str(text))
    text = re.sub(r"Let's", "Let us", str(text))
    text = re.sub(r"you'd", "You would", str(text))
    text = re.sub(r"It's", "It is", str(text))
    text = re.sub(r"Ain't", "am not", str(text))
    text = re.sub(r"Haven't", "Have not", str(text))
    text = re.sub(r"Could've", "Could have", str(text))
    text = re.sub(r"youve", "you have", str(text))
    
    # Others
    text = re.sub("U.S.", "United States", str(text))
    text = re.sub("Dec", "December", str(text))
    text = re.sub("Jan.","January", str(text))
    
    # Punctuations & special characters
    text = re.sub("[^A-Za-z0-9]+"," ", str(text))
    
    # removes stop words
    text = " ".join(str(i).lower() for i in text.split() if i.lower() not in stopwords)

    return text
```
```
df['text'] = df['text'].map(lambda x: clean(x))
df.text.iloc[:3] #selects rows and columns by number(3 rows seledcted)
```
### Splitting into train,test and CV
```
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df.drop('label',axis=1), df.label, test_size=0.2, random_state=42)

print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)
```
(5068, 3) (5068,)
(1267, 3) (1267,)

### Using sklearn TfidfVectorizer
```
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(min_df=0.01,ngram_range=(1,3))
vectorizer.fit(X_train.text)

X_tr = vectorizer.transform(X_train.text)
X_te = vectorizer.transform(X_test.text)

print(X_tr.shape, X_te.shape)
```
### Hyperparameter tuning Logistic Regression
```
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report
import math
clf = SGDClassifier(loss='log')##logistic regression

gs = GridSearchCV(
    estimator = clf,
    param_grid = {'alpha':np.logspace(-10,5,16)},
    cv = 5,
    return_train_score = True,
    scoring = 'accuracy'
    )

gs.fit(X_tr,y_train)

results = pd.DataFrame(gs.cv_results_)

results = results.sort_values(['param_alpha'])
train_auc = results['mean_train_score']
cv_auc = results['mean_test_score']
alpha = pd.Series([ math.log(i) for i in np.array(results['param_alpha']) ]) 

plt.plot(alpha, train_auc, label='Train AUC')
plt.plot(alpha, cv_auc, label='CV AUC')
plt.scatter(alpha, train_auc)
plt.scatter(alpha, cv_auc)
plt.legend()
plt.xlabel('log(alpha): hyperparameter')
plt.ylabel('Accuracy')
plt.title('Hyperparameter vs Accuracy Plot')
plt.grid()
plt.show()

print(gs.best_params_)
```
### Training on best parameters
```
clf = SGDClassifier(loss='log',alpha=1e-06, random_state=42).fit(X_tr,y_train)

print('Training score : %f' % clf.score(X_tr,y_train))
print('Test score : %f' % clf.score(X_te,y_test))
```
Training score : 1.000000
Test score : 0.928966\
```
print(classification_report(y_train.values, clf.predict(X_tr)))
confusion_matrix(y_train, clf.predict(X_tr))
```
### Final Test Scores
```
print(classification_report(y_test.values, clf.predict(X_te)))
pd.DataFrame(confusion_matrix(y_test, clf.predict(X_te)))
```
### Top 50 n-grams
```
coef = [abs(i) for i in clf.coef_.ravel()]
feature_names = vectorizer.get_feature_names()
feature_imp = dict(zip(feature_names,coef))
feature_imp = {k: v for k, v in sorted(feature_imp.items(), key=lambda item: item[1], reverse=True)}

top_50_features = {k: feature_imp[k] for k in list(feature_imp)[0:50]}

fig, ax = plt.subplots(figsize=(6,10))

people = top_50_features.keys()
y_pos = np.arange(len(people))
importance = top_50_features.values()

ax.barh(y_pos, importance,align='center')
ax.set_yticks(y_pos)
ax.set_yticklabels(people)
ax.invert_yaxis()
ax.set_xlabel('Importance')
ax.set_ylabel('Features')
ax.set_title('Top 50 Features')

plt.show()
```
### Bottom 50 n-grams
```
feature_imp = dict(zip(feature_names,coef))
feature_imp = {k: v for k, v in sorted(feature_imp.items(), key=lambda item: item[1], reverse=False)}

bottom_50_features = {k: feature_imp[k] for k in list(feature_imp)[0:50]}

fig, ax = plt.subplots(figsize=(6,10))

people = bottom_50_features.keys()
y_pos = np.arange(len(people))
importance = bottom_50_features.values()

ax.barh(y_pos, importance,align='center')
ax.set_yticks(y_pos)
ax.set_yticklabels(people)
ax.invert_yaxis()
ax.set_xlabel('Importance')
ax.set_ylabel('Features')
ax.set_title('Least 50 important features')

plt.show()
```



