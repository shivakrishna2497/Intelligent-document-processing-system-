import os
import re
from io import StringIO
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from textblob import TextBlob
from textblob import Word
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix
from sklearn.feature_selection import chi2
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score
import seaborn as sns
from sklearn.metrics import confusion_matrix
from IPython.display import display
from sklearn import metrics
from sklearn.metrics import classification_report
from collections import defaultdict

def loaddata(data_dir):
    data = {}
    for split in ["train"]:
        data[split] = []
        for cat in ["Assignments, deeds, conveyances", "Contract", "Correspondence", "Division orders", "Leases", "Other ", "Production-Operation report"]:
            if cat == "Assignments, deeds, conveyances":
                score = "Assignments, deeds, conveyances" 
            elif cat == "Contract":
                score = "Contract" 
            elif cat == "Correspondence":
                score = "Correspondence"
            elif cat == "Division orders":
                score = "Division orders"
            elif cat == "Leases":
                score = "Leases"
            elif cat == "Leases":
                score == "Leases"
            elif cat == "Other ":
                score = "Other "
            else:
                score = "Production-Operation report" 
            path = os.path.join(data_dir, split, cat)
            file_names = os.listdir(path)
            for f_name in file_names:
                with open(os.path.join(path, f_name), "r") as f:
                    review = f.read()
                    data[split].append([review, score])
    np.random.shuffle(data["train"])        
    data["train"] = pd.DataFrame(data["train"],
                                 columns=['text', 'category'])
    print(data["train"])
    print(data["train"].shape)
    return data["train"]
train_data= loaddata(
    data_dir="Text/")
train_data.to_csv('/home/shiva/Sirius/CSV/train.csv', index=False)
# Data Exploration
df = pd.read_csv('/home/shiva/Sirius/CSV/train.csv')
print('Number of samples:',len(df))
print('\n',df.describe())
print('\n',df.info())
print('\n',df.head())
# Basic Feature Extraction
# Number of Words
df['word_count'] = df['text'].apply(lambda x: len(str(x).split(" ")))
# Number of characters
df['char_count'] = df['text'].str.len() 
# Average Word Length
def avg_word(sentence):
  words = str(sentence).split()
  return (sum(len(word) for word in words)/len(words))
df['avg_word'] = df['text'].apply(lambda x: avg_word(x))
# Number of stopwords
stop = stopwords.words('english')
df['stopwords'] = df['text'].apply(lambda x: len([x for x in str(x).split() if x in stop]))
# Number of special characters
df['hashtags'] = df['text'].apply(lambda x: len([x for x in str(x).split() if x.startswith('#')]))
# Number of numerics
df['numerics'] = df['text'].apply(lambda x: len([x for x in str(x).split() if x.isdigit()]))
# Number of Uppercase words
df['upper'] = df['text'].apply(lambda x: len([x for x in str(x).split() if x.isupper()]))
print('\n Basic Feature Extraction \n ', df[['text', 'word_count', 'char_count', 'avg_word', 'stopwords', 'hashtags', 'numerics', 'upper']].head())
# Basic Pre-processing
# Lower case
df['text'] = df['text'].apply(lambda x: " ".join(x.lower() for x in str(x).split()))
# Removing Punctuation
df['text'] = df['text'].str.replace('[^\w\s]','')
# Removal of Stop Words
stop = stopwords.words('english')
df['text'] = df['text'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))
# Common word removal
freq = pd.Series(' '.join(df['text']).split()).value_counts()[:10]
freq = list(freq.index)
df['text'] = df['text'].apply(lambda x: " ".join(x for x in str(x).split() if x not in freq))
# Rare words removal
# freq = pd.Series(' '.join(df['text']).split()).value_counts()[-10:]
# # print(freq)
# freq = list(freq.index)
# df['text'] = df['text'].apply(lambda x: " ".join(x for x in str(x).split() if x not in freq))
# print('\n', df['text'].head())
# Spelling correction
df['text'][:5].apply(lambda x: str(TextBlob(x).correct()))
# Tokenization
TextBlob(df['text'][1]).words
# print(TextBlob(df['text'][1]).words)
# Stemming
st = PorterStemmer()
df['text'][:5].apply(lambda x: " ".join([st.stem(word) for word in str(x).split()]))
# words = stopwords.words("english")
# df['cleaned'] = df['text'].apply(lambda x: " ".join([st.stem(word) for word in re.sub("[^a-zA-Z]", " ", x).split() if word not in words]).lower())
# print(df['text','category','cleaned'])
# Lemmatization
df['text'] = df['text'].apply(lambda x: " ".join([Word(word).lemmatize() for word in str(x).split()]))
# words = stopwords.words("english")
# df['text'] = df['text'].apply(lambda x: " ".join([Word(word).lemmatize() for word in re.sub("[^a-zA-Z]", " ", x).split() if word not in words]).lower())
print('\n', df['text'].head())
col = ['category', 'text']
df = df[col]
df = df[pd.notnull(df['text'])]
df.columns = ['category', 'text']
df['category_id'] = df['category'].factorize()[0]
category_id_df = df[['category', 'category_id']].drop_duplicates().sort_values('category_id')
category_to_id = dict(category_id_df.values)
id_to_category = dict(category_id_df[['category_id', 'category']].values)
print(df.head())
# Checking Imabalanced Classes 
fig = plt.figure(figsize=(8,6))
df.groupby('category').text.count().plot.bar(ylim=0)
plt.show()
# Text Representation
tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1', ngram_range=(1, 2), stop_words='english')
features = tfidf.fit_transform(df.text).toarray()
labels = df.category_id
print(features.shape)
N = 2
for category, category_id in sorted(category_to_id.items()):
  features_chi2 = chi2(features, labels == category_id)
  indices = np.argsort(features_chi2[0])
  feature_names = np.array(tfidf.get_feature_names())[indices]
  unigrams = [v for v in feature_names if len(v.split(' ')) == 1]
  bigrams = [v for v in feature_names if len(v.split(' ')) == 2]
  print("# '{}':".format(category))
  print("  . Most correlated unigrams:\n. {}".format('\n. '.join(unigrams[-N:])))
  print("  . Most correlated bigrams:\n. {}".format('\n. '.join(bigrams[-N:])))
# Multi-Class Classifier: Features and Design
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['category'],train_size = 0.67, test_size=0.33, random_state=42)
X_train.to_csv('/home/shiva/Sirius/CSV/X_train.csv',index=False, header=False)
print('\n Number of Training samples (X_train):',len(X_train))
X_test.to_csv('/home/shiva/Sirius/CSV/X_test.csv',index=False, header=False)
print('\n Number of Testing samples (X_test):',len(X_test))
y_train.to_csv('/home/shiva/Sirius/CSV/y_train.csv',index=False, header=False)
print('\n Number of Training samples (y_train):',len(y_train))
y_test.to_csv('/home/shiva/Sirius/CSV/y_test.csv',index=False, header=False)
print('\n Number of Testing samples (y_test):',len(y_test))
print(df.groupby('category').count())
print(y_train.value_counts())
print(y_test.value_counts())
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(X_train)
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
clf = MultinomialNB().fit(X_train_tfidf, y_train)
print(clf.predict(count_vect.transform(["Zamara Mendoza\n\nFrom: Jeanne Butler\n\nSent:..."])))
df[df['text'] == "Zamara Mendoza\n\nFrom: Jeanne Butler\n\nSent:..."]
# Model Selection
models = [
    RandomForestClassifier(n_estimators=200, max_depth=3, random_state=0),
    LinearSVC(),
    MultinomialNB(),
    LogisticRegression(random_state=0,solver='lbfgs', multi_class='auto'),
]
CV = 5
cv_df = pd.DataFrame(index=range(CV * len(models)))
entries = []
for model in models:
  model_name = model.__class__.__name__
  accuracies = cross_val_score(model, features, labels, scoring='accuracy', cv=CV)
  for fold_idx, accuracy in enumerate(accuracies):
    entries.append((model_name, fold_idx, accuracy))
cv_df = pd.DataFrame(entries, columns=['model_name', 'fold_idx', 'accuracy'])
sns.boxplot(x='model_name', y='accuracy', data=cv_df)
sns.stripplot(x='model_name', y='accuracy', data=cv_df, 
              size=8, jitter=True, edgecolor="gray", linewidth=2)
plt.show()
print(cv_df.groupby('model_name').accuracy.mean())
model = LinearSVC()
X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(features, labels, df.index, test_size=0.33, random_state=0)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
conf_mat = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(10,10))
sns.heatmap(conf_mat, annot=True, fmt='d',
            xticklabels=category_id_df.category.values, yticklabels=category_id_df.category.values)
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()
for predicted in category_id_df.category_id:
  for actual in category_id_df.category_id:
    if predicted != actual and conf_mat[actual, predicted] >= 10:
      print("'{}' predicted as '{}' : {} examples.".format(id_to_category[actual], id_to_category[predicted], conf_mat[actual, predicted]))
      display(df.loc[indices_test[(y_test == actual) & (y_pred == predicted)]][['category', 'text']])
      print('')
model.fit(features, labels)
N = 2
for category, category_id in sorted(category_to_id.items()):
  indices = np.argsort(model.coef_[category_id])
  feature_names = np.array(tfidf.get_feature_names())[indices]
  unigrams = [v for v in reversed(feature_names) if len(v.split(' ')) == 1][:N]
  bigrams = [v for v in reversed(feature_names) if len(v.split(' ')) == 2][:N]
  print("# '{}':".format(category))
  print("  . Top unigrams:\n       . {}".format('\n       . '.join(unigrams)))
  print("  . Top bigrams:\n       . {}".format('\n       . '.join(bigrams)))
report = metrics.classification_report(y_test, y_pred, target_names=df['category'].unique())
print(report)
def report2dict(cr):
    # Parse rows
    tmp = list()
    for row in cr.split("\n"):
        parsed_row = [x for x in row.split("  ") if len(x) > 0]
        if len(parsed_row) > 0:
            tmp.append(parsed_row)
    
    # Store in dictionary
    measures = tmp[0]

    D_class_data = defaultdict(dict)
    for row in tmp[1:]:
        class_label = row[0]
        for j, m in enumerate(measures):
            D_class_data[class_label][m.strip()] = float(row[j + 1].strip())
    return D_class_data





report2dict(classification_report(y_test, y_pred, target_names=df['category'].unique()))
dataframe=pd.DataFrame(report2dict(classification_report(y_test, y_pred, target_names=df['category'].unique()))).T
dataframe.to_csv('/home/shiva/Sirius/CSV/classification_report.csv',index = False, index_label= str, header = True)
data = pd.read_csv('/home/shiva/Sirius/CSV/classification_report.csv')

# target_names = ['Correspondence','Assignments, deeds, conveyances','Leases','Division orders','Contract','Production-Operation report','Other']
# data.insert(0, "Class Labels", target_names)

data.insert(4, "Training_Samples", y_train.value_counts()) 
data.insert(5, "Testing_Samples", y_test.value_counts())
data.to_csv('/home/shiva/Sirius/CSV/report.csv', index = False, index_label=str, header = True)
