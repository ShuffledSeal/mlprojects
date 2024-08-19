import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv).
from gensim.models import Word2Vec
import nltk

df = pd.read_csv('Rejection Data - Sheet1.csv')
print(df.tail())


import string
import matplotlib.pyplot as plt
import nltk


# For the purposes of this assessment, all text processing actions in this cell have been removed except the replacement of newline characters by space characters

# convert email text to lowercase
#df.Email = df.Email.apply(lambda x: x.lower())
# remove punctuation
#df.Email = df.Email.apply(lambda x: x.translate(str.maketrans('','', string.punctuation)))
# remove numbers
#df.Email = df.Email.apply(lambda x: x.translate(str.maketrans('','','1234567890')))
# remove newline tags
df.Email = df.Email.apply(lambda x: x.translate(str.maketrans('\n',' ')))
df.tail()

# Add a column 'Tokens' to df, to hold the email contents as a list of tokens
df['Tokens'] = [nltk.word_tokenize(e) for e in df.Email]

print(df.Tokens[0]) 

#Q1

df['Length'] = df['Tokens'].apply(len)

#Q2a

groups = df.groupby('Status')['Length']


labels, data = zip(*[(label, d) for label, d in groups])


plt.figure(figsize=(10, 6))

plt.boxplot(data, labels=labels)
plt.title('Distribution of Email Lengths by Status')
plt.xlabel('Status')
plt.ylabel('Length of Email (number of tokens)')
plt.show()

#2b
'''
Box plots show outliers as well as median and the quartiles of the length making it easier to compare the numerical data between
the two email types. I am visualising numerical data and box plots help visualise it in a more orderly manner to compare it in.
The patterns seen show that reject emails are very close together bcs the tail and the head of the box plot are very close to each other
compared to non-reject emails. The length of the emails are slightly different with reject emails having about 10 less tokens in general.
Outliers and the spread of non-reject shows how varied the data for non-reject is.
'''

#q3
df['LexRich'] = df['Tokens'].apply(lambda tokens: len(set(token.lower() for token in tokens)) / len(tokens) if tokens else 0)

#q4a
lex_reject = df[df.Status == 'reject']['LexRich'].mean()
lex_not_reject = df[df.Status != 'reject']['LexRich'].mean()

print(f"Mean Lexical Richness for 'reject' emails: {lex_reject}")
print(f"Mean Lexical Richness for 'not_reject' emails: {lex_not_reject}")

#q4b
'''
The richness of reject and not_reject are very close to each other showing that the depth of the vocabulary is the same.
Also shows that both rejection and non_rejection being close to each other that the vocabulary used in a rejection email is as relatable
as non-rejection email which is done specially to lessen the blow of being rejected
'''


#q5

sentences = df['Tokens'].tolist()
model = Word2Vec(sentences=df['Tokens'], vector_size=100, window=5, min_count=1, workers=4)

similar_tokens = model.wv.most_similar('developer', topn=20)
for token, similarity in similar_tokens:
    print(f"{token}: {similarity}")

#from sklearn.feature_extraction.text import CountVectorizer
#from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer
#from sklearn.preprocessing import LabelEncoder
#from sklearn.naive_bayes import MultinomialNB
#from sklearn.model_selection import train_test_split
#from sklearn.metrics import confusion_matrix
#from sklearn.model_selection import cross_val_score
#from sklearn.metrics import classification_report

# pull data into vectors to create collection of text/tokens
#vectorizer = CountVectorizer()
#x = vectorizer.fit_transform(df.Email)

#encoder = LabelEncoder()
#y = encoder.fit_transform(df.Status)

# split into train and test sets
#x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

# Shape of sets
#print(x_train.shape)
#print(y_train.shape)
#print(x_test.shape)
#print(y_test.shape)

#%%time
#nb = MultinomialNB()
#nb.fit(x_train, y_train)


#def predict_reject(email):
#    category_names = {'reject':'reject','not_reject':'not-reject'}
#    cod = nb.predict(vectorizer.transform([email]))
#    return category_names[encoder.inverse_transform(cod)[0]]

#print(predict_reject('Unfortunately we will not be moving forward'))
#print(predict_reject('I found some job listings you may be interested in'))
#print(predict_reject('We were very fortunate to have a strong group of applicants to consider for this role and have recently filled this position. Unfortunately, because this role is no longer available, we will not be moving forward with your application.'))
#print(predict_reject(''))

#from sklearn.linear_model import LogisticRegression
#from sklearn.multiclass import OneVsRestClassifier

# Init the classfifier
#clf = OneVsRestClassifier(LogisticRegression())

# Fit classifier to training data
#clf.fit(x_train, y_train)

# Print accuracy
#print(f'Accuracy: {clf.score(x_test, y_test)}')

#x_test_clv_pred = clf.predict(x_test)
#confusion_matrix(y_test, x_test_clv_pred)
#print(classification_report(y_test, x_test_clv_pred, target_names=encoder.classes_))