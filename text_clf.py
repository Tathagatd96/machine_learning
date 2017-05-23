from sklearn.datasets import fetch_20newsgroups
import sklearn.datasets
from sklearn.feature_extraction.text import CountVectorizer,CharNGramAnalyzer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier

from sklearn.svm.sparse import LinearSVC
import numpy as np

sklearn.datasets.load_files("C://Users/Tathagat Dasgupta/Desktop/ML Project/20news-18828")

categories=['alt.atheism','soc.religion.christian','comp.graphics','sci.med']
print "hello"

twenty_train=fetch_20newsgroups(subset='train',categories=categories,shuffle=True,random_state=42)

#twenty_train.target_names=['alt.atheism','comp.graphics','sci.med','soc.religion.christian']

print len(twenty_train.data)

print("\n".join(twenty_train.data[0].split("\n")[:3]))

print(twenty_train.target_names[twenty_train.target[0]])

print(twenty_train.target[:10])

for t in twenty_train.target[:10]:
    print(twenty_train.target_names[t])
    
#Preprocessing

#Tokenizing text

count_vect=CountVectorizer()
X_train_counts=count_vect.fit_transform(twenty_train.data)

print(X_train_counts.shape)

print(count_vect.vocabulary_.get(u'algorithm')) 
#vocabulary contains={word:frequency in corpus}

#tf-idf
tfidf_transformer=TfidfTransformer()

X_train_tfidf=tfidf_transformer.fit_transform(X_train_counts)
print(X_train_tfidf.shape)


#Classifier Training


clf=MultinomialNB().fit(X_train_tfidf,twenty_train.target)

docs_new=['God is love','OpenGL on the GPU is fast']
X_new_counts=count_vect.transform(docs_new)
X_new_tfidf=tfidf_transformer.transform(X_new_counts)

predicted=clf.predict(X_new_tfidf)

for doc,category in zip(docs_new,predicted):
    print('%r=>%s'%(doc,twenty_train.target_names[category]))
    
    
#Building a pipeline

text_clf=Pipeline([('vect',CountVectorizer()),('tfidf',TfidfTransformer()),('clf',MultinomialNB())])

text_clf=text_clf.fit(twenty_train.data,twenty_train.target)


#Performance on test set

twenty_test=fetch_20newsgroups(subset='test',categories=categories,shuffle=True,random_state=42)
doc_test=twenty_test.data
predicted=text_clf.predict(doc_test)
print "Classifier Accuracy:"
print(np.mean(predicted==twenty_test.target))

#SVM Implementation

text_clf=Pipeline([('vect',CountVectorizer()),('tfidf',TfidfTransformer()),('clf',SGDClassifier(loss='hinge',alpha=1e-3,n_iter=5,random_state=42))])
text_clf.fit(twenty_train.data,twenty_train.target)
predicted=text_clf.predict(doc_test)

print "SVM Accuracy:"
print(np.mean(predicted==twenty_test.target))


#Language Identifier




class LowerCasePreprocessor(object):
    """Simple preprocessor that should be available by default"""

    def preprocess(self, unicode_content):
        return unicode_content.lower()

    def __repr__(self):
        return "LowerCasePreprocessor()"

analyzer1 = CharNGramAnalyzer(
    min_n=1,
    max_n=3,
    preprocessor=LowerCasePreprocessor(),
)

# Build a vectorizer / classifier pipeline using the previous analyzer
clf = Pipeline([
    ('vec', CountVectorizer(analyzer=analyzer1)),
    ('tfidf', TfidfTransformer(use_idf=False)),
    ('clf', LinearSVC(loss='l2', penalty='l1', dual=False, C=100)),
])

# Fit the pipeline on the training set
clf.fit(twenty_train.data,twenty_train.target)

# Predict the outcome on the testing set
y_predicted = clf.predict(doc_test)


# Predict the result on some short new sentences:
sentences = [
    u'This is a language detection test.',
    u'Ceci est un test de d\xe9tection de la langue.',
    u'Dies ist ein Test, um die Sprache zu erkennen.',
]
predicted = clf.predict(sentences)

for s, p in zip(sentences, predicted):
    print u'The language of "%s" is "%s"' % (s, twenty_train.target_names[p])
