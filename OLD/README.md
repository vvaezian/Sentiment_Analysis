<details><summary>Summary</summary>
    
([source](https://monkeylearn.com/sentiment-analysis/))
- Sentiment analysis, just as many other NLP problems, can be modeled as a classification problem where two sub-problems must be resolved:  

    - Subjectivity classification: Fact or opinion
    - Polarity classification: positive, negative or neutral.

- Opinions or facts can be Direct/Comparative, Explicit/Implicit, their scope can be document_level/sentence_level/sub-sentence level.  
- SA tools can focus on polarity (good), feelings (happy) or intentions (interested). They may use fine-grained polarity (very good) and may identify which type of feeling a polarity has (negative and sad).  
- Usually we are interested in not polarity, but also which specific aspect or feature is being judged (E.x. Location of a store may be bad but its food quality be good).
- Algorithms can be rule-based (i.e. hand-crafted) or ML-based or hybrid.  
- The ML-based approach involves two major steps: Feature extraction and classification step. 
    - Feature extraction is to transform the text into a numerical representation, usually a vector. This is also called *text vectorization*. The classical approaches have been bag-of-words (we have a dictionary and count occurance of each word in the sentence and use that number in the vector) or bag-of-ngrams (similar to bag-of-words but now we consider n-tuples of connected words).
    - The classification step usually involves a statistical model like Naïve Bayes (works particularly well with NLP problems), Logistic Regression, Support Vector Machines, or Neural Networks.
- Among the challenges of SA are subjectivity and tone, context-based polarity, irony and sarcasm, emoji's, defining 'neutral' (e.x. objective text [the bag is red] and irrelevant data).
- Since in NLP, number of features is less, we should inspect the errors that classifier makes to improve the model by adding more features. We can repeat this process, but each time we should select a different eval-set/training split, to avoid overfitting. 
</p></details>
    

<details><summary>Libraries for Extracting Data</summary>
    
### Pattern ([source](https://www.clips.uantwerpen.be/pages/pattern-web))
Pattern is a web mining module for Python. Here we only focus in `pattern.web`. For other modules see [here](https://www.clips.uantwerpen.be/pattern).
```Python
from pattern.web import URL, extension
url = URL('http://www.clips.ua.ac.be/media/pattern_schema.gif')
f = open('test' + extension(url.page), 'wb') # save as test.gif
f.write(url.download())
f.close()

from pattern.web import download
html = download('http://www.clips.ua.ac.be/', unicode=True)
# The plaintext() function removes HTML formatting. It has many options. See the source for more details.

from pattern.web import Google, plaintext
engine = Google((license=None, throttle=0.5, language=None)  # Google, Yahoo, Bing, DuckDuckGo, Twitter, Facebook, Wikipedia, Wiktionary, Wikia, DBPedia, Flickr and Newsfeed.
for result in engine.search('"John Doe"'): # .search(query, type = SEARCH/IMAGE/NEWS, start = 1, count = 10, size = None/TINY/SMALL/MEDIUM/LARGE (for images), cached = True)
  print(plaintext(result.text)) # .text: summary, .url, .title, .language, .author, .date: for news items and images

# Twitter Search
# Since new tweets become available more quickly than we can query pages (~8700 tweets/second), the best way to get a continuous chunk of tweets is to pass the last seen tweet id:
from pattern.web import Twitter
t = Twitter()
i = None
for j in range(3):
  for tweet in t.search('win', start=i, count=10):
    print(tweet.text)
    print()
    i = tweet.id
  print('--------')

# Twitter Streaming
import time
from pattern.web import Twitter
s = Twitter().stream('#win')
while True:
  time.sleep(1)
  s.update(bytes=1024)
  print(s[-1].text if s else '')

# PDF Parser (since it relies on PDFMiner, only works with Python 2)
# The PDF object (based on PDFMiner) parses the source text from a PDF file:
# *** haven't got this to work properly yet ***
from pattern.web import URL, PDF
url = URL('http://.../name.pdf')
pdf = PDF(url.download())
print(pdf.string)
````


### textract ([Source](https://textract.readthedocs.io/en/stable/index.html))
Getting text out of PDF's, images, Word document, ... .  
Works only with local files.

#### Installation
````BASH
$sudo apt-get install python-dev libxml2-dev libxslt1-dev antiword unrtf poppler-utils pstotext tesseract-ocr \
flac ffmpeg lame libmad0 libsox-fmt-mp3 sox libjpeg-dev swig libpulse-dev
$pip install textract
````
#### Usage
````BASH
import textract
text = textract.process("path/to/local/file")
````

### pdftotext
with [`pdftotext`](https://github.com/jalan/pdftotext) we have the option to extract certain pages.  
Works only with local files.
</details>


<details><summary>Libraries for model building</summary>
    
### Scikit-Learn ([Source](https://www.twilio.com/blog/2017/12/sentiment-analysis-scikit-learn.html))
````Python
from sklearn.feature_extraction.text import CountVectorizer

data = []
data_labels = []
with open("pos_tweets.txt", encoding="utf8") as f:
    for i in f: 
        data.append(i) 
        data_labels.append('pos')  # label can be anything (doesn't have to be 'pos' and 'neg')
with open("neg_tweets.txt", encoding="utf8") as f:
    for i in f: 
        data.append(i)
        data_labels.append('neg')
        
vectorizer = CountVectorizer(token_pattern = '[a-z][a-z]+')  # tokens need to be at least 2 letters (no numbers or ...)
print(vectorizer) # see the default options

features = vectorizer.fit_transform(data)
print('Number of Features: ', len(vectorizer.get_feature_names()))
print('Some of features: ', vectorizer.get_feature_names()[:10])
features_nd = features.toarray()
print('Dimentions of features matrix: ', features_nd.shape) # we have 2004 lines of data (tweets)
print(features_nd[:5])

from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val  = train_test_split(features_nd, data_labels, test_size=0.2, random_state=42)

from sklearn.linear_model import LogisticRegression
log_model = LogisticRegression(solver='liblinear')
log_model.fit(X=X_train, y=y_train)
y_pred = log_model.predict(X_val)

from sklearn.metrics import accuracy_score
print(round(accuracy_score(y_val, y_pred), 3))    # 0.830
````
#### Parameter-tuning
````Python
best_acc = 0
best_params = {}

from sklearn.model_selection import ParameterGrid
param_grid = { 'loss' : ['hinge', 'log', 'modified_huber', 'squared_hinge', 'perceptron'], 
            # 'hinge' gives a linear SVM, 'log' gives Logistic Regression, ...
            'alpha' : [.001, .0005, .0001],
            'max_iter' : [10, 50],
            }
for params in ParameterGrid(param_grid):
    pipe = Pipeline([#('tfidf', TfidfTransformer()),
                        ('clf-svm', SGDClassifier(**params, penalty='l2', random_state=42)),])
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_val)
    acc = round(accuracy_score(y_val, y_pred), 4)
    print(params, acc)
    if acc > best_acc:
        best_params = params
        best_acc = acc
print('\n', best_params, best_acc)
````
#### Finding Misclassified Items
````Python
def find_misclassified_samples(X_val, y_val, y_pred, data, vectorizer):
  # 'data' is vectorized and then split into train and validation (X_val) sets. 
  # y_val is the actual label for X_val. y_pred is the predicted label for X_val.
  # vectorizer is Scikit's CountVectorizer instance.
  
  misclassified_transformed_data = []
  
  for index, polarity in enumerate(y_pred):
    if y_val[index] != polarity:
      misclassified_transformed_data.append((X_val[index], polarity))

  for item in misclassified_transformed_data:
    features_list = vectorizer.inverse_transform(item[0])
    features = features_list[0]
    for line in data:
      flag = True
      for feature in features:
        if feature not in line.lower():
          flag = False
          break
      if flag:
        print('classifiend incorrectly as', item[1], line.rstrip('\n'))
````

### NLTK

````Python
import nltk
from nltk import word_tokenize
from collections import Counter
import random

data_labels = []
data = []
with open('pos_tweets.txt', encoding='utf8') as f:
  pos = f.read()
  f.seek(0)
  for i in f:
    data.append(i)
    data_labels.append('pos')
with open('neg_tweets.txt', encoding='utf8') as f:
  neg = f.read()
  f.seek(0)
  for i in f:
    data.append(i)
    data_labels.append('neg')

tokens = word_tokenize(pos) + word_tokenize(neg)
print('# of tokens: ', len(tokens))
from nltk.corpus import stopwords 
stop_words = set(stopwords.words('english')) 
filtered_tokens = [w for w in tokens if w not in stop_words] 
print('# of filtered tokens: ', len(filtered_tokens))

from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

from nltk.corpus import stopwords 
stop_words = set(stopwords.words('english')) 
filtered_tokens = [w for w in tokens if w not in stop_words] 

for n in [1150]: # out of all tokens we select n of the most comon of them
  most_common_tokens = {i[0] for i in Counter(lemmatizer.lemmatize(w.lower()) for w in filtered_tokens).most_common(n) if len(i[0]) > 1}
  #print('number of chosen distinct tokens: ', len(most_common_tokens))
  
  # creating featureset
  for index, item in enumerate(data):
    data[index] = [{word: True for word in word_tokenize(item) if word.lower() in most_common_tokens}, data_labels[index]]

  random.seed(43)
  # split data for validation
  random.shuffle(data)
  training = data[:int((.8)*len(data))]
  val = data[int((.8)*len(data)):]
  
  from nltk.classify import NaiveBayesClassifier
  classifier = NaiveBayesClassifier.train(training)
  
  preds = classifier.classify_many([fs for (fs, l) in val])
  from nltk.classify.util import accuracy
  print(n, ': ', round(accuracy(classifier, val), 3))  # accuracy ranges around .79 to .88 depending on the seed
  
  classifier.show_most_informative_features()
  print(classifier.classify(featureset("Cats are awesome!")))  # pos
````

### TextBlob ([source](https://textblob.readthedocs.io/en/dev/))    

"TextBlob stands on the giant shoulders of NLTK and pattern, and plays nicely with both."  
- TextBlob is pre-trained and performs not too bad in general. It had71% accuracy on the small tweeter data (although if we include only those polarities `p` that `abs(p) > .9` then the accuracy becomes 96%). For example polarity of 'glorious' is 0 and polarity of 'Thnx man, that means an awful lot to me' is -1.
- Polarity of 'hater' is 0 and polarity of 'hate' is -0.8. Couldn't find a way to get 'hate' from 'hater'. Stemming and lemmatization methods I found either return 'hater' or 'hat'.
```Python
from textblob import TextBlob

text = '...'  
blob = TextBlob(text)  # textblobs are like Python strings; they can be sliced, concated and common string methods can be applied on them

for sentence in blob.sentences:
    print(sentence.sentiment.polarity)  # sentence.sentiment is a tuple of polarity (in [-1, 1]) and subjectivity (in [0,1], 1 being completely subjective).

blob.words
blob.sentences  # list of sentences. Sentences are seperated by periods '.'
blob.sentences[0].words
blob.tags           # List of part of speech tags. E.x. ('The', 'DT'). Full list: https://www.clips.uantwerpen.be/pages/mbsp-tags)
blob.noun_phrases   # list of noun phrases like 'ultimate movie monster'.

blob.words.lemmatize() # simplifies and uses one format for different variations. We can use WordNet POS like word.lemmatize('v') which outputs 'go' given 'went'.
blob.words.pluralize() # 'study' -> 'studies' (applies on everything: 'the' -> 'thes' (but transforms "'s" to "s'"))
blob.words.singularize() # 'studies' -> 'study', "'s" -> "'"
blob.correct() # makes spelling corrections. Currently works in Python < 3.7

from textblob import Word
Word('driught').spellcheck()  # [('draught', 0.875), ('drought', 0.125)]

blob.words.count('hi', case_sensitive=True) # default False
blob.noun_phrases.count('hi there', case_sensitive=True) # default False

# Language translation and detection is powered by the Google Translate API.
blob.translate(from_lang="en", to="es")  # translates to the specified language. If source is not specified it tries to auto-detect.
blob.detect_language()

# n-grams return a list of tuples of n successive words.
blob = TextBlob("Now is better than never.")
blob.ngrams(n=3)  # [WordList(['Now', 'is', 'better']), WordList(['is', 'better', 'than']), WordList(['better', 'than', 'never'])]
```
[Creating a custom sentiment analyzer](https://textblob.readthedocs.io/en/dev/classifiers.html#classifiers)

</details>



  
### To-Do
- [ ] All-caps tweets are usually negative. But converting to lower-case we lose this info.
- [x] Multi_class Logistic regression
- [ ] Explore misclassfications
- [x] Correct the `Compress` function
- [x] Create simple baseline rule_based model
- [ ] Reconsider applying Lemmatization, Stemming and POS tagging to the new cleaned dataset
- [x] Test performance with cleaned stopwords

### Rule_based
- Pos: http://ptrckprry.com/course/ssd/data/positive-words.txt
- Neg: http://ptrckprry.com/course/ssd/data/negative-words.txt

### Remarks
- "A lemmatizer needs a part of speech tag to work correctly. This is usually inferred using the `pos_tag` nltk function before tokenization."
- Consider using stemming before or after lemmatization
- [The paper](https://s3.amazonaws.com/academia.edu.documents/34632156/Twitter_Sentiment_Classification_using_Distant_Supervision.pdf?response-content-disposition=inline%3B%20filename%3DTwitter_Sentiment_Classification_using_D.pdf&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAIWOWYYGZ2Y53UL3A%2F20190620%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20190620T213431Z&X-Amz-Expires=3600&X-Amz-SignedHeaders=host&X-Amz-Signature=99677c7040f123dec6fff770d493bda4218015f4c24fd3d8d676a8eef18c55b5) that produced the dataset. It labelled tweets based on :-) and :-( symbols. Its best accuracy is 83%. 

### Services
- Doing sentiment analysis on brands
- Exploring the correlation between a brand name and demographics, weather, ...

### Research Topics
- Look into LDA ([Latent Dirichlet Allocation](https://en.wikipedia.org/wiki/Latent_Dirichlet_allocation))

### Tools and Libraries to Explore
- [Spacy](https://en.wikipedia.org/wiki/SpaCy)
