'''
A script to learn LDA topics from raw Excel files.

'''

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from gensim import matutils
from gensim.corpora import Dictionary
from gensim.models.ldamodel import LdaModel
# in case the dataset is very large, distributed implementation: 
# but alpha and eta need to be provided explicitly
from gensim.models.ldamulticore import LdaMulticore 
from gensim.models.hdpmodel import HdpModel
from nltk import word_tokenize          
from nltk.stem import WordNetLemmatizer 
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import pyLDAvis
import pyLDAvis.gensim


path = str(input('Please provide the full path to the Excel file.\n'))
sheet = str(input('Please provide the sheetname where the data is stored. It should only have the column with raw text.\n'))
num_topics = int(input('Please provide the number of topics you would like the algorithm to discover.\n'))

class topic_model(object):
	def __init__(self, path, sheet, num_topics, lemmatise=True, 
		stem=False, alpha=None, eta=None, custom_stop_words=None, 
		**kwargs):
		'''
		if stemming/lemmatising needs to be applied, add: tokenizer=LemmaTokenizer()
		to the list of arguments
		'''
		if alpha is not None: 
			self.alpha = alpha 
		# learns an asymmetric prior directly from your data
		else: 
			self.alpha = 'auto'
		if eta is not None: 
			self.eta = eta 
		# learns an asymmetric prior directly from your data
		else: 
			self.eta = 'auto'

		self.num_topics = num_topics

		df = pd.read_excel(path, sheetname=sheet)
		try:
			df.columns = ['text']
		except ValueError:
			print('Error! There are more than one column in the spreadsheet.')

		vectorizer = CountVectorizer(stop_words='english', **kwargs)
		self.dtm = vectorizer.fit_transform(df['text'])
		#self.stop_words = vectorizer.stop_words_
		#self.vocab = dict((value, key) for key, value in vectorizer.vocabulary_.items())

		stop_words = set(stopwords.words('english'))
		stop_words.update(['.', ',', '"', "'", '?', '!', ':', ';', '(', ')', '[', ']', '{', '}', '..', '...'])
		if type(custom_stop_words) == 'list' and custom_stop_words is not None:
			stop_words.update(custom_stop_words)

		lemmatiser = WordNetLemmatizer()
		p_stemmer = PorterStemmer()
		docs = []
		for doc in df['text']:
			raw = doc.lower()
			if lemmatise:
				tokens = [lemmatiser.lemmatize(t) for t in word_tokenize(raw)] # lemmatisation of tokens
			else:
				tokens = word_tokenize(raw)
			stopped_tokens = [i for i in tokens if i not in stop_words]
			if stem:
				stopped_tokens = [p_stemmer.stem(i) for i in stopped_tokens]
			docs.append(stopped_tokens)
		self.vocab = Dictionary(docs)
		self.corpus = [self.vocab.doc2bow(text) for text in docs]

		self.num_docs, self.vocab_size = (len(docs), len(self.vocab.keys()))
		print('The excel file has been successfully transformed into a document-word matrix.')
		print('%d documents and %d unique words (terms) were identified in the corpus.' %(self.num_docs, self.vocab_size))
		print('The mapping between words and the matrix indices is stored in *your_instance_name*.vocab')
		print('The matrix is stored in *your_instance_name*.dtm')
		print('To access it, you can use .dtm.toarray()')

	def fit_lda(self, iterations=100):
		print('Fitting LDA')
		self.model = LdaModel(self.corpus, 
			num_topics=self.num_topics,
			id2word=self.vocab, # id2word=dict([(i, s) for i, s in enumerate(vectorizer.get_feature_names())]
			alpha=self.alpha,
			eta=self.eta,
			iterations=iterations)
		print('Learning completed successfully!')

	def visualise(self, **kwargs):
		'''
		API documentation: http://pyldavis.readthedocs.io/en/latest/modules/API.html
		'''
		
		pyLDAvis.enable_notebook()
		vis_data = pyLDAvis.gensim.prepare(self.model, self.corpus, self.vocab, sort_topics=False, **kwargs)
		pyLDAvis.display(vis_data)

		pyLDAvis.show(vis_data)

	def fit_hdp(self, iterations=100):
		'''
		The Hierarchical Dirichlet Process: learns the number of topics automatically. T is the max number of topics allowed
		'''
		self.model = HdpModel(matutils.Sparse2Corpus(self.dtm), self.vocab, T=50)

class LemmaTokenizer(object):
	'''
	WordNet Lemmatizer

    Lemmatize using WordNet's built-in morphy function.
    Returns the input word unchanged if it cannot be found in WordNet.

    source: http://www.nltk.org/_modules/nltk/stem/wordnet.html
	'''
	def __init__(self):
		self.wnl = WordNetLemmatizer()
	def __call__(self, doc):
		return [self.wnl.lemmatize(t) for t in word_tokenize(doc)]

# instance = topic_model(path, sheet, num_topics)
# topicmodel.fit_lda()
# topicmodel.visualise()