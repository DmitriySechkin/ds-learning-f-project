import time

import numpy as np
from gensim.models import FastText, Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer
from bs4 import BeautifulSoup
from abc import ABCMeta, abstractmethod, abstractproperty


class Vectorizer():
    __metaclass__ = ABCMeta

    def __init__(self, processor):
        self._processor = processor

    @abstractmethod
    def fit(self, docs):
        pass

    @abstractmethod
    def vectorize(self, docs):
        pass

    def process(self, docs):
        pass


class TFIDFVectorizer(Vectorizer):
    def __init__(self, processor):
        super().__init__(processor)
        self.vectorizer = TfidfVectorizer()

    def fit(self, docs):
        flat_docs = self.process(docs)
        return self.vectorizer.fit_transform(flat_docs)

    def vectorize(self, docs):
        flat_docs = self.process(docs)
        return self.vectorizer.transform(flat_docs)

    def process(self, docs):
        # нужен список строк, лемматизация
        return [' '.join(self._processor.spacy_tokenize(doc)) for doc in docs]


class TextProcessor:
    def __init__(self):
        import re
        import spacy

        self.nlp = spacy.load("ru_core_news_sm")
        self.stop_words = self.nlp.Defaults.stop_words
        self.most_counter_words = ['компания', 'мы', 'год', 'работать', 'день', 'команда', 'месяц']

    def spacy_tokenize(self, text):
        doc = self.nlp(text)

        tokens = []

        for token in doc:
            if token.is_space:
                continue

            if token.lemma_ != '':
                tokens.append(token.lemma_)

        return tokens

    def remove_most_counter_words(self, data):
        return [i for i in data if i not in self.most_counter_words]

    def preprocess_text(self, text, is_stop_words=True):
        import re

        words = []

        text = re.sub(r'(ООО|ПАО|ЗАО|ОАО|АО)([\"«])', r'\1 \2', text)
        text = re.sub(r'[^А-ЯЁа-яёA-Za-z\s]', '', text)

        doc = self.nlp(text)
        ignored_tokens = set()

        for ent in doc.ents:
            if ent.label_ in ["ORG", "LOC"]:
                ignored_tokens.update([token.text for token in ent])

        for word in doc:
            if word.is_punct or word.is_space:
                continue

            if word.text in self.stop_words and is_stop_words:
                continue

            if word.text in ignored_tokens:
                continue

            words.append(word.text.lower())

        text = ' '.join(words)

        return text

    def clean_html_text(self, text):
        if not text:
            return ''

        soup = BeautifulSoup(text, 'html.parser')

        for script in soup(["script"]):
            script.extract()

        for style in soup.find_all('style'):
            style.extract()

        clean_text = soup.get_text(separator=' ', strip=True)

        return clean_text


class Vectorizer():
    __metaclass__ = ABCMeta

    def __init__(self, processor):
        self._processor = processor

    @abstractmethod
    def fit(self, docs):
        pass

    @abstractmethod
    def vectorize(self, docs):
        pass

    def process(self, docs):
        pass


class TFIDFVectorizer(Vectorizer):
    def __init__(self, processor):
        super().__init__(processor)
        self.vectorizer = TfidfVectorizer()

    def fit(self, docs):
        start = time.time()
        flat_docs = self.process(docs)
        self.vectorizer.fit_transform(flat_docs)
        end = time.time() - start
        return end

    def vectorize(self, docs):
        flat_docs = self.process(docs)
        return self.vectorizer.transform(flat_docs)

    def process(self, docs):
        return [' '.join(doc) for doc in docs]


class Word2VecVectorizer(Vectorizer):
    def __init__(self, processor, size=100, window=5, min_count=1, workers=2):
        super().__init__(processor)
        self.size = size
        self.window = window
        self.min_count = min_count
        self.workers = workers
        self.model = None

    def fit(self, docs):
        start = time.time()
        self.model = Word2Vec(docs, vector_size=self.size, min_count=self.min_count)
        end = time.time() - start
        return end

    def vectorize(self, docs):
        vectors = []
        #         print(docs[0])
        for words in docs:
            words_vecs = [self.model.wv[word] for word in words if word in self.model.wv]

            if len(words_vecs) == 0:
                words_vecs = [np.zeros(self.size)]

            vectors.append(np.sum(words_vecs, axis=0))

        return np.array(vectors)


class FastTextVectorizer(Vectorizer):
    def __init__(self, processor, size=100, window=5, min_count=1, workers=2):
        super().__init__(processor)
        self.size = size
        self.window = window
        self.min_count = min_count
        self.workers = workers
        self.model = None

    def fit(self, docs):
        start = time.time()
        self.model = FastText(
            docs,
            vector_size=self.size,
            window=self.window,
            min_count=self.min_count,
            workers=self.workers
        )
        end = time.time() - start
        return end

    def vectorize(self, docs):
        return np.array([np.mean([self.model.wv[word] for word in words if word in self.model.wv]
                                 or [np.zeros(self.size)], axis=0)
                         for words in docs])


# class BERTVectorizer:
#     def __init__(self, model_name='bert-base-multilingual-cased', batch_size=32):
#         self.tokenizer = BertTokenizer.from_pretrained(model_name)
#         self.model = BertModel.from_pretrained(model_name)
#         self.batch_size = batch_size
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         self.model.to(self.device)

#     def transform(self, texts):
#         self.model.eval()
#         with torch.no_grad():
#             embeddings = []
#             for i in range(0, len(texts), self.batch_size):
#                 batch = texts[i:i+self.batch_size]
#                 encoded_input = self.tokenizer(batch, return_tensors='pt', padding=True, truncation=True).to(self.device)
#                 outputs = self.model(**encoded_input)
#                 embeddings.append(outputs[1].cpu().numpy())
#             return np.vstack(embeddings)

# Использование класса
# bert_vectorizer = BERTVectorizer(batch_size=16)  # Можете настроить размер батча в зависимости от доступной памяти GPU
# vectors = bert_vectorizer.transform(["Пример текста", "Еще один текст"])


processor = TextProcessor()
