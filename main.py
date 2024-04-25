import joblib
import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.openapi.utils import get_openapi
from fastapi.openapi.docs import get_redoc_html, get_swagger_ui_html

from adapters import SqlLitePdAdapter
from models import TFIDFVectorizer, TextProcessor, processor, FastTextVectorizer, Word2VecVectorizer
from gensim.models import Word2Vec, FastText
from typing import List
from sklearn.metrics.pairwise import cosine_similarity
import sqlite3
import spacy

app = FastAPI()

# tfidf_vectorizer = joblib.load(r'tfidf_vectorizer.joblib')
print('загрузка модели word2vec')
word2vec_model = Word2Vec.load('word2vec_model')

sql_adapter = SqlLitePdAdapter('', 'vacancies', 'vacancies')

print('получение векторов в БД')
vacancies = sql_adapter.read_fetchall(target_columns=['id', 'name', 'description_all', 'word2vec_v'])


word2vec_vectorizer = Word2VecVectorizer(processor)
word2vec_vectorizer.model = word2vec_model

vectors = np.array([vacancy[3].split(',') for vacancy in vacancies])
vacancy_vecs = np.vstack(vectors)

print('запрос обработан успешно')


@app.get("/openapi.json")
async def get_open_api_endpoint():
    return get_openapi(title="Your API title", version="1.0.0", routes=app.routes)


@app.get("/docs", include_in_schema=False)
async def get_documentation():
    return get_swagger_ui_html(openapi_url="/openapi.json", title="docs")


@app.get("/hello")
async def hello():
    return "HELLO!"


@app.post("/predict_word2vec")
async def predict_word2vec(docs: List[str]):
    return [word2vec_model.wv[doc] for doc in docs]


# @app.post("/predict_fasttext")
# async def predict_fasttext(docs: List[str]):
#     vectors = [fasttext_model.wv[doc].tolist() for doc in docs]
#     return vectors


# @app.post("/predict_tfidf")
# async def predict_tfidf(docs: List[str]):
#     return tfidf_vectorizer.vectorize(docs).toarray().tolist()


@app.post("/get_vacancies")
async def get_vacancies(resume_text: str, number_relevants: int):
    result = []

    print('предобработка текста резюме')
    text = processor.preprocess_text(resume_text)
    text_toc = processor.spacy_tokenize(text)
    text_toc = processor.remove_most_counter_words(text_toc)

    print('преобразование текста резюме в вектор')
    resume_vecs = word2vec_vectorizer.vectorize([text_toc])

    print('вычисляем косинусную меру схожести')
    cos_sim = cosine_similarity(resume_vecs, vacancy_vecs)

    top_10_preds = np.argsort(-cos_sim, axis=1)[:, : number_relevants]

    print('получение данных по релевантным вакансиям')
    for ind in top_10_preds[0]:
        vacancy = vacancies[ind]
        id_ = vacancy[0]
        title = vacancy[1]
        description = vacancy[2]

        similarity_score = cos_sim[0][ind]

        result.append({
            "title": title,
            "url": f'https://nn.hh.ru/vacancy/{id_}',
            "description": description,
            "score": str(similarity_score)
        })

    return {"data": result}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)

