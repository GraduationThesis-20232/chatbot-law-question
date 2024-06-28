import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pickle
import json
from text_untils import *
import torch
import pymongo
from bson import ObjectId
from flask import Flask, request, jsonify
from flask_cors import CORS
from pymongo import MongoClient

app = Flask(__name__)
CORS(app)

top_20_indices_title = []
top_20_indices_anwser = []
doc_scores = []
top_docs = []
ids_title = []
similarities_title = []
sentences_title = []
sentences_anwser = []
ids_anwser = []
similarities_anwser = []

def create_answer(row):
    conclusion_value = row['conclusion']

    if isinstance(conclusion_value, list):
        conclusion_value = ' '.join(conclusion_value)
    elif pd.isna(conclusion_value):
        conclusion_value = ''

    if pd.isna(row['quote']):
        return conclusion_value
    else:
        reference = row['reference'] if not pd.isna(row['reference']) else ''
        merged_quote = merge_quote(row['quote'])
        return f"{reference} {merged_quote}"

def merge_quote(quote):
    if isinstance(quote, dict):
        name = quote.get('name', ' ')
        content = ' '.join(quote.get('content', []))

        if name is None:
            name = ' '
        if content is None:
            content = ' '

        return f"{name}. {content}"
    else:
        return None

def get_selected_rows_from_mongodb(database_name, collection_name, top_docs):
    client = MongoClient('mongodb://localhost:27017/')
    db = client[database_name]
    collection = db[collection_name]

    doc_ids = [ObjectId(doc_dict['id']) for doc_dict in top_docs]

    cursor = collection.find({'_id': {'$in': doc_ids}})
    data = list(cursor)
    df = pd.DataFrame(data)

    # client.close()
    return df

def preprocess(query):
    query = clean_text(query)
    query = word_segment(query)
    query = remove_stopword(normalize_text(query))
    return query
def load_docs_from_file(file_path):
    docs = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            try:
                doc = json.loads(line)
                docs.append(doc)
            except json.JSONDecodeError:
                continue

    return docs

def handle_bm25(query):
    query_bm25 = preprocess(query)
    query_bm25 = query_bm25.split()

    with open('output/bm25/bm25plus.pkl', 'rb') as f:
        bm25plus = pickle.load(f)

    docs = load_docs_from_file('output/bm25/docs.jsonl')

    global top_docs
    top_docs = bm25plus.get_top_n(query_bm25, [doc for doc in docs], n=50)
    bm25_scores = bm25plus.get_scores(query_bm25)
    global doc_scores
    doc_scores = np.sort(bm25_scores)[::-1][:500]
    # for doc_dict, score in zip(top_docs, doc_scores):
    #     doc_id = doc_dict['id']
    #     doc_title = doc_dict['title']
    #     print(f"ID: {doc_id}, Document: {doc_title}, Score: {score}")

    selected_rows = get_selected_rows_from_mongodb('lawlaboratory', 'questions_cleaned', top_docs)

    return selected_rows

def handle_sbert_title(query, selected_rows):
    global sentences_title
    global ids_title

    for idx in selected_rows.index:
        sentences_title.append(selected_rows.at[idx, 'title'])
        ids_title.append(selected_rows.at[idx, '_id'])

    model_title = SentenceTransformer('keepitreal/vietnamese-sbert', device='cuda')
    embeddings_title = model_title.encode(sentences_title)

    query_sbert = preprocess(query)
    query_sbert = query_sbert.replace('_', ' ')
    query_embedding_title = model_title.encode([query_sbert], convert_to_tensor=True)
    query_embedding_title = query_embedding_title.cpu().numpy()
    global similarities_title
    similarities_title = cosine_similarity(query_embedding_title, embeddings_title)
    global top_20_indices_title
    top_20_indices_title = np.argsort(similarities_title[0])[::-1][:20]

    # for idx in top_50_indices_title:
    #     similarity_score = similarities_title[0][idx]
    #     similar_sentence = sentences_title[idx]
    #     doc_id = ids_title[idx]
    #     print(f"ID: {doc_id}, Similarity Score: {similarity_score:.4f}, Sentence: {similar_sentence}")

def handle_sbert_answer(query, selected_rows):
    global sentences_anwser
    global ids_anwser

    selected_rows['conclusion'] = selected_rows.apply(lambda row: merge_quote(row['quote']) if row['conclusion'] == [] else row['conclusion'], axis=1)
    selected_rows['answer'] = selected_rows.apply(create_answer, axis=1)

    for idx in selected_rows.index:
        sentences_anwser.append(selected_rows.at[idx, 'answer'])
        ids_anwser.append(selected_rows.at[idx, '_id'])

    query_sbert = preprocess(query)
    query_sbert = query_sbert.replace('_', ' ')

    model_anwser = SentenceTransformer('keepitreal/vietnamese-sbert', device='cuda')
    embeddings_anwser = model_anwser.encode(sentences_anwser)

    query_embedding_anwser = model_anwser.encode([query_sbert], convert_to_tensor=True)
    query_embedding_anwser = query_embedding_anwser.cpu().numpy()
    global similarities_anwser
    similarities_anwser = cosine_similarity(query_embedding_anwser, embeddings_anwser)
    global top_20_indices_anwser
    top_20_indices_anwser = np.argsort(similarities_anwser[0])[::-1][:20]

    # for idx in top_50_indices_anwser:
    #     similarity_score = similarities_anwser[0][idx]
    #     similar_sentence = sentences_anwser[idx]
    #     doc_id = ids_anwser[idx]
    #     print(f"ID: {doc_id}, Similarity Score: {similarity_score:.4f}, Sentence: {similar_sentence}")

def calculate_score():
    score_title_ids = []
    for idx in top_20_indices_title:
        score_title_ids.append({
            'id': ids_title[idx],
            'score': similarities_title[0][idx]
        })

    score_answer_ids = []
    for idx in top_20_indices_anwser:
        score_answer_ids.append({
            'id': ids_anwser[idx],
            'score': similarities_anwser[0][idx]
        })

    title_scores_by_id = {score_dict['id']: score_dict['score'] for score_dict in score_title_ids}
    answer_scores_by_id = {score_dict['id']: score_dict['score'] for score_dict in score_answer_ids}

    score_sbert_ids = []
    for id in set(title_scores_by_id.keys()).intersection(answer_scores_by_id.keys()):
        final_score = title_scores_by_id[id] * answer_scores_by_id[id]
        score_sbert_ids.append({'id': id, 'score': final_score})

    score_sbert_ids = sorted(score_sbert_ids, key=lambda x: x['score'], reverse=True)
    score_bm25_ids = [{'id': doc_dict['id'], 'score': doc_score} for doc_dict, doc_score in zip(top_docs, doc_scores)]
    final_scores = []
    for score_sbert in score_sbert_ids:
        for score_bm25 in score_bm25_ids:
            score_sbert_id = str(score_sbert['id'])
            if score_sbert_id == score_bm25['id']:
                final_score = score_sbert['score'] * score_bm25['score']
                final_scores.append({'id': score_sbert_id, 'final_score': final_score})
                break

    final_scores = sorted(final_scores, key=lambda x: x['final_score'], reverse=True)

    return final_scores[:5]

def getAnswer(doc_id):
    client = pymongo.MongoClient("mongodb://localhost:27017/")
    database = client["lawlaboratory"]
    collection = database["questions_cleaned"]

    document = collection.find_one({"_id": ObjectId(doc_id)})
    result = {}
    if document:
        result['id'] = str(doc_id)
        if 'title' in document and document['title']:
            result['title'] = document.get('title')

        if 'field' in document and document['field']:
            result['field'] = document.get('field')

        if 'source_url' in document and document['source_url']:
            result['source_url'] = document.get('source_url')

        if 'reference' in document and document['reference']:
            result['reference'] = document.get('reference')

        if 'quote' in document:
            quote = document['quote']
            if quote is not None:
                result['quote'] = {
                    'name': quote.get('name'),
                    'content': quote.get('content', [])
                }

        conclusion = document.get('conclusion', [])
        result['conclusion'] = conclusion

    return result

def same_question(questions_final):
    client = pymongo.MongoClient("mongodb://localhost:27017/")
    database = client["lawlaboratory"]
    collection = database["questions_cleaned"]

    same_questions = []
    for index, row in enumerate(questions_final):
        if index == 0:
            continue
        doc_id_str = str(row['id'])
        document = collection.find_one({"_id": ObjectId(doc_id_str)})
        if document:
            title = document.get('title')
            source_url = document.get('source_url')
            same_questions.append({"id": doc_id_str, "title": title, "source_url": source_url})

    return same_questions

@app.route('/query', methods=['POST'])
def query():
    data = request.json
    query = data.get('question')
    selected_rows = handle_bm25(query)
    handle_sbert_title(query, selected_rows)
    handle_sbert_answer(query, selected_rows)
    questions_final = calculate_score()

    doc_id_value = questions_final[0]['id']
    main_answer = getAnswer(doc_id_value)
    similar_questions = same_question(questions_final)

    response = {
        "main_answer": main_answer,
        "similar_questions": similar_questions
    }
    reset_variable()

    return jsonify(response)

def getAllLaws(database_name, collection_name):
    client = pymongo.MongoClient('mongodb://localhost:27017/')
    db = client[database_name]
    collection = db[collection_name]
    result = collection.find()

    return list(result)

def find_empty_laws(all_laws):
    empty_laws = []
    for law in all_laws:
        if not law.get('parts') and not law.get('chapters') and not law.get('articles'):
            empty_laws.append(law)
    return empty_laws

def delete_empty_laws_db(database_name, collection_name, empty_laws):
    client = pymongo.MongoClient('mongodb://localhost:27017/')
    db = client[database_name]
    collection = db[collection_name]

    for law in empty_laws:
        identifier = law['identifier']
        collection.delete_one({'identifier': identifier})
@app.route('/delete/emptyLaw', methods=['POST'])
def delete_empty_law_api():
    laws = getAllLaws('lawlaboratory', 'laws')
    codes = getAllLaws('lawlaboratory', 'codes')
    constitution = getAllLaws("lawlaboratory", "constitution")
    all_laws = laws + codes + constitution

    empty_laws = find_empty_laws(all_laws)

    delete_empty_laws_db('lawlaboratory', 'laws', empty_laws)
    delete_empty_laws_db('lawlaboratory', 'codes', empty_laws)
    delete_empty_laws_db('lawlaboratory', 'constitution', empty_laws)


def reset_variable():
    global top_20_indices_title
    global top_20_indices_anwser
    global doc_scores
    global top_docs
    global ids_title
    global similarities_title
    global sentences_title
    global sentences_anwser
    global ids_anwser
    global similarities_anwser

    top_20_indices_title = []
    top_20_indices_anwser = []
    doc_scores = []
    top_docs = []
    ids_title = []
    similarities_title = []
    sentences_title = []
    sentences_anwser = []
    ids_anwser = []
    similarities_anwser = []

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000)