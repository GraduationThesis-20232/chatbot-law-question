import pandas as pd
from sentence_transformers import CrossEncoder
import numpy as np
import pickle
import json
from text_untils import *
from bson import ObjectId
from flask import Flask, request, jsonify
from flask_cors import CORS
from pymongo import MongoClient
from transformers import AutoTokenizer, AutoModel
import torch
import faiss
import math
from collections import defaultdict
import time
import logging
import random

app = Flask(__name__)
CORS(app)

final_ensemble = {}

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

alpha = 0.6
threshold = 0.6
logging.basicConfig(filename='output/log.txt', level=logging.INFO, format='%(message)s')

model_cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', max_length=512, default_activation_function=torch.nn.Sigmoid(), device='cuda')
tokenizer = AutoTokenizer.from_pretrained('bkai-foundation-models/vietnamese-bi-encoder')
model_bi_encoder = AutoModel.from_pretrained('bkai-foundation-models/vietnamese-bi-encoder')
docs = load_docs_from_file('output/bm25/bm25_65k_questions_new/docs.jsonl')
with open('output/bm25/bm25_65k_questions_new/bm25plus.pkl', 'rb') as f:
    bm25plus = pickle.load(f)
index = faiss.read_index('output/sbert/embeddings_65k_index.index')


def preprocess(query):
    query = clean_text(query)
    query = word_segment(query)
    query = normalize_text(query)
    return query

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def get_selected_rows_from_mongodb(database_name, collection_name, top_docs):
    client = MongoClient('mongodb://localhost:27017/')
    db = client[database_name]
    collection = db[collection_name]

    doc_ids = [ObjectId(doc_dict['id']) for doc_dict in top_docs]

    cursor = collection.find({'_id': {'$in': doc_ids}}, {
        'title': 1,
        'reference': 1,
        'quote': 1,
        'conclusion': 1
    })
    data = list(cursor)
    df = pd.DataFrame(data)

    # client.close()
    return df

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

def count_tokens(text):
    tokens = tokenizer.tokenize(text)
    return len(tokens)

def create_answer(row):
    conclusion_value = row['conclusion']

    if isinstance(conclusion_value, list):
        conclusion_value = ' '.join(conclusion_value)
    elif pd.isna(conclusion_value):
        conclusion_value = ''

    quote_value= ''
    if not pd.isna(row['quote']):
        reference = row['reference'] if not pd.isna(row['reference']) else ''
        merged_quote = merge_quote(row['quote'])
        quote_value = f"{reference} {merged_quote}"

    conclusion_tokens = count_tokens(conclusion_value)
    quote_tokens = count_tokens(quote_value)

    if conclusion_tokens >= quote_tokens:
        return conclusion_value
    else:
        return quote_value
def calculate_ensemble_score(bm25_score, bi_encoder_score, cross_encoder_score):
    return (1-alpha) * bm25_score + alpha * math.sqrt(bi_encoder_score * cross_encoder_score)

def bm25_title(query):
    query_bm25 = remove_stopword(query)
    query_bm25 = query_bm25.split()

    bm25_scores = bm25plus.get_scores(query_bm25)
    top_n_indices = np.argsort(bm25_scores)[::-1][:50]
    top_n_score = bm25_scores[top_n_indices]

    topn_docs_bm25 = [docs[idx] for idx in top_n_indices]

    max_score = max(top_n_score)
    if max_score <= threshold:
        return []
    rescaled_scores = [1 * score / max_score for score in top_n_score]

    final_top_docs_bm25 = []
    for (doc_dict, score) in zip(topn_docs_bm25, rescaled_scores):
        doc_id = doc_dict['id']
        doc_title = doc_dict['title']
        final_top_docs_bm25.append({'id': doc_id, 'bm25_score': score})

    return final_top_docs_bm25

def bi_encoder_title(query):
    encoded_query = tokenizer([query], padding=True, truncation=True, return_tensors='pt', max_length=256)
    with torch.no_grad():
        model_output_query = model_bi_encoder(**encoded_query)

    embeddings_query = mean_pooling(model_output_query, encoded_query['attention_mask'])
    embeddings_query = embeddings_query.cpu().numpy()
    faiss.normalize_L2(embeddings_query)

    distances_bi_encoder, indices = index.search(embeddings_query, 50)
    topn_doc_bi_encoder = [docs[idx] for idx in indices[0]]

    if distances_bi_encoder[0][0] <= threshold:
        return []

    final_top_docs_bi_encoder = []
    for distance, doc_dict in zip(distances_bi_encoder[0], topn_doc_bi_encoder):
        doc_id = doc_dict['id']
        doc_title = doc_dict['title']
        final_top_docs_bi_encoder.append({'id': doc_id, 'bi_encoder_score': distance})

    return final_top_docs_bi_encoder

def combined_retrieval(final_top_docs_bm25, final_top_docs_bi_encoder):
    combined_retrieval_scores = defaultdict(lambda: {'bm25_score': 0, 'bi_encoder_score': 0})

    for doc in final_top_docs_bm25:
        combined_retrieval_scores[doc['id']]['bm25_score'] = doc['bm25_score']

    for doc in final_top_docs_bi_encoder:
        combined_retrieval_scores[doc['id']]['bi_encoder_score'] = doc['bi_encoder_score']

    final_retrieval_scores = []
    for doc_id, scores in combined_retrieval_scores.items():
        if scores['bm25_score'] > 0 and scores['bi_encoder_score'] > 0:
            combined_score = math.sqrt(scores['bm25_score'] * scores['bi_encoder_score'])
            final_retrieval_scores.append({'id': doc_id, 'retrieval_score': combined_score})

            if doc_id not in final_ensemble:
                final_ensemble[doc_id] = {'id': doc_id}
            final_ensemble[doc_id]['bm25_score'] = scores['bm25_score']

            if doc_id not in final_ensemble:
                final_ensemble[doc_id] = {'id': doc_id}
            final_ensemble[doc_id]['bi_encoder_score'] = scores['bi_encoder_score']

    final_scores_retrieval = sorted(final_retrieval_scores, key=lambda x: x['retrieval_score'], reverse=True)
    top_20_scores_retrieval = final_scores_retrieval[:20]

    selected_rows = get_selected_rows_from_mongodb('lawlaboratory', 'questions_cleaned', top_20_scores_retrieval)

    return selected_rows

def cross_encoder_title(query, selected_rows):
    selected_rows['conclusion'] = selected_rows.apply(lambda row: merge_quote(row['quote']) if row['conclusion'] == [] else row['conclusion'], axis=1)
    selected_rows['answer'] = selected_rows.apply(create_answer, axis=1)

    answers = selected_rows['answer'].tolist()
    query_answer_pairs = [(query, answer) for answer in answers]

    scores = model_cross_encoder.predict(query_answer_pairs)
    final_top_docs_cross_encoder = []
    for index, row in selected_rows.iterrows():
        doc_id = row['_id']
        title = row['title']
        score = scores[index]
        final_top_docs_cross_encoder.append({'id': doc_id, 'cross_encoder_score': score})

    for record in final_top_docs_cross_encoder:
        doc_id = str(record['id'])
        if doc_id not in final_ensemble:
            final_ensemble[doc_id] = {'id': doc_id}
        final_ensemble[doc_id]['cross_encoder_score'] = record['cross_encoder_score']

def ensemble():
    final_combined_scores = {}

    for doc_id, scores in final_ensemble.items():
        cross_encoder_score = scores.get('cross_encoder_score', None)
        bm25_score = scores.get('bm25_score', None)
        bi_encoder_score = scores.get('bi_encoder_score', None)

        if cross_encoder_score is not None and bm25_score is not None and bi_encoder_score is not None:
            final_combined_scores[doc_id] = calculate_ensemble_score(bm25_score, bi_encoder_score, cross_encoder_score)

    final_combined_scores = sorted(final_combined_scores.items(), key=lambda x: x[1], reverse=True)

    if final_combined_scores[0][1] <= threshold:
        print(final_combined_scores[0][1])
        return []

    return final_combined_scores[:5]

def getAnswer(doc_id):
    client = MongoClient("mongodb://localhost:27017/")
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

def same_question(final_combined_scores):
    client = MongoClient("mongodb://localhost:27017/")
    database = client["lawlaboratory"]
    collection = database["questions_cleaned"]

    doc_ids = [ObjectId(item[0]) for item in final_combined_scores]

    cursor = collection.find({'_id': {'$in': doc_ids}}, {
        'title': 1,
        'source_url': 1
    })
    data = list(cursor)

    for document in data:
        del document['_id']

    return data[1:]

@app.route('/query', methods=['POST'])
def query():
    error_response = {
        "status": "ERROR",
        "message": "Xin lỗi, tôi chưa biết câu trả lời cho câu hỏi này",
    }

    data = request.json
    query = data.get('question')
    query = preprocess(query)

    final_top_docs_bm25 = bm25_title(query)
    if len(final_top_docs_bm25) == 0:
        return jsonify(error_response)

    final_top_docs_bi_encoder = bi_encoder_title(query)
    if len(final_top_docs_bi_encoder) == 0:
        return jsonify(error_response)

    selected_rows = combined_retrieval(final_top_docs_bm25, final_top_docs_bi_encoder)
    cross_encoder_title(query, selected_rows)

    final_combined_scores = ensemble()
    if len(final_combined_scores) == 0:
        return jsonify(error_response)

    main_answer = getAnswer(final_combined_scores[0][0])
    similar_questions = same_question(final_combined_scores)

    response = {
        "status": "OK",
        "main_answer": main_answer,
        "similar_questions": similar_questions
    }
    final_ensemble.clear()

    return jsonify(response)

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000)

    # client = MongoClient('mongodb://localhost:27017/')
    # db = client["lawlaboratory"]
    # source_collection = db["questions_new"]
    #
    # titles = source_collection.find({}, {'title': 1})
    # titles_list = [title['title'] for title in titles]
    #
    # random_titles = random.sample(titles_list, min(1000, len(titles_list)))
    # client.close()
    #
    # for query in random_titles:
    #     start_time = time.time()
    #     query = preprocess(query)
    #
    #     final_top_docs_bm25 = bm25_title(query)
    #     bm25_time = time.time() - start_time
    #
    #     final_top_docs_bi_encoder = bi_encoder_title(query)
    #     bi_encoder_time = time.time() - start_time
    #
    #     selected_rows = combined_retrieval(final_top_docs_bm25, final_top_docs_bi_encoder)
    #
    #     cross_encoder_title(query, selected_rows)
    #     cross_encoder_time = time.time() - start_time
    #
    #     final_combined_scores = ensemble()
    #     ensemble_time = time.time() - start_time
    #
    #     main_answer = getAnswer(final_combined_scores[0][0])
    #     similar_questions = same_question(final_combined_scores)
    #     fetch_answers_time = time.time() - start_time
    #
    #     log_message = f"{bm25_time}, {bi_encoder_time}, {cross_encoder_time}, {ensemble_time}, {fetch_answers_time}"
    #     logging.info(log_message)
    #
    #     final_ensemble.clear()
