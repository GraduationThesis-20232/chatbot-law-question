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

df = pd.read_json('data/cleaned/temp.jsonl', lines=True)
top_50_indices_title = []
top_50_indices_anwser = []
doc_scores = []
top_docs = []
ids_title = []
similarities_title = []
sentences_title = []
sentences_anwser = []
ids_anwser = []
similarities_anwser = []

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

    with open('output/bm25/texts.pkl', 'rb') as f:
        texts = pickle.load(f)
    with open('output/bm25/bm25plus.pkl', 'rb') as f:
        bm25plus = pickle.load(f)

    docs = load_docs_from_file('output/bm25/docs.jsonl')

    global top_docs
    top_docs = bm25plus.get_top_n(query_bm25, docs, n=100)
    bm25_scores = bm25plus.get_scores(query_bm25)
    global doc_scores
    doc_scores = np.sort(bm25_scores)[::-1][:100]
    # for doc_dict, score in zip(top_docs, doc_scores):
    #     doc_id = doc_dict['id']
    #     doc_title = doc_dict['title']
    #     print(f"ID: {doc_id}, Document: {doc_title}, Score: {score}")

    doc_ids = [doc_dict['id'] for doc_dict in top_docs]
    selected_rows = df[df['_id'].isin(doc_ids)]

    return selected_rows

def handle_sbert_title(query, selected_rows):
    global sentences_title
    global ids_title

    for idx in selected_rows.index:
        sentences_title.append(selected_rows.at[idx, 'title'])
        ids_title.append(selected_rows.at[idx, '_id'])

    model_title = SentenceTransformer('keepitreal/vietnamese-sbert')
    if torch.cuda.is_available():
        model_title = model_title.to(torch.device("cuda"))
    embeddings_title = model_title.encode(sentences_title)

    query_sbert = preprocess(query)
    query_sbert = query_sbert.replace('_', ' ')
    query_embedding_title = model_title.encode([query_sbert], convert_to_tensor=True)
    query_embedding_title = query_embedding_title.cpu().numpy()
    global similarities_title
    similarities_title = cosine_similarity(query_embedding_title, embeddings_title)
    global top_50_indices_title
    top_50_indices_title = np.argsort(similarities_title[0])[::-1][:50]

    # for idx in top_50_indices_title:
    #     similarity_score = similarities_title[0][idx]
    #     similar_sentence = sentences_title[idx]
    #     doc_id = ids_title[idx]
    #     print(f"ID: {doc_id}, Similarity Score: {similarity_score:.4f}, Sentence: {similar_sentence}")

def handle_sbert_answer(query, selected_rows):
    global sentences_anwser
    global ids_anwser

    for idx in selected_rows.index:
        sentences_anwser.append(selected_rows.at[idx, 'answer'])
        ids_anwser.append(selected_rows.at[idx, '_id'])

    query_sbert = preprocess(query)
    query_sbert = query_sbert.replace('_', ' ')

    model_anwser = SentenceTransformer('keepitreal/vietnamese-sbert')
    if torch.cuda.is_available():
        model_anwser = model_anwser.to(torch.device("cuda"))
    embeddings_anwser = model_anwser.encode(sentences_anwser)


    query_embedding_anwser = model_anwser.encode([query_sbert], convert_to_tensor=True)
    query_embedding_anwser = query_embedding_anwser.cpu().numpy()
    global similarities_anwser
    similarities_anwser = cosine_similarity(query_embedding_anwser, embeddings_anwser)
    global top_50_indices_anwser
    top_50_indices_anwser = np.argsort(similarities_anwser[0])[::-1][:50]

    # for idx in top_50_indices_anwser:
    #     similarity_score = similarities_anwser[0][idx]
    #     similar_sentence = sentences_anwser[idx]
    #     doc_id = ids_anwser[idx]
    #     print(f"ID: {doc_id}, Similarity Score: {similarity_score:.4f}, Sentence: {similar_sentence}")

def calculate_score():
    df_title = pd.DataFrame({
        'doc_id': [ids_title[idx] for idx in top_50_indices_title],
        'similarity_score_title': [similarities_title[0][idx] for idx in top_50_indices_title],
        'sentence_title': [sentences_title[idx] for idx in top_50_indices_title]
    })

    df_anwser = pd.DataFrame({
        'doc_id': [ids_anwser[idx] for idx in top_50_indices_anwser],
        'similarity_score_anwser': [similarities_anwser[0][idx] for idx in top_50_indices_anwser],
    })

    df_merged = pd.merge(df_title, df_anwser, on='doc_id')
    df_merged['sbert_score'] = df_merged['similarity_score_title'] * df_merged['similarity_score_anwser']

    df_merged_sorted = df_merged.sort_values(by='sbert_score', ascending=False)
    top_20_rows = df_merged_sorted.head(20)
    df_bm25 = pd.DataFrame({
        'doc_id': [doc_dict['id'] for doc_dict in top_docs],
        'bm25_score': doc_scores
    })

    df_bm25_filtered = df_bm25[df_bm25['doc_id'].isin(top_20_rows['doc_id'])]
    df_final = pd.merge(df_bm25_filtered, top_20_rows, on='doc_id')
    df_final['final_score'] = df_final['sbert_score'] * df_final['bm25_score']
    df_final_sorted = df_final.sort_values(by='final_score', ascending=False)

    return df_final_sorted.head(5)

def print_answer(doc_id):
    client = pymongo.MongoClient("mongodb://localhost:27017/")

    database = client["lawlaboratory"]
    collection = database["temp"]

    document = collection.find_one({"_id": ObjectId(doc_id)})
    print(doc_id)
    if document:
        # print("Title:", document.get('title'))

        if 'field' in document and document['field']:
            print("Lĩnh vực câu hỏi: ", document.get('field'))

        if 'reference' in document and document['reference']:
            print("Điều luật tham khảo: ", document.get('reference'))

        if 'quote' in document:
            quote = document['quote']
            print("Nội dung trích dẫn văn bản pháp luật như sau")
            print(quote.get('name'))

            quote_content = quote.get('content', [])
            for item in quote_content:
                print(item)

        conclusion = document.get('conclusion', [])
        print("Kết luận:")
        for item in conclusion:
            print(item)

def same_question(df_final):
    client = pymongo.MongoClient("mongodb://localhost:27017/")

    database = client["lawlaboratory"]
    collection = database["temp"]

    print("Các câu hỏi tương tự")
    for index, row in df_final.iterrows():
        doc_id_str = str(row['doc_id'])
        document = collection.find_one({"_id": ObjectId(doc_id_str)})
        if document:
            title = document.get('title')
            print(f"Câu hỏi: {title}")



if __name__ == '__main__':
    query = input("Câu hỏi: ")

    selected_rows = handle_bm25(query)

    handle_sbert_title(query, selected_rows)
    handle_sbert_answer(query, selected_rows)

    df_final = calculate_score()

    doc_id_value = df_final.iloc[0]['doc_id']
    print_answer(doc_id_value)
    same_question(df_final)