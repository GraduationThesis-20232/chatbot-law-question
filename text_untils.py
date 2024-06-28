import re, string
from pyvi.ViTokenizer import tokenize
import py_vncorenlp

def remove_stopword(text):
    filename = './vietnamese.txt'
    with open(filename, 'r', encoding='utf-8') as file:
        list_stopwords = file.read().splitlines()
    pre_text = []
    words = text.split()
    for word in words:
        if word not in list_stopwords:
            pre_text.append(word)
    text2 = ' '.join(pre_text)

    return text2

def clean_text(text):
    text = re.sub('<.*?>', '', text).strip()        # Remove HTML tags
    text = re.sub('(\s)+', r'\1', text)            # Remove extra spaces
    return text

def normalize_text(text):
    listpunctuation = string.punctuation.replace('_', '')
    for i in listpunctuation:
        text = text.replace(i, ' ')         # Remove punctuation
    return text.lower()

def word_segment(text):
    text = tokenize(text.encode('utf-8').decode('utf-8'))

    # rdrsegmenter = py_vncorenlp.VnCoreNLP(annotators=["wseg"], save_dir='E:\Desktop\Python\GraduationResearch2\output/vncorenlp')
    # text = rdrsegmenter.word_segment(text)

    return text
