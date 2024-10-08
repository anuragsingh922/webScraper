from bs4 import BeautifulSoup as bs
from flask import Flask, request, jsonify
from googlesearch import search
from rank_bm25 import BM25Okapi
import string
from sklearn.feature_extraction import _stop_words
from tqdm.autonotebook import tqdm
import numpy as np
import concurrent.futures
import time
import requests
from requests.packages.urllib3.exceptions import InsecureRequestWarning

# Suppress warnings from urllib3
requests.packages.urllib3.disable_warnings(InsecureRequestWarning)

# Initialize Flask app
app = Flask(__name__)

# Tokenizer function for BM25
def bm25_tokenizer(text):
    tokenized_doc = []
    for token in text.lower().split():
        token = token.strip(string.punctuation)
        if len(token) > 0 and token not in _stop_words.ENGLISH_STOP_WORDS:
            tokenized_doc.append(token)
    return tokenized_doc

# BM25 processing function
def BM25func(passages, query):
    tokenized_corpus = [bm25_tokenizer(passage) for passage in passages]
    bm25 = BM25Okapi(tokenized_corpus)
    bm25_scores = bm25.get_scores(bm25_tokenizer(query))

    top_n = np.argpartition(bm25_scores, -10)[-10:]
    bm25_hits = [{'corpus_id': idx, 'score': bm25_scores[idx]} for idx in top_n]
    bm25_hits = sorted(bm25_hits, key=lambda x: x['score'], reverse=True)
    
    bm25_passages = [' '.join(passages[hit["corpus_id"]].split()[:100]) for hit in bm25_hits]
    return bm25_passages

# Scraper function
def scraper(url, con, DataWrtUrls, passages):
    try:
        session = requests.Session()
        my_headers = {
            "User-Agent": "Mozilla/5.0 (X11; CrOS x86_64 14685.0.0) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/102.0.4992.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9"
        }
        result = session.get(url, headers=my_headers, verify=True, timeout=3)
        doc = bs(result.content, "html.parser")
        contents = doc.find_all("p")
        for content in contents:
            passages.append(content.text)
            con.append(content.text + "\n")
        DataWrtUrls[url] = str(con)
    except Exception as e:
        print(f"Error scraping {url}: {e}")

# Main internet query handling function
def internet(query):
    urls, passages, con = [], [], []
    search_results = list(search(query, tld="com", num=10, stop=10, pause=0.75))
    for j in search_results:
        urls.append(j)

    DataWrtUrls = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        for url in urls:
            executor.submit(scraper, url, con, DataWrtUrls, passages)

    # Select passages with minimum length to avoid empty BM25 input
    passages2 = [p for p in passages if len(p.split()) >= 80]

    # Compute BM25 scores and get top-ranked passages
    bi_encoder_searched_passages = BM25func(passages2, query)

    # Prepare the response with top passages and their respective URLs
    UrlWrtRank = {}
    k = 0
    for passage in bi_encoder_searched_passages:
        for url, value in DataWrtUrls.items():
            if passage in value:
                UrlWrtRank[passage] = url
                break

    # Response structure with passages and corresponding URLs
    response = {
        "query": query,
        "top_passages": [{"passage": passage, "url": UrlWrtRank.get(passage)} for passage in bi_encoder_searched_passages]
    }

    return response

# Flask endpoints
@app.route('/check', methods=['GET'])
def check():
    return "Fine."

@app.route('/internet', methods=['POST'])
def query():
    data = request.json
    query_text = data['query']
    response = internet(query_text)
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True, port=8081)
