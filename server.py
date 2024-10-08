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

# Disable warnings about insecure requests
requests.packages.urllib3.disable_warnings(InsecureRequestWarning)

app = Flask(__name__)

# Tokenizer for BM25
def bm25_tokenizer(text):
    tokenized_doc = []
    for token in text.lower().split():
        token = token.strip(string.punctuation)
        if len(token) > 0 and token not in _stop_words.ENGLISH_STOP_WORDS:
            tokenized_doc.append(token)
    return tokenized_doc

# BM25 function
def BM25func(passages, query):
    tokenized_corpus = [bm25_tokenizer(passage) for passage in tqdm(passages)]
    bm25 = BM25Okapi(tokenized_corpus)
    bm25_scores = bm25.get_scores(bm25_tokenizer(query))

    print("BM25 SCORES:", len(bm25_scores))
    
    # Handle cases with fewer than 10 scores
    top_n = min(10, len(bm25_scores))
    
    # Use argpartition to find the indices of the top scores
    if top_n > 0:
        top_n_indices = np.argpartition(bm25_scores, -top_n)[-top_n:]  # Get top N indices
        bm25_hits = [{'corpus_id': idx, 'score': bm25_scores[idx]} for idx in top_n_indices]
        bm25_hits = sorted(bm25_hits, key=lambda x: x['score'], reverse=True)

        bm25_passages = [' '.join(passages[hit["corpus_id"]].split()[:100]) for hit in bm25_hits]
        print(bm25_passages)
        return bm25_passages
    else:
        print("No passages found for the query.")
        return []

# Web scraper
def scraper(url, con, DataWrtUrls, passages):
    print("Scrapper running")
    session = requests.Session()
    my_headers = {
        "User-Agent": "Mozilla/5.0 (X11; CrOS x86_64 14685.0.0) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/102.0.4992.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9"
    }
    try:
        result = session.get(url, headers=my_headers, verify=True, timeout=3)
        doc = bs(result.content, "html.parser")
        contents = doc.find_all("p")
        for content in contents:
            passages.append(content.text)
            con.append(content.text + "\n")
        DataWrtUrls[url] = str(con)
    except Exception as e:
        print(f"Error scraping {url}: {e}")

# Internet search function
def internet(query):
    customer_message = query
    urls = []
    passages = []
    con = []
    start = time.time()

    search_results = list(search(customer_message, tld="com", num=10, stop=10, pause=0.75))
    urls.extend(search_results)
    print("URLS=", urls)

    DataWrtUrls = {}
    time_for_scraping = time.time()

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        executor.map(lambda url: scraper(url, con, DataWrtUrls, passages), urls)

    print("Passages=", passages)
    print("time for scraping: ", time.time() - time_for_scraping)

    passages2 = []
    i = 0
    try:
        while i < len(passages) and len(passages2) < 10:
            P = ""
            Z = ""
            while len(Z) <= 80 and i < len(passages):
                P += (passages[i])
                Z = P.split()
                i += 1
            passages2.append(P)
    except Exception as e:
        print(f"Error processing passages: {e}")

    end = time.time() - start
    start = time.time()
    bi_encoder_searched_passages = BM25func(passages2, customer_message)

    end = time.time()
    print(f"Runtime of the program is {end - start}")

    supporting_texts = "\n".join(
        [f"Supporting Text {i+1}: {text}" for i, text in enumerate(bi_encoder_searched_passages[:7])]
    )
    print(supporting_texts)

    UrlWrtRank = {}
    k = 0
    for i in range(len(bi_encoder_searched_passages)):
        for url, value in DataWrtUrls.items():
            if k == 7:
                break
            if str(bi_encoder_searched_passages[i]) in str(value):
                UrlWrtRank[k] = url
                k += 1

    return {"supporting_texts": supporting_texts, "UrlWrtRank": UrlWrtRank}

# Basic route for home
@app.route('/')
def home():
    return "Welcome to TheCoder API!"

# Health check route
@app.route('/check', methods=['GET'])
def check():
    return "Fine."

# Main query route
@app.route('/internet', methods=['POST'])
def query():
    print(str(request.json))
    data = request.json
    query_text = data['query']
    response = internet(query_text)
    return jsonify(response)

if __name__ == '__main__':
    # Use this for development only
    # app.run(debug=True, port=8081)

    # For production use, make sure to bind to all interfaces
    app.run(host='0.0.0.0', port=8080)  # Change port if needed

