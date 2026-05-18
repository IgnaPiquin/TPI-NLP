import pandas as pd
import numpy as np
from rank_bm25 import BM25Okapi
import nltk
from nltk.tokenize import word_tokenize
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# Download punkt tokenizer data if not present
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

class BM25Searcher:
    def __init__(self, df, text_column='content_narrative'):
        self.df = df
        self.text_column = text_column
        self.corpus = self.df[self.text_column].fillna('').tolist()
        
        # Tokenize the corpus
        print("Tokenizing corpus for BM25...")
        self.tokenized_corpus = [self._tokenize(doc) for doc in self.corpus]
        self.bm25 = BM25Okapi(self.tokenized_corpus)
        print("BM25 Index built successfully.")

    def _tokenize(self, text):
        return word_tokenize(text.lower())

    def search(self, query, top_k=5, use_exact_match=False):
        exact_titles = set()
        results = []
        
        if use_exact_match:
            query_lower = query.lower()
            exact_indices = np.where(self.df['title'].str.lower().str.contains(query_lower, regex=False))[0]
            
            for idx in exact_indices:
                title = self.df.iloc[idx]['title']
                exact_titles.add(title)
                results.append({
                    'title': title,
                    'score': 999.0,
                    'rank': len(results) + 1,
                    'method': 'bm25'
                })
                if len(results) >= top_k:
                    return results

        tokenized_query = self._tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)
        
        top_indices = np.argsort(scores)[::-1]
        
        for idx in top_indices:
            title = self.df.iloc[idx]['title']
            if title not in exact_titles:
                results.append({
                    'title': title,
                    'score': float(scores[idx]),
                    'rank': len(results) + 1,
                    'method': 'bm25'
                })
            if len(results) >= top_k:
                break
                
        return results


class DenseSearcher:
    def __init__(self, df, model_name='paraphrase-multilingual-MiniLM-L12-v2', embedding_col='embedding'):
        self.df = df
        self.embedding_col = embedding_col
        print(f"Loading SentenceTransformer model ({model_name})...")
        self.model = SentenceTransformer(model_name)
        
        # Extract embeddings as a numpy array for fast cosine similarity
        self.doc_embeddings = np.stack(self.df[self.embedding_col].values)
        print("Dense embeddings loaded successfully.")

    def search(self, query, top_k=5, use_exact_match=False):
        exact_titles = set()
        results = []
        
        if use_exact_match:
            query_lower = query.lower()
            exact_indices = np.where(self.df['title'].str.lower().str.contains(query_lower, regex=False))[0]
            
            for idx in exact_indices:
                title = self.df.iloc[idx]['title']
                exact_titles.add(title)
                results.append({
                    'title': title,
                    'score': 999.0,
                    'rank': len(results) + 1,
                    'method': 'dense'
                })
                if len(results) >= top_k:
                    return results

        # Embed the query
        query_embedding = self.model.encode([query])
        
        # Calculate cosine similarity
        similarities = cosine_similarity(query_embedding, self.doc_embeddings)[0]
        
        top_indices = np.argsort(similarities)[::-1]
        
        for idx in top_indices:
            title = self.df.iloc[idx]['title']
            if title not in exact_titles:
                results.append({
                    'title': title,
                    'score': float(similarities[idx]),
                    'rank': len(results) + 1,
                    'method': 'dense'
                })
            if len(results) >= top_k:
                break
                
        return results


class HybridSearcher:
    def __init__(self, bm25_searcher, dense_searcher):
        self.bm25_searcher = bm25_searcher
        self.dense_searcher = dense_searcher

    def search(self, query, top_k=5, rrf_k=60, use_exact_match=False):
        exact_titles = set()
        results = []
        
        if use_exact_match:
            query_lower = query.lower()
            df = self.bm25_searcher.df
            exact_indices = np.where(df['title'].str.lower().str.contains(query_lower, regex=False))[0]
            
            for idx in exact_indices:
                title = df.iloc[idx]['title']
                exact_titles.add(title)
                results.append({
                    'title': title,
                    'score': 999.0,
                    'rank': len(results) + 1,
                    'method': 'hybrid'
                })
                if len(results) >= top_k:
                    return results

        # To do a good RRF fusion, we need to retrieve more items initially
        initial_k = max(top_k * 2, 50)
        
        bm25_results = self.bm25_searcher.search(query, top_k=initial_k, use_exact_match=False)
        dense_results = self.dense_searcher.search(query, top_k=initial_k, use_exact_match=False)
        
        # RRF Scoring
        rrf_scores = {}
        
        for res in bm25_results:
            title = res['title']
            if title not in exact_titles:
                if title not in rrf_scores:
                    rrf_scores[title] = 0.0
                rrf_scores[title] += 1.0 / (rrf_k + res['rank'])
            
        for res in dense_results:
            title = res['title']
            if title not in exact_titles:
                if title not in rrf_scores:
                    rrf_scores[title] = 0.0
                rrf_scores[title] += 1.0 / (rrf_k + res['rank'])
            
        # Sort by RRF score
        sorted_results = sorted(rrf_scores.items(), key=lambda item: item[1], reverse=True)
        
        for title, score in sorted_results:
            results.append({
                'title': title,
                'score': score,
                'rank': len(results) + 1,
                'method': 'hybrid'
            })
            if len(results) >= top_k:
                break
            
        return results
