import os
import pickle
import pandas as pd
import numpy as np
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from rank_bm25 import BM25Okapi
import faiss
from sentence_transformers import SentenceTransformer

# Download nltk data if not present
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

class BM25Searcher:
    def __init__(self, df, text_column='content_narrative', index_path='bm25_index.pkl'):
        self.df = df
        self.text_column = text_column
        self.index_path = index_path
        self.stop_words = set(stopwords.words('english'))
        self.tokenizer = RegexpTokenizer(r'\w+')
        
        if os.path.exists(self.index_path):
            print(f"Loading precalculated BM25 index from {self.index_path}...")
            with open(self.index_path, 'rb') as f:
                data = pickle.load(f)
                self.bm25 = data['bm25']
            print("BM25 Index loaded successfully.")
        else:
            print("Tokenizing corpus for BM25...")
            corpus = self.df[self.text_column].fillna('').tolist()
            tokenized_corpus = [self._tokenize(doc) for doc in corpus]
            print("Building BM25 Index...")
            self.bm25 = BM25Okapi(tokenized_corpus)
            print(f"Saving BM25 index to {self.index_path}...")
            with open(self.index_path, 'wb') as f:
                pickle.dump({'bm25': self.bm25}, f)
            print("BM25 Index built and saved successfully.")

    def _tokenize(self, text):
        tokens = self.tokenizer.tokenize(text.lower())
        return [t for t in tokens if t not in self.stop_words]

    def search(self, query, top_k=5):
        tokenized_query = self._tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        results = []
        for rank, idx in enumerate(top_indices, start=1):
            title = self.df.iloc[idx]['title']
            results.append({
                'title': title,
                'score': float(scores[idx]),
                'rank': rank,
                'method': 'bm25'
            })
        return results


class DenseSearcher:
    def __init__(self, df, model_name='paraphrase-multilingual-MiniLM-L12-v2', embedding_col='embedding', index_path='faiss_index.bin', map_path='faiss_id_map.pkl'):
        self.df = df
        self.embedding_col = embedding_col
        self.index_path = index_path
        self.map_path = map_path
        
        print(f"Loading SentenceTransformer model ({model_name})...")
        self.model = SentenceTransformer(model_name)
        
        if os.path.exists(self.index_path) and os.path.exists(self.map_path):
            print(f"Loading precalculated FAISS index from {self.index_path}...")
            self.index = faiss.read_index(self.index_path)
            with open(self.map_path, 'rb') as f:
                self.id_map = pickle.load(f)
            print("FAISS Index and ID map loaded successfully.")
        else:
            print("Building FAISS Index...")
            # Extract embeddings as a numpy array
            embeddings = np.stack(self.df[self.embedding_col].values).astype('float32')
            # Normalize embeddings for cosine similarity search (inner product)
            faiss.normalize_L2(embeddings)
            
            dimension = embeddings.shape[1]
            self.index = faiss.IndexFlatIP(dimension)
            self.index.add(embeddings)
            
            # Map index offset to DataFrame index/ID
            self.id_map = {i: self.df.index[i] for i in range(len(self.df))}
            
            print(f"Saving FAISS index to {self.index_path}...")
            faiss.write_index(self.index, self.index_path)
            with open(self.map_path, 'wb') as f:
                pickle.dump(self.id_map, f)
            print("FAISS Index built and saved successfully.")

    def search(self, query, top_k=5):
        # Embed the query
        query_embedding = self.model.encode([query]).astype('float32')
        # Normalize query vector
        faiss.normalize_L2(query_embedding)
        
        # Search FAISS index
        scores, indices = self.index.search(query_embedding, top_k)
        scores = scores[0]
        indices = indices[0]
        
        results = []
        for rank, (score, offset) in enumerate(zip(scores, indices), start=1):
            if offset == -1: # FAISS padding if not enough results
                continue
            df_idx = self.id_map[int(offset)]
            title = self.df.loc[df_idx, 'title']
            results.append({
                'title': title,
                'score': float(score),
                'rank': rank,
                'method': 'dense'
            })
        return results


class HybridSearcher:
    def __init__(self, bm25_searcher, dense_searcher):
        self.bm25_searcher = bm25_searcher
        self.dense_searcher = dense_searcher
        self.df = self.bm25_searcher.df
        # Pre-lowercase titles to make exact match substring checks extremely fast
        self.titles_lower = self.df['title'].str.lower().tolist()

    def search(self, query, top_k=5, rrf_k=60, use_exact_match=True):
        exact_titles = set()
        exact_indices = []
        results = []
        
        if use_exact_match:
            query_lower = query.lower()
            # Fast substring match on titles_lower list
            exact_indices = [i for i, t in enumerate(self.titles_lower) if query_lower in t]
            
            for idx in exact_indices:
                title = self.df.iloc[idx]['title']
                exact_titles.add(title)
                results.append({
                    'title': title,
                    'score': 999.0,
                    'rank': len(results) + 1,
                    'method': 'hybrid'
                })
                if len(results) >= top_k:
                    return results

        # For hybrid search, we retrieve more items initially to combine them
        initial_k = max(top_k * 2, 50)
        
        # 1. Run BM25 search
        tokenized_query = self.bm25_searcher._tokenize(query)
        bm25_scores = self.bm25_searcher.bm25.get_scores(tokenized_query)
        bm25_top_indices = np.argsort(bm25_scores)[::-1][:initial_k]
        
        # 2. Run FAISS search
        query_embedding = self.dense_searcher.model.encode([query]).astype('float32')
        faiss.normalize_L2(query_embedding)
        faiss_scores, faiss_indices = self.dense_searcher.index.search(query_embedding, initial_k)
        faiss_scores = faiss_scores[0]
        faiss_indices = faiss_indices[0]
        
        # Perform Reciprocal Rank Fusion on DataFrame index offsets
        rrf_scores = {}
        
        # Add BM25 ranks
        for rank, idx in enumerate(bm25_top_indices, start=1):
            idx = int(idx)
            # Filter out exact matches
            if idx in exact_indices:
                continue
            if idx not in rrf_scores:
                rrf_scores[idx] = 0.0
            rrf_scores[idx] += 1.0 / (rrf_k + rank)
            
        # Add Dense ranks
        for rank, offset in enumerate(faiss_indices, start=1):
            if offset == -1:
                continue
            idx = self.dense_searcher.id_map[int(offset)]
            # Filter out exact matches
            if idx in exact_indices:
                continue
            if idx not in rrf_scores:
                rrf_scores[idx] = 0.0
            rrf_scores[idx] += 1.0 / (rrf_k + rank)
            
        # Sort by RRF score
        sorted_results = sorted(rrf_scores.items(), key=lambda item: item[1], reverse=True)
        
        # Only retrieve metadata (titles) for the final top results
        for idx, score in sorted_results:
            title = self.df.iloc[idx]['title']
            results.append({
                'title': title,
                'score': score,
                'rank': len(results) + 1,
                'method': 'hybrid'
            })
            if len(results) >= top_k:
                break
                
        return results
