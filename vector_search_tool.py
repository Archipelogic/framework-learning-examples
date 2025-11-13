#!/usr/bin/env python3
"""
Custom Vector Search Tool for CrewAI
Loads pre-computed embeddings from JSON files (like teammate's implementation)
"""

import json
import numpy as np
from pathlib import Path
from typing import Optional
from sklearn.metrics.pairwise import cosine_similarity
from crewai.tools import BaseTool
import boto3


class VectorSearchTool(BaseTool):
    name: str = "Vector Search"
    description: str = "Search through project documents using semantic similarity. Provide a natural language query to find relevant information."
    
    embeddings: Optional[np.ndarray] = None
    metadata: Optional[list] = None
    bedrock_client: Optional[any] = None
    
    def __init__(self):
        super().__init__()
        # Load embeddings and metadata from JSON files
        data_dir = Path(__file__).parent / 'data'
        
        # Load text embeddings
        embeddings_file = data_dir / 'text_embeddings.json'
        if embeddings_file.exists():
            with open(embeddings_file, 'r') as f:
                embeddings_data = json.load(f)
                self.embeddings = np.array(embeddings_data)
        else:
            raise FileNotFoundError(f"Embeddings file not found: {embeddings_file}")
        
        # Load metadata
        metadata_file = data_dir / 'metadata.json'
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                self.metadata = json.load(f)
        else:
            raise FileNotFoundError(f"Metadata file not found: {metadata_file}")
        
        # Initialize Bedrock client for query embeddings
        self.bedrock_client = boto3.client('bedrock-runtime', region_name='us-east-1')
    
    def _get_query_embedding(self, query: str) -> np.ndarray:
        """Get embedding for the query using Bedrock"""
        response = self.bedrock_client.invoke_model(
            modelId='amazon.titan-embed-text-v1',
            body=json.dumps({"inputText": query})
        )
        response_body = json.loads(response['body'].read())
        return np.array(response_body['embedding'])
    
    def _run(self, query: str, top_k: int = 3) -> str:
        """
        Search for relevant documents using cosine similarity
        
        Args:
            query: The search query
            top_k: Number of top results to return
            
        Returns:
            Formatted string with top results
        """
        try:
            # Get query embedding
            query_embedding = self._get_query_embedding(query)
            query_embedding = query_embedding.reshape(1, -1)
            
            # Calculate cosine similarity
            similarities = cosine_similarity(query_embedding, self.embeddings)[0]
            
            # Get top k indices
            top_indices = np.argsort(similarities)[-top_k:][::-1]
            
            # Format results
            results = []
            for idx in top_indices:
                score = similarities[idx]
                meta = self.metadata[idx]
                results.append(f"Score: {score:.4f}\n{meta.get('text', 'No text available')}\n")
            
            return "\n---\n".join(results)
            
        except Exception as e:
            return f"Error performing vector search: {str(e)}"
