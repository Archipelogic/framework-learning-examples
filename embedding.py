#!/usr/bin/env python3
"""
Create embeddings from DS_Projects_Docs.json and save to files
Run this BEFORE running crewai_unified.py

Also provides helper functions for loading pre-computed embeddings into FAISS
"""

import json
import boto3
import numpy as np
from pathlib import Path
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_aws import BedrockEmbeddings


def get_embedding(text, bedrock_client, model_id):
    """
    Generates an embedding vector for the given text using the specified Bedrock model.
    
    Args:
        text (str): The input text to embed.
        bedrock_client: The AWS Bedrock client used to invoke the model.
        model_id (str): The identifier of the embedding model to use.
    
    Returns:
        list: The embedding vector as a list of floats.
    
    Raises:
        ValueError: If no embedding is found in the response.
    """
    # Prepare the payload for the Titan Text Embeddings model
    body = json.dumps({"inputText": text})
    
    # Invoke the model
    response = bedrock_client.invoke_model(
        body=body,
        modelId=model_id,
        accept="application/json",
        contentType="application/json"
    )
    
    # Parse the response
    response_body = json.loads(response['body'].read())
    embedding = response_body['embedding']
    
    if embedding is None:
        raise ValueError("No embedding found in the response")
    
    # Convert to numpy float32 array
    embedding = np.array(embedding, dtype=np.float32)
    
    return embedding.tolist()


def normalize_cosine_distance(distance):
    """Normalize cosine distance to similarity score"""
    return (distance + 1.0) / 2.0


def get_embedding_for_vectordb(bedrock_client, model_id):
    """
    Get embedding function for vectorstore.
    
    Args:
        bedrock_client: The AWS Bedrock client.
        model_id (str): The identifier of the embedding model to use.
    
    Returns:
        Callable: The embedding function that can be used by FAISS.
    """
    embedding_function = BedrockEmbeddings(client=bedrock_client, model_id=model_id)
    return embedding_function.embed_query


def create_langchain_faiss_vectorstore(text_embeddings_path, metadata_path, embedding_function):
    """
    Create FAISS vectorstore from pre-computed embeddings.
    
    Args:
        text_embeddings_path: Path to JSON file with [[text, vector], ...] format.
        metadata_path: Path to JSON file with metadata list.
        embedding_function: Function to embed queries (not used for pre-computed docs).
    
    Returns:
        FAISS vectorstore instance.
    """
    # Load pre-computed embeddings and metadata
    text_embeddings = json.load(open(text_embeddings_path))
    text_embeddings = [(text, np.array(vec)) for text, vec in text_embeddings]
    metadata_list = json.load(open(metadata_path))
    
    # Create FAISS vectorstore
    vectorstore = FAISS.from_embeddings(
        text_embeddings=text_embeddings,
        embedding=embedding_function,
        metadatas=metadata_list,
        distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT,
        relevance_score_fn=lambda distance: (distance + 1.0) / 2.0
    )
    
    return vectorstore


def load_json(json_file):
    with open(json_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_metadata(metadata, filename):
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=4)


def save_embeddings(text_embeddings, text_embeddings_file):
    # Extract just the embeddings (not the text)
    embeddings = [emb for text, emb in text_embeddings]
    with open(text_embeddings_file, 'w') as f:
        json.dump(embeddings, f)


def create_vector_db_data(json_file, text_embeddings_file, metadata_file):
    model_id = "amazon.titan-embed-text-v1"
    bedrock_client = boto3.client('bedrock-runtime', region_name='us-east-1')
    data = load_json(json_file)
    
    text_embeddings = []
    metadata = []
    
    for project in data:
        project_name = project['project_name']
        print(project_name)
        
        for md_file in project['md']:
            for filename, content in md_file.items():
                text = f"Project Name: {project_name}\n{content}"
                
                try:
                    # Get embedding using helper function
                    embedding = get_embedding(text, bedrock_client, model_id)
                    text_embeddings.append((text, embedding))
                    metadata.append({'project_name': project_name, 'filename': filename, 'text': text})
                except Exception as e:
                    print(f"Failed to embed file {filename} from project {project_name}: {e}")
                    continue
    
    assert len(metadata) == len(text_embeddings)
    save_metadata(metadata, metadata_file)
    save_embeddings(text_embeddings, text_embeddings_file)


def main():
    model_id = "amazon.titan-embed-text-v1"
    json_file = 'data/DS_Projects_Docs.json'
    text_embeddings_file = 'data/text_embeddings.json'
    metadata_file = 'data/metadata.json'
    
    create_vector_db_data(json_file, text_embeddings_file, metadata_file)
    
    print(f"\nCreated:")
    print(f"  - {text_embeddings_file}")
    print(f"  - {metadata_file}")


if __name__ == "__main__":
    main()
