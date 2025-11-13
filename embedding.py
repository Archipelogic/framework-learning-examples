#!/usr/bin/env python3
"""
Create embeddings from DS_Projects_Docs.json and save to files
Run this BEFORE running crewai_unified.py
"""

import json
import boto3
import numpy as np
from pathlib import Path


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
                    # Get embedding using Bedrock
                    body = json.dumps({"inputText": text})
                    response = bedrock_client.invoke_model(
                        modelId=model_id,
                        body=body,
                        accept="application/json",
                        contentType="application/json"
                    )
                    response_body = json.loads(response['body'].read())
                    embedding = response_body['embedding']
                    
                    # Convert to numpy float32 array
                    embedding = np.array(embedding, dtype=np.float32)
                    
                    text_embeddings.append((text, embedding.tolist()))
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
