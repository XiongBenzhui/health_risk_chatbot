from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
import torch
import json
import os
 
Embedding_Model_path = 'models/multilingual_e5_large_instruct'
CUDA_Devices = ['cuda:0', 'cuda:1'] 

torch.cuda.set_device(1)  # change to your own device

def find_max_word_count(docs):
    max_word_count = max(len(doc.split()) for doc in docs)
    print(f"Maximum word count in a single document: {max_word_count}")
    return max_word_count + 100  

def load_cleaned_data_from_json(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    preprocessed_texts = []
    for title, content in data.items():
        compound_info = f"This information provides a complete description of one chemical compound:\n"
        text_content = "\n".join(filter(None, [
            f"The compound is titled '{title}' and is described as: {content.get('description', 'No description available.')}.",
            f"{title} is commonly known as: {', '.join(content.get('synonyms', []))}." if content.get('synonyms', []) else "",
            f"{title} belongs to the category: {content.get('category', 'No category information available.')}.",
            f"{title}'s sources and common uses include: {content.get('sources_uses', 'No specific sources or uses available.')}.",
            f"Comments related to {title}'s safety or effects: {content.get('comments', 'No additional comments available.')}.",
            f"{title}'s known restrictions: {content.get('restrictions', 'No restrictions mentioned.')}.",
            f"Occupational diseases associated with {title} include: {', '.join(content.get('diseases', []))}." if content.get('diseases', []) else "",
            f"Industrial processes involving {title} include: {', '.join(content.get('processes', []))}." if content.get('processes', []) else "",
            f"Activities linked to {title} include: {', '.join(content.get('activities', []))}." if content.get('activities', []) else "",
            f"Key chemical properties of {title} include: PLV of {content.get('plv', 'N/A')}, KH of {content.get('kh', 'N/A')}, KOC of {content.get('koc', 'N/A')}, "
            f"lipophilicity (Lipo) of {content.get('lipo', 'N/A')}, solubility (SW) of {content.get('sw', 'N/A')}, mutagenicity index (Muta) of {content.get('muta', 'N/A')}, "
            f"hydrogen bond acceptor count of {content.get('hacceptor', 'N/A')}, and topological polar surface area (TPSA) of {content.get('tpsa', 'N/A')}."
        ]))

        cleaned_text = compound_info + text_content
        word_count = len(cleaned_text.split())
        print(f"Total word count: {word_count}")
        preprocessed_text = cleaned_text.strip()
        preprocessed_texts.append(preprocessed_text)
    return preprocessed_texts

def split_text(txt, chunk_size=1024, overlap=0):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap, separators=["\n\n", "\n"])
    docs = splitter.split_text(txt)
    return docs

def create_embeddings():
    embeddings = HuggingFaceEmbeddings(
        model_name=Embedding_Model_path, 
        model_kwargs={'device': CUDA_Devices[0]}  # change to your CUDA_Devices
    )
    return embeddings

def create_vector_store(docs, embeddings, store_path):
    vector_store = FAISS.from_texts(docs, embeddings)
    vector_store.save_local(store_path)
    return vector_store

def load_vector_store(store_path, embeddings):
    if os.path.exists(store_path):
        vector_store = FAISS.load_local(store_path, embeddings, 
            allow_dangerous_deserialization=True)
        return vector_store
    else:
        return None

def load_or_create_vector_store(store_path, json_file_path):
    embeddings = create_embeddings()
    vector_store = load_vector_store(store_path, embeddings)
    if not vector_store:
        preprocessed_texts = load_cleaned_data_from_json(json_file_path)
        max_word_count = find_max_word_count(preprocessed_texts)
        print(f'Max word count: {max_word_count}')
        docs = []
        for text in preprocessed_texts:
            docs.extend(split_text(text))
        vector_store = create_vector_store(docs, embeddings, store_path)
    return vector_store

json_file_path = 'data/Health_risk_knowledge_base_raw.json'  # use your own knowledge base file 
vector_store_path = 'data/Health_risk_knowledge_base.faiss'

vector_store = load_or_create_vector_store(vector_store_path, json_file_path)



