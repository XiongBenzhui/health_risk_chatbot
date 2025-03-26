import os
import re
import torch
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import time
from flask import Flask, render_template, request
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langchain_core.runnables import RunnableSequence
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1" # change to your GPU number

app = Flask(__name__, static_folder='./static')

Embedding_Model_path = 'models/multilingual_e5_large_instruct'
vector_store_path = 'data/Health_risk_knowledge_base.faiss'

LLM_Model = ChatOllama(
model="deepseek-r1:32b",
top_p=1.0,
temperature=1.0,
repeat_penalty=1.0,
device_map="auto",  
load_in_8bit=True, 
torch_dtype=torch.float16,
low_cpu_mem_usage=True, 
num_gpu=81,  # change to your num_gpu 
)

def create_embeddings():
    embeddings = HuggingFaceEmbeddings(
        model_name=Embedding_Model_path, 
        model_kwargs={
            'device': 'cuda'
        }
    )
    return embeddings

def load_vector_store(store_path):
    embeddings = create_embeddings()
    if os.path.exists(store_path):
        vector_store = FAISS.load_local(store_path, embeddings, 
            allow_dangerous_deserialization=True)
        return vector_store
    else:
        return None

def query_vector_store(vector_store, query, k, relevance_threshold):
    similar_docs = vector_store.similarity_search_with_relevance_scores(query, k=k)
    
    related_docs = list(filter(lambda x: x[1] > relevance_threshold, similar_docs))
    combined_context = "\n\n".join([doc[0].page_content for doc in related_docs])
    return combined_context

def extract_thoughts_and_answer(response):
    print(f'Response: {response}')
    response = response.content   
    print(f'Response content: {response}')
    thoughts_match = re.search(r'<think>(.*?)</think>', response, re.DOTALL)
    answer_match = re.search(r'</think>(.*)', response, re.DOTALL)
    if thoughts_match:
        thoughts = thoughts_match.group(1).strip() 
    else:
        thoughts = "No thought process found."

    if answer_match:
        answer = answer_match.group(1).strip()
    else:
        answer = "No final answer found."

    return thoughts, answer

LLM_TEMPLATE = """
You are a knowledgeable assistant specialized in health risks, with expertise in chemical properties and their effects on human health and the environment.

Now, answer the following question, aiming for at least 500 words:

{question}
"""

def get_LLM_response(LLM_TEMPLATE, quest):
    llm_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a friendly and knowledgeable assistant specialized in health risks."),
        ("human", LLM_TEMPLATE)
    ])
    
    llm_chain = RunnableSequence(llm_prompt | LLM_Model)
    start_time = time.time()
    response = llm_chain.invoke(
        {
            "question": quest,
            "max_tokens": 1024
        })
    thoughts, answer = extract_thoughts_and_answer(response)
    duration = time.time() - start_time
    return thoughts, answer, llm_prompt.format(question=quest), duration

LLM_RAG_FINAL_TEMPLATE = """
You are a knowledgeable assistant specialized in health risks, with expertise in chemical properties and their effects on human health and the environment.

Please carefully read and analyze the following context:

<context>
{context}
</context>

After reading the above context, answer the question by thoroughly considering these chemical properties and integrating them into your analysis:
- **PLV, KH, KOC, Lipo, SW, Muta, TPSA**: These key properties influence how the compound behaves in biological and environmental systems. Explain how each of these properties impacts its bioaccumulation, toxicity, and overall health risk profile.
- Use relevant numerical data such as **LD50**, **exposure limits**, and any other available figures to quantify the health risks in both immediate and long-term exposure scenarios.

Your answer should:
- **Be Complete**: Address all parts of the question in detail, explaining each aspect of the health risk in-depth.
- **Be Quantitative**: Provide specific numerical data to support your analysis, such as concentration thresholds, toxicity levels, and any other relevant measures.
- **Be Comprehensive**: Discuss both immediate and long-term risks. Consider the effects of exposure on different populations (e.g., workers, vulnerable communities) and ecosystems. Provide a detailed analysis of the potential long-term environmental and health consequences of exposure.
- **Provide Detailed Explanations**: For each health risk or suggestion, explain how it arises from the chemical properties and processes involved, discussing the mechanisms behind toxicity and how the chemical interacts with biological systems or the environment.
- **Be Creative**: Propose alternative substances, processes, or strategies that could reduce the identified health risks. Offer practical solutions based on the available data and knowledge.

Now, answer the following question, aiming for at least 500 words:

{question}
"""

def get_LLM_RAG_response(LLM_RAG_TEMPLATE, quest, vector_store):
    context = query_vector_store(vector_store, quest, k=10, relevance_threshold=0.7)
    rag_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a friendly and knowledgeable assistant specialized in health risks."),
        ("human", LLM_RAG_TEMPLATE)
    ])

    llm_chain = RunnableSequence(rag_prompt | LLM_Model)
    start_time = time.time()
    response = llm_chain.invoke(
        {
        "context": context, 
        "question": quest,
        "max_tokens": 1024
            })
                
    thoughts, answer = extract_thoughts_and_answer(response)
    duration = time.time() - start_time
    return thoughts, answer, rag_prompt.format(question=quest, context=context), duration

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        quest = request.form.get('question', None)  
        if not quest:
            return "Question not provided", 400 

        vector_store = load_vector_store(vector_store_path)
        
        if 'ask_llm' in request.form:
            llm_thought, llm_response, prompt_llm, duration_llm = get_LLM_response(LLM_TEMPLATE, quest)
            print(f'llm_response: {llm_response}')
            return render_template('index.html', question=quest, response=llm_response, thoughts=llm_thought)
        elif 'ask_llm_rag' in request.form:
            llm_rag_thought, llm_rag_response, llm_rag_prompt, llm_rag_duration = get_LLM_RAG_response(LLM_RAG_FINAL_TEMPLATE, quest, vector_store)
            print(f'llm_rag_response: {llm_rag_response}')
            return render_template('index.html', question=quest, response=llm_rag_response, thoughts=llm_rag_thought)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)

