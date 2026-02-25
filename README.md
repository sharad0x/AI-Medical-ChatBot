# Medical-Chatbot

## Project Description
The **Medical Chatbot** is an end-to-end Generative AI application built using a **Retrieval-Augmented Generation (RAG)** pipeline. It is designed to act as a virtual medical assistant, providing accurate, context-aware responses to user queries based on a specialized medical knowledge base. By leveraging **LangChain** for orchestration and the powerful **Llama 3.3 70B** model via the **NVIDIA NIM API**, the system ensures highly relevant and intelligent conversational outputs.

---

## How to run?
### STEPS:

Clone the repository

```bash
git clonehttps://github.com/entbappy/Build-a-Complete-Medical-Chatbot-with-LLMs-LangChain-Pinecone-Flask-AWS.git
```
### STEP 01- Create a conda environment after opening the repository

```bash
conda create -n medibot python=3.10 -y
```

```bash
conda activate medibot
```


### STEP 02- install the requirements
```bash
pip install -r requirements.txt
```


### Create a `.env` file in the root directory and add your Pinecone & openai credentials as follows:

```ini
PINECONE_API_KEY = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
NVIDIA_API_KEY = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
```


```bash
# run the following command to store embeddings to pinecone
python store_index.py
```

```bash
# Finally run the following command
python app.py
```

Now,
```bash
open up localhost:
```


## Key Features
* **RAG Architecture:** Eliminates LLM hallucinations by grounding answers in a reliable medical document corpus.
* **Semantic Search:** Uses advanced vector embeddings to understand the context of user questions rather than just keyword matching.
* **Scalable Vector Database:** Integrates **Pinecone** for fast and efficient high-dimensional data retrieval.
* **Interactive UI:** Features a responsive web interface deployed using **Flask**.

## Tech Stack
* **Language:** Python
* **Generative AI & LLM:** Llama 3.3 70B (via NVIDIA NIM API)
* **LLM Orchestration:** LangChain
* **Vector Database:** Pinecone DB
* **Embeddings:** Sentence-Transformers (e.g., `all-MiniLM-L6-v2`)
* **Backend Framework:** Flask

## How It Works (The Pipeline)
1. **Data Ingestion:** Medical documents (PDFs) are loaded and processed.
2. **Text Chunking:** The text is split into smaller, manageable chunks using LangChain's text splitters.
3. **Embedding Generation:** Text chunks are converted into vector embeddings.
4. **Vector Storage:** Embeddings are indexed and stored in Pinecone for rapid retrieval.
5. **Query Processing:** When a user asks a question, it is converted into a vector.
6. **Retrieval & Generation:** Pinecone retrieves the most relevant chunks, which are passed to the Llama 3.3 model as context to generate an accurate, conversational response.

---