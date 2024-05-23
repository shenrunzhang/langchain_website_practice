# Langchain Practice

### Setup

- Create a virtual environment with Python 3.10.
- Install prerequisites
    - `pip install -r requirements.txt`

- Add OpenAI and HuggingFace API keys in `.env`
    - `OPENAI_API_KEY=` and `HUGGINGFACEHUB_API_TOKEN`

### RUN

- Run `streamlit run src/web_app.py`

### Notes

- Supports RAG for multiple websites and uploaded PDF documents. 
- Currently uses ChatGPT API as the LLM and for embedding, but this can be adapted to locally run models from HuggingFace.
- External media (websites and PDF documents) are read, chunked, embedded, and stored in a vector database. I used Facebook AI Similarity Search (Faiss) for vector storage. When the user asks a question, the LLM queries the database to fetch the chunks of context that are relevant to the user's input. With the aid of the larger context, the LLM then answers the user's question in full.

### References

Code heavily adapted from 
- https://github.com/alejandro-ao/chat-with-websites/blob/master/src/app.py
- https://github.com/alejandro-ao/ask-multiple-pdfs/tree/main