# AI-Knowledge-Assistant-RAG-based-LLM-System-
Built a Retrieval-Augmented Generation (RAG) system that indexes documents using embeddings and  stores them in a vector database. When a user asks a question, the backend retrieves the most relevant documents  and sends them to an LLM to generate accurate contextual answers.
<br><br>
<b>Architecture:</b> Document ingestion → Embedding generation → Vector database storage → Semantic search → Context retrieval → LLM response generation. 
<br><br>
<b>requiremnets.txt</b>
pip install --upgrade langchain-chroma langchain-google-genai langchain-huggingface langchain-experimental
pip install --upgrade langchain-core langchain langchain-community langchain-text-splitters
