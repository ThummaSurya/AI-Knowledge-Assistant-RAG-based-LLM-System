from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()

persistent_directory = "db/chroma_db"

#Load embeddings and vectorstore
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

db = Chroma(
    persist_directory=persistent_directory,
    embedding_function=embedding_model,
    collection_metadata={"hnsw:space": "cosine"}
)

#search for relevant documents
query = "Why are electric vehicles considered more environmentally friendly than internal combustion engine vehicles?"

retriever = db.as_retriever(search_kwargs={"k": 3})

# retriever = db.as_retriever(
#     search_type="similarity_score_threshold",
#     search_kwargs={
#         "k": 5,
#         "score-threshold": 0.3 #only return chunks with cosine similarity>=0.3
#     }
# )

relevant_docs = retriever.invoke(query)

print(f"User Query: {query}")
#Display results
print("---context---")
for i, doc in enumerate(relevant_docs, 1):
    print(f"Document {i}:\n{doc.page_content}\n")




#Combine the query and the relevant document contents
combined_input = f"""Based on the following documents, please answer this question{query}

   Documents:
   {chr(10).join([f"- {doc.page_content}" for doc in relevant_docs])}

   Please provide a clear, helpful answer using only the information from these documents.
   If you can't find the answer in the documents, say "I don't have enough information to answer the 
   question on the provided documents.
"""

#Create an llm

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.3
)

#Define the message for the model
messages = [
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content=combined_input),
]

#Invoke the model with the combined input
response = llm.invoke(messages)

#Display the full result and content only
print("\n--- Generated Response ---")

print("Content only:")
print(response.content)
