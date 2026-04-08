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




    #synthetic Questions:
    # 1. Why are electric vehicles considered more environmentally friendly than internal combustion engine vehicles?
    # 2.What role does fossil fuel depletion play in the growth of electric vehicles?
    # 3.How does Tesla contribute to innovation in the automotive industry?
    # 4.Why is the EV industry expected to grow rapidly in the future?
    # 5. What factors are driving the global demand for electric vehicles?
    # 6.What are Tesla’s main strengths in the electric vehicle industry?
    # 7.What weaknesses does Tesla face in terms of technology and finances?
    # 8.How does Tesla’s Gigafactory help reduce manufacturing costs?
    # 9.What risks does Tesla face due to its high debt and investment spending?
    # 10.How does Tesla’s pricing strategy help increase its market share?