from langchain_ollama import ChatOllama
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore

## pdf loading
file_path = "./nke-10k-2023.pdf"
loader = PyPDFLoader(file_path)
docs = loader.load()

## splitting
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200, add_start_index=True
)
all_splits = text_splitter.split_documents(docs)

## embedding
# embeddings = OllamaEmbeddings(model="deepseek-r1:8b")
embeddings = OllamaEmbeddings(model="deepseek-r1:1.5b")
vector_1 = embeddings.embed_query(all_splits[0].page_content)
vector_2 = embeddings.embed_query(all_splits[1].page_content)

assert len(vector_1) == len(vector_2)

## Vectore stores
vector_store = InMemoryVectorStore(embeddings)
ids = vector_store.add_documents(documents=all_splits)
# results = vector_store.similarity_search(
#     "How many distribution centers does Nike have in the US?"
# )
#
# print(results[0])
embedding = embeddings.embed_query("How were Nike's margins impacted in 2023?")

results = vector_store.similarity_search_by_vector(embedding)
print(results[0])
# results = vector_store.similarity_search_with_score("What was Nike's revenue in 2023?")
# doc, score = results[0]
# print(f"Score: {score}\n")
# print(doc)
