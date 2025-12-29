from langchain_ollama import ChatOllama
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings

file_path = "./nke-10k-2023.pdf"
loader = PyPDFLoader(file_path)
docs = loader.load()
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200, add_start_index=True
)
all_splits = text_splitter.split_documents(docs)

print(len(all_splits))
# llm = ChatOllama(
#         model="deepseek-r1:1.5b",
#         temperature=0,
#         base_url="http://localhost:11434"
#         )
#
# prompt = "who is the document from"
# print(docs[0].page_content[:200])
# msgs=[
#         {"role":"system","content":"You are an ai assistant that explains the docs"},
#         {"role":"system","content":docs[0].page_content[:200]},
#         {"role":"user","content":prompt}
#         ]
# resp=llm.invoke(msgs).content
# print(resp)
#
