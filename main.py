from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain.agents import create_agent
from langchain_core.vectorstores import InMemoryVectorStore
import bs4
from langchain.agents.middleware import dynamic_prompt, ModelRequest
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.tools import tool
from langchain.chat_models import init_chat_model

model = init_chat_model(
        model_provider="ollama",
        model="qwen2.5:7b",
        temperature=0,
        base_url="http://localhost:11434"
        )

embeddings = OllamaEmbeddings(model="qwen2.5:7b")
vector_store = InMemoryVectorStore(embeddings)

## Loading Document
# Only keep post title, headers, and content from the full HTML.
bs4_strainer = bs4.filter.SoupStrainer(class_=("post-title", "post-header", "post-content"))
loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs={"parse_only": bs4_strainer},
)
docs = loader.load()

assert len(docs) == 1

## Splitting Document
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,  # chunk size (characters)
    chunk_overlap=200,  # chunk overlap (characters)
    add_start_index=True,  # track index in original document
)
all_splits = text_splitter.split_documents(docs)

## Storing Document
document_ids = vector_store.add_documents(documents=all_splits)

@dynamic_prompt
def prompt_with_context(request: ModelRequest) -> str:
    """Inject context into state messages."""
    last_query = request.state["messages"][-1].text
    retrieved_docs = vector_store.similarity_search(last_query)

    docs_content = "\n\n".join(doc.page_content for doc in retrieved_docs)

    system_message = (
        "You are a helpful assistant. Use the following context in your response:"
        f"\n\n{docs_content}"
    )

    return system_message


agent = create_agent(model, tools=[], middleware=[prompt_with_context])
query = "What is task decomposition?"
for step in agent.stream(
    {"messages": [{"role": "user", "content": query}]},
    stream_mode="values",
):
    step["messages"][-1].pretty_print()
