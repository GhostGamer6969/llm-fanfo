from langchain_ollama import ChatOllama

llm = ChatOllama(
        model="deepseek-r1:1.5b",
        temperature=0,
        base_url="http://localhost:11434"
        )

prompt = "Hi! What's up?"
msgs=[
        {"role":"system","content":"You are an ai assistant"},
        {"role":"user","content":prompt}
        ]
resp=llm.invoke(msgs).content
print(resp)
