from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_community.vectorstores.chroma import Chroma
from langchain_community.chat_models.ollama import ChatOllama
# from llama_cpp import Llama


from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

# # Create embeddingsclear
embeddings = OllamaEmbeddings(model="nomic-embed-text", show_progress=True)
# embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

db = Chroma(persist_directory="./db-mawared",
            embedding_function=embeddings)

# # Create retriever
retriever = db.as_retriever(
    search_type="similarity",
    search_kwargs= {"k": 5}
)


# # Create Ollama language model - Gemma 2



# # Create the LLM with HuggingFacePipeline
# llm = HuggingFacePipeline(pipeline=pipe)
local_llm = 'llama3.1'

llm = ChatOllama(model=local_llm,
                 keep_alive="3h", 
                 num_ctx=1024,  # Changed from max_tokens to num_ctx
                 temperature=0.8)

# Create prompt template
template = """You are a helpful assistant specialized in Mawared HR System . Use the following pieces of context to answer the question at the end. If you don't know the answer, Ask more questions to get more context.

Context:
{context}

Question: {question}

Answer:"""
prompt = ChatPromptTemplate.from_template(template)

# Create the RAG chain using LCEL with prompt printing and streaming output
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)


# Function to ask questions
def ask_question(question):
    print("Answer:\n\n", end=" ", flush=True)
    for chunk in rag_chain.stream(question):
        print(chunk, end="", flush=True)
    print("\n")

# Example usage
if __name__ == "__main__":
    while True:
        user_question = input("Ask a question (or type 'quit' to exit): ")
        if user_question.lower() == 'quit':
            break
        answer = ask_question(user_question)
        # print("\nFull answer received.\n")

