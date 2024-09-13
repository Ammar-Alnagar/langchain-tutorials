from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_community.vectorstores.chroma import Chroma
from langchain_community.chat_models.ollama import ChatOllama
# from llama_cpp import Llama


from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

# # Create embeddingsclear
embeddings = OllamaEmbeddings(model="nomic-embed-text", show_progress=False)
# embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

db = Chroma(persist_directory="./db-mawared",
            embedding_function=embeddings)

# Create retriever
retriever = db.as_retriever(
    search_type="similarity",
    search_kwargs= {"k": 4}
)




# # Create Ollama language model - Gemma 2



# # Create the LLM with HuggingFacePipeline
# llm = HuggingFacePipeline(pipeline=pipe)
local_llm = 'phi3.5'

llm = ChatOllama(
    model=local_llm,
    keep_alive="3h",
    num_ctx=2048,  # Context size (number of tokens in context)
    temperature=0.7,  # Sampling temperature, controls randomness
    top_p=0.95,  # Nucleus sampling (cumulative probability threshold)
    top_k=50,  # Top-k sampling (limits to top-k most likely tokens)
    repetition_penalty=1.5,  # Penalize repeating tokens # type: ignore  # Penalize frequent tokens # type: ignore
    use_cache=True,  # Use past conversation history # type: ignore
      # Beam search, controls exploration of multiple sequences # type: ignore
)

# Create prompt template
template = """
You are an expert assistant specializing in the Mawared HR System. Your role is to answer the user's question based strictly on the provided context. If the context does not contain the answer, you should ask clarifying questions to gather more information.

Make sure to:
1. Use only the provided context to generate the answer.
2. Be concise and direct.
3. If the context is insufficient, ask relevant follow-up questions instead of speculating.
4. Only answer from the context.

Context:
{context}

Question: {question}

Answer:
"""

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
    print("Answer:\t", end=" ", flush=True)
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

