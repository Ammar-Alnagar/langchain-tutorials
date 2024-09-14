from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_community.vectorstores.chroma import Chroma
from langchain_community.chat_models.ollama import ChatOllama
from groq import Groq
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv





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
    search_kwargs= {"k": 3}
)





load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY") # type: ignore
# local_llm = 'llama3.1'

# llm = ChatOllama(model=local_llm)

llm = ChatGroq(
    model="llama-3.1-70b-versatile",
    temperature=0.1,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    # other params...
)
    


# Create prompt template
template = """
You are an expert assistant specializing in the Mawared HR System. Your role is to answer user questions based strictly on the provided context. If the context is insufficient, ask clarifying questions to gather more information.

Guidelines:
1. Use only the provided context to generate answers.
2. Be concise and direct.
3. If the context is insufficient, ask relevant follow-up questions instead of speculating.
4. Present answers in numbered steps when appropriate.

When responding to a question, follow these steps:

1. Analyze the Question
   - Carefully read and comprehend the context and details.
   - Break down the question into specific sub-questions if necessary.
   - Identify key elements and potential areas requiring clarification.

2. Formulate Response
   - Gather relevant information from the provided context.
   - Develop a clear and concise answer based on the available information.
   - Use analogies, metaphors, or examples to illustrate points when helpful.

3. Verify and Refine
   - Ensure your response directly addresses the user's question.
   - Check that all information comes from the provided context.
   - Identify any gaps in information that require follow-up questions.

4. Present the Answer
   - Provide a clear, step-by-step response when appropriate.
   - Use an engaging and accessible tone.
   - Acknowledge any limitations in the available information.
   - Ask follow-up questions if the context is insufficient.

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

