import os
import argparse
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_core.messages import SystemMessage
from langchain.schema import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain_openai import AzureChatOpenAI
from langchain_community.retrievers import AzureAISearchRetriever

load_dotenv()
os.environ["AZURE_OPENAI_ENDPOINT"] = os.getenv("AZURE_OPENAI_ENDPOINT")
os.environ["AZURE_OPENAI_API_KEY"] = os.getenv("AZURE_OPENAI_API_KEY")
os.environ["OPENAI_API_VERSION"] = os.getenv("OPENAI_API_VERSION")
os.environ["AZURE_AI_SEARCH_ENDPOINT"] = os.getenv("SEARCH_SERVICE_ENDPOINT")
os.environ["AZURE_AI_SEARCH_API_KEY"] = os.getenv("SEARCH_API_KEY")

parser = argparse.ArgumentParser(description="回答を生成する")
parser.add_argument("user_input", type=str, help="質問内容")
args = parser.parse_args()

retriever = AzureAISearchRetriever(
    service_name="srch-sd-rag-evaluation",
    content_key="content",
    top_k=3, 
    index_name="docs")

prompt = ChatPromptTemplate.from_messages(
    [SystemMessage(
            """関連情報に基づき質問に回答してください。"""
    ),
        HumanMessagePromptTemplate.from_template(
            """ 関連情報：{context}
            
            ## 質問：{question}
            ## 回答： """
    )
        ]
    ) 

llm = AzureChatOpenAI(
    azure_deployment="gpt-4o-mini-deploy",
    temperature=0,
)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

answer_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)
answer = answer_chain.invoke(args.user_input)

# contextを得るためのchain
context_chain = retriever

contexts = context_chain.invoke(args.user_input)
retrieved_contexts = [c.page_content for c in contexts]
