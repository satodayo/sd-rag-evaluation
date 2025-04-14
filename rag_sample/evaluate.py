import os
from typing import Any
import asyncio
from dotenv import load_dotenv
from ragas.metrics.base import Metric
from langsmith.schemas import Example, Run
from langsmith.evaluation import aevaluate
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.metrics import ContextPrecision, Faithfulness
from ragas import SingleTurnSample
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_core.messages import SystemMessage
from langchain.schema import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain_community.retrievers import AzureAISearchRetriever

# .envファイルの読み込み
load_dotenv()

# 環境変数の設定
os.environ["AZURE_OPENAI_API_KEY"] = os.getenv("AZURE_OPENAI_API_KEY")
os.environ["AZURE_OPENAI_ENDPOINT"] = os.getenv("AZURE_OPENAI_ENDPOINT")
os.environ["OPENAI_API_VERSION"] = os.getenv("OPENAI_API_VERSION")
os.environ["AZURE_AI_SEARCH_ENDPOINT"] = os.getenv("SEARCH_SERVICE_ENDPOINT")
os.environ["AZURE_AI_SEARCH_API_KEY"] = os.getenv("SEARCH_API_KEY")

class RagasEvaluator:
    # 対象となるMetricsを設定
    def __init__(self, metrics:list[Metric]):
        self.metrics = metrics

    # 実際に引き渡す評価用の関数
    # runはtargetの返り値, exampleはDatasetに登録された値を示す
    async def evaluate(self, run: Run, example: Example) -> dict[str, Any]:
        context_strs = run.outputs["retrieved_contexts"] #RAGシステムから得られた関連情報

        single_turn_sample = SingleTurnSample(
            user_input=example.inputs["user_input"],#テストセットから得られた質問文
            retrieved_contexts=context_strs,
            response=run.outputs["response"], #RAGシステムから得られた回答
            reference=example.outputs["reference"] #テストセットから得られた真の回答
        )
        
        results = []
        # 各Metricに対して評価を実行
        for metric in self.metrics:
            score = await metric.single_turn_ascore(single_turn_sample)
            results.append({"key": metric.name, "score": score})
        
        return results

async def predict(inputs: dict[str, Any]) -> dict[str, Any]:
    user_input = inputs["user_input"]
    
    retriever = AzureAISearchRetriever(
        service_name="srch-sd-rag-evaluation",
        content_key="content",
        top_k=3, 
        index_name="docs_di_1500")
    
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
    answer = answer_chain.invoke(user_input)

    # contextを得るためのchain
    context_chain = retriever

    contexts = context_chain.invoke(user_input)
    retrieved_contexts = [c.page_content for c in contexts]

    return {
        "response": answer,
        "retrieved_contexts": retrieved_contexts,
    }

async def main():
    dataset_name = "syugyo-kisoku"
        
    evaluator_llm = LangchainLLMWrapper(AzureChatOpenAI(
        azure_deployment="gpt-4o-mini",
        temperature=0.8,
    ))

    evaluator_embeddings = LangchainEmbeddingsWrapper(AzureOpenAIEmbeddings(
        azure_deployment="text-embedding-3-small-deploy"
    ))

    # 評価を実行するMetricsを定義
    metrics = [
        ContextPrecision(llm=evaluator_llm),
        Faithfulness(llm=evaluator_llm)
    ]

    # 評価の実行
    await aevaluate(
        predict,
        data=dataset_name,
        evaluators=[
            RagasEvaluator(metrics).evaluate
        ],
    )

if __name__ == "__main__":
    asyncio.run(main())