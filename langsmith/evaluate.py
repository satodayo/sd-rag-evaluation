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
from ragas.metrics import ContextPrecision, ResponseRelevancy
from ragas import SingleTurnSample

# .envファイルの読み込み
load_dotenv()

# 環境変数の設定
os.environ["AZURE_OPENAI_API_KEY"] = os.getenv("AZURE_OPENAI_API_KEY")
os.environ["AZURE_OPENAI_ENDPOINT"] = os.getenv("AZURE_OPENAI_ENDPOINT")
os.environ["OPENAI_API_VERSION"] = os.getenv("OPENAI_API_VERSION")

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
    return {
        # RAGからの回答(ダミー)
        "response": "富士山は山梨県と静岡県に跨る活火山で、標高は3776.12mです。日本最高峰（剣ヶ峰）の独立峰で、垂曲線の山容を有した玄武岩質成層火山で構成されており、山頂部には浅間大神が鎮座するとされています。2013年には世界文化遺産に登録されました。",
        # 参考にした関連情報(ダミー)
        "retrieved_contexts": [
            "富士山は山梨県と静岡県に跨る活火山である。標高3776.12m、日本最高峰（剣ヶ峰）の独立峰で、その優美な風貌は日本国外でも日本の象徴として広く知られている。",
            "懸垂曲線の山容を有した玄武岩質成層火山で構成され、その山体は駿河湾の海岸まで及ぶ。"
        ],
    }


async def main():
    dataset_name = os.getenv("DATASET_NAME")
        
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
        ResponseRelevancy(llm=evaluator_llm, embeddings=evaluator_embeddings)
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