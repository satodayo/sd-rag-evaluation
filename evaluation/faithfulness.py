import os
from dotenv import load_dotenv
from ragas.llms import LangchainLLMWrapper
from langchain_openai import AzureChatOpenAI
from ragas.metrics import Faithfulness
from ragas import SingleTurnSample

load_dotenv()

os.environ["AZURE_OPENAI_API_KEY"] = os.getenv("AZURE_OPENAI_API_KEY")
os.environ["AZURE_OPENAI_ENDPOINT"] = os.getenv("AZURE_OPENAI_ENDPOINT")
os.environ["OPENAI_API_VERSION"] = os.getenv("OPENAI_API_VERSION")

evaluator_llm = LangchainLLMWrapper(AzureChatOpenAI(
    azure_deployment=os.getenv("LLM_DEPLOYMENT_NAME"),
    temperature=0,
))

single_turn_sample = SingleTurnSample(
    user_input="富士山について教えてください。",
    retrieved_contexts=["富士山は山梨県と静岡県に跨る活火山である。標高3776.12m、日本最高峰（剣ヶ峰）の独立峰で、その優美な風貌は日本国外でも日本の象徴として広く知られている。",
                        "懸垂曲線の山容を有した玄武岩質成層火山で構成され、その山体は駿河湾の海岸まで及ぶ。",
                        "古来より霊峰とされ、特に山頂部は浅間大神が鎮座するとされたため、神聖視された。"],
    response = "富士山は山梨県と静岡県に跨る活火山で、標高は3776.12mです。日本最高峰（剣ヶ峰）の独立峰で、垂曲線の山容を有した玄武岩質成層火山で構成されており、山頂部には浅間大神が鎮座するとされています。2013年には世界文化遺産に登録されました。",
)

metric = Faithfulness(llm=evaluator_llm)
score = metric.single_turn_score(single_turn_sample)

print(f"{metric.name} : {score}")