import os
from dotenv import load_dotenv
from ragas.llms import LangchainLLMWrapper
from langchain_openai import AzureChatOpenAI
from ragas.metrics import ContextPrecision
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
    user_input="日本で一番標高の高い山は何ですか？",
    retrieved_contexts=["富士山は山梨県と静岡県に跨る活火山である。標高 3776.12 m、日本最高峰の独立峰で、その優美な風貌は日本国外でも日本の象徴として広く知られている。",
                        "エベレスト（英: Everest）は、ヒマラヤ山脈にある世界最高峰の山である。山頂は、ネパールと中国・チベット自治区との国境上にある。"],
    reference="富士山"
)

metric = ContextPrecision(llm=evaluator_llm)
score = metric.single_turn_score(single_turn_sample)

print(f"{metric.name} : {score}")