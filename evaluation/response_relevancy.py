import os
from dotenv import load_dotenv
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from ragas.metrics import ResponseRelevancy
from ragas import SingleTurnSample

load_dotenv()

os.environ["AZURE_OPENAI_API_KEY"] = os.getenv("AZURE_OPENAI_API_KEY")
os.environ["AZURE_OPENAI_ENDPOINT"] = os.getenv("AZURE_OPENAI_ENDPOINT")
os.environ["OPENAI_API_VERSION"] = os.getenv("OPENAI_API_VERSION")

evaluator_llm = LangchainLLMWrapper(AzureChatOpenAI(
    azure_deployment=os.getenv("LLM_DEPLOYMENT_NAME"),
    temperature=0.8,
))

evaluator_embeddings = LangchainEmbeddingsWrapper(AzureOpenAIEmbeddings(
    azure_deployment=os.getenv("EMBEDDING_DEPLOYMENT_NAME"),
))

single_turn_sample = SingleTurnSample(
    user_input="富士山は何県にある山で、標高は何mですか？",
    response = "富士山は日本の山梨県と静岡県にまたがって位置しており、標高は約3,776メートルで、日本最高峰の山です。",
)

metric = ResponseRelevancy(llm=evaluator_llm, embeddings=evaluator_embeddings)
score = metric.single_turn_score(single_turn_sample)

print(f"{metric.name} : {score}")