import os
import asyncio
from dotenv import load_dotenv
from langsmith import Client
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.testset import TestsetGenerator
from ragas.testset.persona import Persona
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_community.document_loaders import DirectoryLoader
from ragas.testset.synthesizers import SingleHopSpecificQuerySynthesizer
from ragas.testset.transforms.extractors.llm_based import NERExtractor
from ragas.testset.transforms.splitters import HeadlineSplitter

load_dotenv()

os.environ["AZURE_OPENAI_API_KEY"] = os.getenv("AZURE_OPENAI_API_KEY")
os.environ["AZURE_OPENAI_ENDPOINT"] = os.getenv("AZURE_OPENAI_ENDPOINT")
os.environ["OPENAI_API_VERSION"] = os.getenv("OPENAI_API_VERSION")

async def main():
    
    generator_llm = LangchainLLMWrapper(AzureChatOpenAI(
        azure_deployment="gpt-4o-mini",
        temperature=0.8,
    ))

    generator_embeddings = LangchainEmbeddingsWrapper(AzureOpenAIEmbeddings(
        azure_deployment="text-embedding-3-small-deploy"
    ))

    generator = TestsetGenerator(llm=generator_llm, embedding_model=generator_embeddings)

    # 対象ファイルの読み込み
    loader = DirectoryLoader("source_docs/", glob="fujisan.txt")
    docs = loader.load()
    
    personas = [
        Persona(
            name="Tourist",
            role_description="富士山の見どころや、周辺の観光情報を知りたい観光客",
            )
    ]
    
    # Generator初期化時に各モデルとペルソナ情報を引数として渡す
    generator = TestsetGenerator(
        llm=generator_llm, embedding_model=generator_embeddings, persona_list=personas
    )
    
    transforms = [HeadlineSplitter(), NERExtractor(llm=generator_llm)]
    

    distribution = [
        (SingleHopSpecificQuerySynthesizer(llm=generator_llm), 1.0),
    ]

    # 日本語化対応
    for query, _ in distribution:
        prompts = await query.adapt_prompts("japanese", llm=generator_llm)
        query.set_prompts(**prompts)

    # テストセット生成
    dataset = generator.generate_with_langchain_docs(
        docs[:],
        testset_size=2, #生成するテストセットの数
        transforms=transforms,
        query_distribution=distribution,
    )   

    df = dataset.to_pandas()
    dataset_name = os.getenv("DATASET_NAME")
    
    # LangSmithクライアントを構築
    client = Client()
    if client.has_dataset(dataset_name=dataset_name):
        client.read_dataset(dataset_name=dataset_name)
    else:
        client.create_dataset(dataset_name=dataset_name)
    
    # 利用するパラメータの初期化
    inputs = []
    outputs = []
    metadatas = []

    # Ragasで生成したテストセット(df)から内容を読み取り、パラメータに追加
    for _, testset_record in df.iterrows():
        inputs.append(
            {
                "user_input": testset_record['user_input']
            }
        )
        outputs.append(
            {
                "reference_contexts": testset_record['reference_contexts'],
                "reference": testset_record['reference']
            }
        )
        # metadataとして今回はRagasの持つsynthesizerの情報を付与
        metadatas.append(
            {
                "synthesizer_name": testset_record['synthesizer_name']
            }
        )
        
     # Datasetへの登録   
    client.create_examples(
        inputs=inputs,
        outputs=outputs,
        metadata=metadatas,
        dataset_name=dataset_name
    )


if __name__ == "__main__":
    asyncio.run(main())