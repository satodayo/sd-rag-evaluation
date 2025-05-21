import os
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential
from azure.search.documents.indexes.models import *
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import AzureOpenAIEmbeddings
from langchain_experimental.text_splitter import SemanticChunker
from dotenv import load_dotenv
from azure.ai.documentintelligence import DocumentIntelligenceClient

filepath = "../rag_source_docs/syugyo-kisoku.pdf"  # 対象ファイルパス

# .envファイルの読み込み
load_dotenv()

search_endpoint = os.environ["SEARCH_SERVICE_ENDPOINT"]
search_api_key = os.environ["SEARCH_API_KEY"]

#インデックスの作成
def create_index():
    client = SearchIndexClient(endpoint= search_endpoint, credential=AzureKeyCredential(search_api_key))
    name = "docs_semantic_chunking"

    # すでにインデックスが作成済みである場合には何もしない
    if 'docs_semantic_chunking' in client.list_index_names():
        print("Already index exists")
        return

    # インデックスのフィールドを定義する
    fields = [
        SimpleField(name="id", type=SearchFieldDataType.String, key=True),
        SearchableField(name="content", type="Edm.String", analyzer_name="ja.microsoft"),
    ]

    # インデックスを作成する
    index = SearchIndex(name=name, fields=fields)
    client.create_index(index)

def extract_text_from_docs(filepath):
    load_dotenv()  # Load environment variables from .env file
    endpoint = os.getenv("AZURE_FORM_RECOGNIZER_ENDPOINT")
    key = os.getenv("AZURE_FORM_RECOGNIZER_KEY")

    client = DocumentIntelligenceClient(endpoint=endpoint, credential=AzureKeyCredential(key))
    with open(filepath, "rb") as f:
        poller = client.begin_analyze_document(
            model_id="prebuilt-layout",
            body=f,
            content_type="application/octet-stream"
        )
        content = poller.result().content
    return content

def create_chunk(content):
    text_splitter = SemanticChunker(
        LangchainEmbeddingsWrapper(
            AzureOpenAIEmbeddings(
                azure_deployment=os.getenv("EMBEDDING_DEPLOYMENT_NAME")
            )
        ),
        sentence_split_regex=r"(?<=[。！？\.\?\\n])\s*|\n",  # Adjusted regex for Japanese sentence boundaries
        min_chunk_size=100
    )
    docs = text_splitter.create_documents([content])
    return docs

def index_docs(chunks: list):
    searchClient = SearchClient(
        endpoint=search_endpoint,
        index_name="docs_semantic_chunking",
        credential=AzureKeyCredential(search_api_key)
    )

    # チャンク化されたテキストとそのテキストのベクトルをAzure AI Searchにアップロードする
    for i, chunk in enumerate(chunks):
        document = {"id": str(i), "content": chunk.page_content}
        searchClient.upload_documents([document])
        print(i)

def main():
    # インデックスを作成する
    create_index()
    
    # 対象ファイルパスのファイルを読み込んで、DocumentIntelligenceを利用してテキストを抽出する
    content = extract_text_from_docs(filepath)
    
    # テキストをMarkdownHeaderTextSplitterでチャンク化する
    chunks = create_chunk(content)
    
    # テキストをAzure AI Searchにインデックスする
    index_docs(chunks)

if __name__ == '__main__':
    main()