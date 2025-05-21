import os
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential
from azure.search.documents.indexes.models import *
from langchain.text_splitter import RecursiveCharacterTextSplitter
from azure.ai.documentintelligence import DocumentIntelligenceClient
from dotenv import load_dotenv
import time

# 環境変数からAzure AI Searchのエンドポイント等を取得する
load_dotenv()
search_endpoint = os.environ["SEARCH_SERVICE_ENDPOINT"]
search_api_key = os.environ["SEARCH_API_KEY"]

filepath = "../rag_source_docs/syugyo-kisoku.pdf"  # 対象ファイルパス

#インデックスの作成
def create_index():
    client = SearchIndexClient(endpoint= search_endpoint, credential=AzureKeyCredential(search_api_key))
    name = "docs_di_1500"

    # すでにインデックスが作成済みである場合には何もしない
    if 'docs_di_1500' in client.list_index_names():
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
            content_type="application/octet-stream",
            output_content_format="markdown"
        )
        content = poller.result().content
    return content

def create_chunk(content: str, separator: str, chunk_size: int = 512, overlap: int = 0):
    splitter = RecursiveCharacterTextSplitter(chunk_overlap=overlap, chunk_size=chunk_size, separators=separator)
    chunks = splitter.split_text(content)
    return chunks


def index_docs(chunks: list):
    searchClient = SearchClient(
        endpoint=search_endpoint,
        index_name="docs_di_1500",
        credential=AzureKeyCredential(search_api_key)
    )

    # チャンク化されたテキストとそのテキストのベクトルをAzure AI Searchにアップロードする
    for i, chunk in enumerate(chunks):
        document = {"id": str(i), "content": chunk}
        searchClient.upload_documents([document])
        print(i)

def main():
    # チャンク化のパラメータ
    chunksize = 1500  # チャンクサイズ
    overlap = 128  # オーバーラップサイズ
    separator = ["\n\n", "\n", "。", "、", " ", ""]  # 区切り文字

    # インデックスを作成する
    create_index()

    # 対象ファイルパスのファイルを読み込んで、テキストを抽出する
    content = extract_text_from_docs(filepath)
    
    # テキストを指定したサイズで分割する
    chunks = create_chunk(content, separator, chunksize, overlap)
    
    # テキストをAzure AI Searchにインデックスする
    index_docs(chunks)

if __name__ == '__main__':
    main()