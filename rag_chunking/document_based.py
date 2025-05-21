import os
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential
from azure.search.documents.indexes.models import *
from langchain.text_splitter import MarkdownHeaderTextSplitter
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
    name = "docs_document_based2"

    # すでにインデックスが作成済みである場合には何もしない
    if 'docs_document_based' in client.list_index_names():
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

def create_chunk(content):
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on)
    md_header_splits = markdown_splitter.split_text(content)
    return md_header_splits

def index_docs(chunks: list):
    searchClient = SearchClient(
        endpoint=search_endpoint,
        index_name="docs_document_based2",
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

    # チャンク化したファイルをローカルの/chuncked_documentフォルダに保存する
    output_dir = "./chunked_document"
    os.makedirs(output_dir, exist_ok=True)

    for i, chunk in enumerate(chunks):
        output_path = os.path.join(output_dir, f"chunk_{i}.txt")
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(chunk.page_content)

    # テキストをAzure AI Searchにインデックスする
    #index_docs(chunks)
if __name__ == '__main__':
    main()