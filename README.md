# RAG Evaluation

## 概要
このプロジェクトは、SoftwareDesign誌 2025年1月号から連載中の「RAGアプリケーション評価・改善の極意」におけるサンプルコード集です。
詳細に関しては本誌をご覧ください。

## 事前準備

本アプリケーションはAzure OpenAI Serviceを利用します。
事前にAzure上にAzure OpenAI Serviceのリソースを作成し、LLMとEmbeddingモデルをデプロイする必要があります。

## プロジェクト構成

```
root/ 
├─ .devcontainer/ # DevContainerに関するファイルが含まれています
├─ evaluation/ # 2025年3月号のSoftwareDesign誌で掲載している各Metricの評価実装コードが含まれています。
├─ langsmith/ # 2025年4月号のSoftwareDesign誌で掲載しているLangSmithを利用した評価サイクル実装コードが含まれています。
├─ source_docs/ # 2025年4月号のSoftwareDesign誌で掲載しているテストセット生成時に使用する元データが含まれています。
├─ rag_sample/ # 2025年5月号のSoftwareDesign誌で掲載しているRAGの実装が含まれています。
├─ rag_source_docs/ # 2025年5月号のSoftwareDesign誌で掲載しているRAGの実装で利用する就業規則が含まれています。
├─ .env.sample # 本アプリケーションで必要となる.envファイルのサンプルです。こちらを元に.envファイルを生成してください。 
├─ requirements.txt # 本アプリケーションで必要となるパッケージリストになります。 
└─ README.md
```

## 実行方法

1. 必要なパッケージのインストール
(DevContainerを利用している場合はコンテナビルド後に自動で行われます)

```
pip install -r requirements.txt
```

2. .envファイルの生成
.env.sampleをコピーし、.envファイルを生成してください。

```
AZURE_OPENAI_ENDPOINT={{Azure OpenAI Serviceのリソースのエンドポイント}}
AZURE_OPENAI_API_KEY={{Azure OpenAI ServiceのAPIキー}}
OPENAI_API_VERSION={{Azure OpenAI ServiceのAPIバージョン}} //本プロジェクトでは"2024-10-21"を想定しています。
LLM_DEPLYMENT_NAME={{Azure OpenAI Service上にデプロイしたLLMモデルの名称}}
EMBEDDING_DEPLYMENT_NAME={{Azure OpenAI Service上にデプロイしたEmbeddingモデルの名称}}
LANGCHAIN_TRACING_V2=true
LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
LANGCHAIN_API_KEY={{LangSmithのAPIキー}}
DATASET_NAME={{LangSmith上に構築するDataset名}}
```

### 各Metricの評価
evaluationディレクトリの配下に用意してある評価用のコードを実行します。
```
python evaluation/context_precision.py
```

### LangSmitghでの評価

1. Ragasでテストセットを構築し、LangSmith上へ登録する
```
python langsmith/register_testset.py
```

2. 登録したテストセットを利用して評価を行う
```
python langsmith/evaluate.py
```

### RAGのサンプルコードの実行

Azure AI SearchおよびAzure OpenAI Serviceをデプロイし、接続情報を.envファイルに追加する

1. インデクサーの実行
```
python indexer.py
```

2. オーケストレーターの実行
```
python orchestraotr.py
```

3. LangSmith上での評価の実行

```
python evaluate.py
```