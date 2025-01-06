# ポケモンしりとり AI

ポケモンの名前でしりとり対話を行う AI モデルです。Transformer ベースの言語モデルを使用して、ポケモンの名前を学習し、しりとりルールに従った応答を生成します。

<img width="150" src="https://github.com/user-attachments/assets/15878219-e357-4c26-98c7-07949522ef63" />

## 機能

- PokeAPI を使用した最新のポケモン名データの取得
- Transformer ベースの言語モデルによる対話生成
- しりとりルールに基づいた応答生成

## 必要要件

- Python 3.10 以上
- Poetry（依存関係管理）

## インストール

```bash
# リポジトリのクローン
git clone [repository-url]
cd poke-siritori

# 依存関係のインストール
poetry install
```

## Huggingface Space での使用

[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces)

このプロジェクトは Huggingface Space で公開可能です。以下の手順で使用できます：

1. モデルのダウンロード:

   - [Google Drive](https://drive.google.com/file/d/1AYTLqBspFKvIouA2OXXtNkePIRUyAYRa/view?usp=sharing)からモデルファイルをダウンロードし、プロジェクトルートに配置してください。

2. ローカルでの使用:

```bash
poetry run python app.py
```

## モデルの学習

新しいモデルを学習する場合は以下のコマンドを実行します：

```bash
poetry run python main.py
```

## プロジェクト構成

- `main.py`: モデルの学習用スクリプト
- `inference.py`: 学習済みモデルを使用した推論用スクリプト
- `debug.py`: デバッグ用スクリプト
- `pokemon_names.csv`: 正規化されたポケモン名データ
- `pokemon_names_raw.csv`: PokeAPI から取得した生のポケモン名データ
- `tokenizer.pkl`: トークナイザーのデータ
- `best_model.pt`: 学習済みモデル

## 技術詳細

- **モデルアーキテクチャ**: Transformer Decoder
- **トークナイズ**: カスタムトークナイザー（文字単位）
- **データセット**: PokeAPI から取得した 1010 種類のポケモン名
- **学習方法**: 自己回帰的な言語モデリング

## ライセンス

MIT

## 作者

tomoyaf <gazimum@gmail.com>
