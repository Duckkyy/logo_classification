📘 不良品識別モデル – README
📌 概要

このリポジトリは，製品画像（good / bad）を入力として
良品・不良品を自動判定する二値分類モデル を実装したプロジェクトです。

クラス不均衡への対策

複数モデル（ResNet18 / EfficientNet / MobileNet / ViT）の比較

推論スクリプト（inference）

可視化とレポート作成用コード

を含んでいます。

学習済みモデルの重みファイル（weights）は容量が大きいため，
Google Drive で共有しています。

💻 動作環境（例）

以下の環境で動作確認を行いました：

OS：Ubuntu / macOS

Python：3.10

GPU：NVIDIA RTX 系（任意）

ライブラリ：PyTorch, torchvision, timm, scikit-learn など

🛠️ 1. Conda 環境構築

まずは Conda 環境を作成します：

conda env create -f environment.yml
conda activate defect-detection


environment.yml は本リポジトリに含まれています。

📦 2. 学習済みモデル（weights）のダウンロード

プロジェクト直下に weights/ フォルダを作成：

mkdir weights


以下の Google Drive リンクからモデルをダウンロードし，
weights/ フォルダに配置してください：

👉 Google Drive（学習済みモデル）
（ここにあなたのリンクを貼ってください）

🚀 3. 推論の実行（Inference）

次のコマンドで画像の推論を行えます：

python inference.py --image <path_to_image> --model weights/resnet18.pth


例：

python inference.py --image sample.png --model weights/resnet18.pth


推論結果（good / bad）がコンソールに表示されます。

📂 ディレクトリ構成（例）
project/
 ├── weights/             # ← Google Drive からダウンロード
 ├── src/                 # モデル学習/評価コード
 ├── inference.py         # 推論スクリプト
 ├── environment.yml      # conda 環境定義
 └── README.md
