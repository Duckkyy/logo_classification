# 不良品識別モデル

本プロジェクトは、製品画像（good / bad）を入力として
良品・不良品を自動分類するための Python ベースのツールです。
クラス不均衡への対応や複数モデルの比較（ResNet / EfficientNet / MobileNet / ViT）を通じて、
より高精度な自動外観検査モデルの構築を目的としています。


## 1. 環境構築（Conda）

本プロジェクトは Conda 環境で動作します。

```bash
conda env create -f environment.yml
conda activate logo
```

## 2. 学習済みモデル（weights）の準備
1. weights/ フォルダを作成する：
```bash
mkdir weights
```
2. Google Drive から学習済みモデルを[ここ](https://drive.google.com/drive/folders/1_o6ZLPLGuaEySYNjg8hAiJZC3uYWhS9s?usp=sharing
)からダウンロードして配置する：

## 3. 推論実行方法（Inference）
画像を入力として良品/不良品の分類を行うには、以下のコマンドを実行します。

```bash
python inference.py --model-path weights/best_sampler_resnet.pth path/to/image --device cuda --model-type 1
```
他の学習済みモデルを使用したい場合は、--model に指定する重みファイル名 と --model_type の番号を下表のように変更するだけで推論を実行できます。
| weight file  | model_type |
| ------------ | ---------- |
| best_sampler_resnet.pth       | 1          |
| best_sampler_efficientnet.pth | 2          |
| best_sampler_mobilenet.pth    | 3          |
| best_sampler_vit.pth     | 4          |



