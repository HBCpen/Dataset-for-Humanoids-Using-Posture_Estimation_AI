# Dataset for Humanoids Using Posture Estimation AI

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://python.org)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10+-green.svg)](https://mediapipe.dev)
[![License](https://img.shields.io/badge/License-TBD-gray.svg)](#license)

## プロジェクト概要

ヒューマノイドロボット向けの動作学習やシミュレーションに活用できるデータセットを、姿勢推定AI（2D/3D）を用いて画像・動画ファイルから自動生成するためのリポジトリです。データクレンジングからアノテーション整形、フォーマット統合までの一連のパイプラインを構築することを目的とします。

## クイックスタート

### 1. 環境構築

```bash
# リポジトリのクローン
cd /path/to/project

# 仮想環境の作成と有効化
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# 依存パッケージのインストール
pip install -r requirements.txt
```

### 2. パイプラインの実行

```bash
# 単一の動画を処理
python pipeline.py data/raw/sample_video.mp4

# 単一の画像を処理
python pipeline.py data/raw/sample_image.jpg

# ディレクトリ内の全ファイルを処理
python pipeline.py data/raw/ -o data/output/

# カスタム設定ファイルを使用
python pipeline.py video.mp4 -c configs/custom.yaml
```

### 3. Webデモの起動

```bash
cd src
python app.py
# ブラウザで http://localhost:5000 にアクセス
```

## ディレクトリ構成

```
├── README.md
├── requirements.txt          # Python依存パッケージ
├── pipeline.py               # メインパイプラインスクリプト
├── configs/
│   └── default.yaml          # デフォルト設定
├── data/
│   ├── raw/                  # 入力動画・画像
│   ├── interim/              # 前処理済みデータ
│   ├── processed/            # 姿勢推定結果（JSON）
│   └── export/               # エクスポート済みデータ
├── scripts/
│   ├── preprocess.py         # 動画前処理
│   ├── pose_estimation.py    # 姿勢推定（MediaPipe）
│   ├── postprocess.py        # 後処理（平滑化・正規化）
│   └── export.py             # データエクスポート
├── src/
│   ├── app.py                # Webデモサーバー
│   ├── templates/            # HTMLテンプレート
│   └── static/               # CSS/JavaScript
├── notebooks/                # Jupyterノートブック
└── tests/                    # テストコード
```

## パイプライン詳細

### Stage 1: 前処理 (`preprocess.py`)
- 動画の解像度正規化（最大幅1280px）
- フレームレート調整
- 品質フィルタリング

### Stage 2: 姿勢推定 (`pose_estimation.py`)
- MediaPipe Poseによる33点の骨格ランドマーク抽出
- 2D/3D座標の取得
- 信頼度スコアの記録
- 骨格オーバーレイ動画の生成

### Stage 3: 後処理 (`postprocess.py`)
- 時間軸での平滑化（ガウシアンフィルタ）
- 座標の正規化（hip-center原点）
- 欠損値の補間
- 関節角度の計算

### Stage 4: エクスポート (`export.py`)
- **JSON**: ヒューマノイド向けフォーマット
- **CSV**: 時系列データ分析用
- **NPZ**: NumPy高速読み込み用

## 主な目標
- 既存の画像・動画アーカイブから人間のポーズ情報を抽出し、骨格データとして保存する。
- ユースケース別（歩行、作業動作、ジェスチャーなど）に分類可能なラベル設計を行う。
- ロボットシミュレータや強化学習環境で取り扱いやすいフォーマット（JSON、CSV、ROS bag など）へ変換する。
- 再現性の高いパイプラインを構築し、データ更新時も自動的に処理できる自動化スクリプトを提供する。
- 高価なセンサーなどの用意無しに独自のデータセットを比較的簡単に作成することを目標とする。

## 想定技術スタック
- **姿勢推定**: MediaPipe, OpenPose, MoveNet, Detectron2 などのモデルと推論スクリプト
- **前処理/管理**: Python, PyTorch/TensorFlow, OpenCV, NumPy, Pandas
- **ワークフロー自動化**: Prefect, Airflow, Dagster などのワークフローエンジン（必要に応じて）
- **データ保存形式**: JSON（骨格データ）, CSV（メタ情報）, NPZ/Parquet（高速アクセス用）

## データフロー
1. **入力収集**: Raw の画像・動画を `data/raw/` に配置
2. **前処理**: 解像度統一、背景差分、品質フィルタを `scripts/preprocess.py` で実行
3. **姿勢推定**: 推論スクリプトで骨格キーポイントを抽出し `data/processed/` に保存
4. **アノテーション整形**: キーポイントをヒューマノイド座標系に正規化し、ラベルと紐付け
5. **出力生成**: フォーマット別に `data/export/` へ書き出し
6. **品質管理**: 可視化ダッシュボードや統計量で品質評価


## セットアップ手順
1. Python 仮想環境を作成: `python -m venv .venv && source .venv/bin/activate`
2. サンプルデータを `data/raw/` 配下に配置
3. ノートブックまたはスクリプトを実行し、姿勢推定モデルの動作を確認

## ロードマップ
- [x] ベースライン姿勢推定モデルの選定と推論スクリプト作成
- [x] データ前処理・後処理パイプラインの整備
- [ ] ラベルスキーマとメタデータ設計
- [ ] 品質評価指標と可視化ツールの実装
- [ ] CI/CD による自動テスト・バリデーション導入

## コントリビューション
スタイルガイドやテスト方針を整備するまでは、まず Issue でディスカッションしてください。

## ライセンス
利用予定のデータソースやモデルライセンスに合わせたライセンスを検討中。決定次第更新。
