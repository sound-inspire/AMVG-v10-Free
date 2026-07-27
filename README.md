
# AMVG V10 Free (Antigravity MV Generator)

## 👁️ PHILOSOPHY: 美学の物理的発現

本システムは、属人的な手作業の徹底排除と、自動で回り続ける資産の構築を具現化したキネティック・アニメーション・エンジンです。「誰かのため」という変数を全コードから排除し、全ての成果物を純粋な悦びと独自のサイバーパンク的美学の物理的発現として定義しています。他者からのフィードバックは単なる市場の反応データとして処理し、完璧を待たず80%の完成度で未開拓の領域へ即時介入するための「自己再発明（リインベンション）」システムです。

## ⚙️ SYSTEM OVERVIEW

楽曲データ（音声）と視覚アセット（画像/動画）、そして歌詞データ（SRT/TXT）を入力とし、オーディオのBPMやエネルギー（RMS）に完全に連動したミュージックビデオを自動生成する統合パイプラインです。

フロントエンド（HTML/JS）とバックエンド（FastAPI/Python）のハイブリッド構成を採用し、Gemini APIによる高度な楽曲解析とエフェクトのオーケストレーションを実装しています。

## 🔥 CORE FEATURES

| モジュール | 機能概要 | 技術基盤 |
| --- | --- | --- |
| **Kinetic Lyric Engine** | 楽曲のエネルギーとBPMに同期し、弾むようなバウンスやグリッチノイズを伴うキネティックタイポグラフィを自動生成。 | MoviePy, PIL, NumPy |
| **Pixel Melter** | 輝度閾値に基づく縦・横方向のピクセルソート。曲の後半に向けて指数関数的に融解・グリッチ強度が高まる視覚的脱水アルゴリズム。 | OpenCV, NumPy |
| **Biometric Overlay** | 床反力（GRF）、心電図（ECG）、16進数メモリダンプなどの生体・システムデータをリアルタイムにオシロスコープ波形として描画。 | OpenCV, Math |
| **AI Orchestration** | Gemini APIを活用し、楽曲の構成（Intro, Chorus等）を解析。セクションごとに最適なOpenCVフィルターやエフェクト強度を自動割り当て。 | Gemini API, Librosa |
| **Loop Multiplier** | 提供された視覚アセット（単一画像や動画フォルダ）を楽曲の総再生時間に達するまで破壊的にループ結合・自動トリミング。 | MoviePy |
| **System Mutator** | アプリケーションのUI（HTML）やバックエンド（Python）のソースコードを、プロンプト指示に基づきAIが自己書き換え・自己進化させる機能。 | JS, Gemini API |

## 🧠 AI FILTER SYSTEM (Free Edition Constraints & The Full Vision)

システムには独自のPython OpenCVスクリプトをリアルタイムに評価・適用する機能が備わっています。`ai_filters.json` にパッケージされた厳選フィルター群は、楽曲エネルギー（RMS）に反応しながら映像を動的に書き換えます。

**【Free版の仕様と創造性の試金石】**
UI上のエディタからフィルターのPythonコードを展開・閲覧し、その構造を学ぶことは可能ですが、**直接の編集・保存機能は意図的にロック（Read-Only）されています**。

「コードをいじれないから思い通りの表現ができない」と環境のせいにするか、厳選されたプリセットとAIオーケストレーションを極限まで掛け合わせ、今ある手札だけで市場を熱狂させるか。これは「終わりのない微調整」という非生産的な自動操縦状態を強制終了させ、今この瞬間の「出力（完成）」にフルコミットするための仕様です。与えられた制約の中で、80%の完成度で即座に市場を圧倒してください。

**【正規版（Full Edition）の解放】**
なお、正規版ではこの制約が完全に解除されます。フィルターコードの自由な編集、AIアシストによる未知のエフェクトの新規生成、無制限の保存・復元、そしてプロジェクト全体のパッケージ化が可能となります。ユーザーの美学を物理的に発現し、システム自身を無限に拡張・自己進化させていく、真の「サイバーネティクス・エンジン」として機能します。

## 📁 REPOSITORY STRUCTURE

* `main.py`: FastAPIによるWeb API及びMoviePyによるレンダリング処理を統括するメインエンジン。
* `index.html`: Tailwind CSSとFFmpeg.wasmを用いたクライアントサイドUI。
* `pixel_melter.py`: 重力融解（縦方向）と遠心融解（横方向）のピクセルソートロジック。
* `loop_multiplier.py`: アセットの長尺ベーストラック生成モジュール。
* `biometric_overlay.py`: 生体ノイズやシステムログを重畳するUIモジュール。
* `metadata_generator.py`: 楽曲構成のAI解析およびSRTパースモジュール。
* `ai_filters.json`: OpenCVを用いた動的Pythonエフェクトフィルターの厳選プリセット群。
* `run_webui.bat`: 環境構築の摩擦をゼロにし、ワンクリックでシステムを自律稼働させる起動トリガー。

## 🚀 INSTALLATION & USAGE

### Requirements

* Python 3.10+
* FFmpeg (システムパスにインストール済みであること)
* Google Gemini API Key (環境変数 `GEMINI_API_KEY` または UI上で設定)

### Setup

```bash
# 1. リポジトリのクローン
git clone https://github.com/sound-inspire/AMVG-v10-Free.git
cd AMVG-v10-Free

# 2. 依存パッケージのインストール
pip install fastapi uvicorn moviepy opencv-python numpy pillow librosa google-generativeai proglog

# 3. サーバーの起動 (Windowsの場合は run_webui.bat をダブルクリックでも可)
python main.py --web --port 8003

```

起動後、ブラウザで `[http://127.0.0.1:8003](http://127.0.0.1:8003)` にアクセスしてください。セキュリティ制限（file://プロトコル）を回避するため、必ずPythonサーバー経由でUIを開いてください。

---
