# ベースイメージとして、PyTorch 2.3.1 + CUDA 12.1 がプリインストールされた公式イメージを使用
# これにより、PyTorchとCUDAのバージョン不一致の問題を根本的に解決します
FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-devel

# 作業ディレクトリを設定
WORKDIR /app

# 必要なライブラリ（Pythonのバージョンなど）を更新
COPY requirements.txt .

# requirements.txt に基づいてライブラリをインストール
RUN pip install --no-cache-dir -r requirements.txt

# Google Colab接続用の拡張機能をインストールし、有効化する
RUN pip install --no-cache-dir jupyter_http_over_ws && \
    jupyter serverextension enable --py jupyter_http_over_ws

# プロジェクトのコードをコンテナにコピー
COPY . .

# 8888番ポートを開放
EXPOSE 8888

# コンテナ起動時にJupyter Labサーバーを起動
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--NotebookApp.token=''", "--NotebookApp.allow_origin='https-colab.research.google.com'"]