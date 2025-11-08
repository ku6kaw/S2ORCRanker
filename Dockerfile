# ベースイメージ (変更なし)
FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-devel
# 作業ディレクトリを設定
WORKDIR /app

# 必要なライブラリをコピー
COPY requirements.txt .

# requirements.txt に基づいてライブラリをインストール
RUN pip install --no-cache-dir -r requirements.txt

# Google Colab接続用の拡張機能をインストールし、有効化する
RUN pip install --no-cache-dir jupyter_http_over_ws && \
    jupyter server extension enable --py jupyter_http_over_ws

# Colabからのクロスオリジン接続を許可する設定ファイルを作成する
RUN mkdir -p /root/.jupyter/
RUN echo "c.ServerApp.allow_origin = 'https-colab.research.google.com'" >> /root/.jupyter/jupyter_server_config.py
RUN echo "c.ServerApp.allow_credentials = True" >> /root/.jupyter/jupyter_server_config.py

# プロジェクトのコードをコンテナにコピー
COPY . .

# 8888番ポートを開放
EXPOSE 8888

# コンテナ起動時にJupyter Labサーバーを起動
# (commandはdocker-compose.ymlで指定するため、ここでのCMDはシンプルにする)
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]