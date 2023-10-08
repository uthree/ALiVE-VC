# ZVC
軽量なベクトル検索ベースのAIボイスチェンジャ

## 必要なもの
- Python 3.10 or later
- PyTorch 2.0 or later
- PyTorch GPU 環境

## 使い方 (フルスクラッチで訓練する場合)
1. このリポジトリをクローン
```sh
git clone https://github.com/uthree/zvc
cd zvc
```

2. PyTorchを[ここ](https://pytorch.org)からインストール

3. 必要なライブラリをインストール
```sh
pip install -r requirements.txt
```

4. JVScorpusを [ここ](https://sites.google.com/site/shinnosuketakamichi/research-topics/jvs_corpus) からダウンロードし解凍

5. Contente Encoderを訓練する
```sh
python3 train_content_encoder.py path/to/jvs/corpus -d cuda
```

6. Pitch Estimatorを訓練する
```sh
python3 train_pitch_estimator.py path/to/jvs/corpus -d cuda
```

7. Decoderを訓練する
```sh
python3 train_decoder.py path/to/jvs/corpus -d cuda
```


## 参考文献
- [Vocos](https://arxiv.org/abs/2306.00814)
- [kNN-VC](https://arxiv.org/abs/2305.18975)
- [DDPM](https://arxiv.org/abs/2006.11239)
- [DDIM](https://arxiv.org/abs/2010.02502)