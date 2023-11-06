# ALiVE-VC
![Logo](https://github.com/uthree/zvc/blob/main/documents/alive-vc-logo.png)

軽量なベクトル検索ベースのAIボイスチェンジャ  
(このリポジトリは実験段階です。内容は予告なく変更される場合があります。)

## アーキテクチャ
![Arch.](https://github.com/uthree/zvc/blob/main/documents/architecture.png)

## 必要なもの
- Python 3.10 or later
- PyTorch 2.0 or later
- PyTorch GPU 環境

## 使い方 (フルスクラッチで訓練する場合)
1. このリポジトリをクローン
```sh
git clone https://github.com/uthree/ALiVE-VC
cd ALiVE-VC
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

8. 特定の話者でファインチューニング

```sh
# まず、辞書となる音声ライブラリを生成
python3 generate_voice_library.py path/to/target/voices -lib voice_library.pt
# 学習を実行。
python3 fine_tune.py path/to/target/voices -d cuda
```

9. 推論  
`inputs`という名前のフォルダを個のリポジトリの直下に作成し、そこに変換したい音声ファイルを入れる。  
そのあと推論を実行。  
`-p <音階>` でピッチシフトを行える。
```sh
python3 inference.py -lib voice_library.py
```  
すると`outputs`という名前のフォルダが自動生成されるので、その中のファイルを再生し、結果を確認する。

## 参考文献
- [Vocos](https://arxiv.org/abs/2306.00814)
- [kNN-VC](https://arxiv.org/abs/2305.18975)
