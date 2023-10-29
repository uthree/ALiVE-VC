# ZVC
A lightweight vector-search based voice conversion system  
(This repository is in the experimental stage. Contents are subject to change without notice)

[日本語](https://github.com/uthree/zvc/blob/main/documents/readme_ja.md)

## Architecture
![Arch.](https://github.com/uthree/zvc/blob/main/documents/architecture.png)

## Requirements
- Python 3.10 or later
- PyTorch 2.0 or later
- PyTorch GPU environment

## Usage (train full-scratch)
1. Clone this repository and move directory.
```sh
git clone https://github.com/uthree/zvc
cd zvc
```

2. Install  PyTorch from
[here](https://pytorch.org).

3. Install requirements.
```sh
pip install -r requirements.txt
```

4. Download JVS corpus from [here](https://sites.google.com/site/shinnosuketakamichi/research-topics/jvs_corpus) and unzip file.

5. Train content encoder
```sh
python3 train_content_encoder.py path/to/jvs/corpus -d cuda
```

6. Train pitch estimator
```sh
python3 train_pitch_estimator.py path/to/jvs/corpus -d cuda
```

7. Train decoder
```sh
python3 train_decoder.py path/to/jvs/corpus -d cuda
```



## References
- [Vocos](https://arxiv.org/abs/2306.00814)
- [kNN-VC](https://arxiv.org/abs/2305.18975)
