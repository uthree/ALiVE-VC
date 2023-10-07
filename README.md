# ZVC
A lightweight vector-search based AI voice conversion system

## Usage
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

7. Train decoder


## References
- [Vocos](https://arxiv.org/abs/2306.00814)
- [kNN-VC](https://arxiv.org/abs/2305.18975)