# オプション

## train_decoder.py
| オプション名 | 略 | 備考 |
|---| --- | ---|
|`--device`| `-d` | 学習を実行するデバイス |
|`--epoch`|  `-e` | エポック数 |
|`--batch`| `-b`| バッチサイズ。デフォルトが`4`なので、メモリが足りない場合は減らす。 |
|`--learning-rate`| `-lr` | ラーニングレート。通常は変更しない。 |
|`--length`|`-len` | データの長さ。デフォルトは `65536`。通常は変更しない。 |
|`--max-data`| `-m` | 最大ファイル数 |
|  |`-fp16 True`| 16ビット浮動小数点数を利用するかどうか |

## realtime_inference.py
| オプション名 | 略 | 備考 |
|---| --- | ---|
|`--target`|`-t`| 変換先となる音声が録音された音声ファイル |
|`-device`| `-d` | 推論を実行するデバイス |
|`--output`| `-o` | 出力オーディオデバイスのID |
|`--input`| `-i` | 入力オーディオデバイスのID |
|`--loopback`| `-l` | ループバック(２つ目の出力)のオーディオデバイスのID |
|`--gain`| `-g` | 出力のゲイン(dB) |
|`--threshold`| `-thr` | 変換を行う閾値(dB) |
|`--chunk`| `-c` | チャンクサイズ デフォルトは`3072` |
|`--buffersize`,| `-b` | バッファ数 デフォルトは`8`|
|| `-fp16 True`| 16ビット浮動小数点数を利用するかどうか |
|`-f0-rate` | `-f0`| 周波数の倍率。 |
|`--alpha`| `-a` | どれくらい元の音声を混ぜるかの割合。 デフォルトは `0`。 |
||`-k`| kNN回帰のKの数。デフォルトは`4`。 |

## inference.py
| Option name | Alias | Description |
|---| --- | ---|
|`--target`|`-t`| target speaker's speaking file |
|`-device`| `-d` | set inferencing device. you can use `cpu`, `cuda` or `mps` |
|| `-fp16 True`| use 16-bit floatation point (deprecated)|
|`-f0-rate` | `-f0`| rate of F0 pitch |
|`--alpha`| `-a` | bypass level. default is `0`. |
||`-k`| k of kNN regression. |


## Inferencing parameters

### F0 Rate
F0 rate is multiplier of F0-pitch.

set `0.5` to convert female voice to male voice.

set `2.0` to convert male voice to female voice.

### k
K nearest-neighbor regression's K.

default is `4`.

voice will become smoother if increase this.

### alpha
Parameter for how much of the original audio is transmitted.