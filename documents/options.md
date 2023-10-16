# Options

## train_decoder.py
| Option name | Alias | Description |
|---| --- | ---|
|`--device`| `-d` | set training device. you can use `cpu`, `cuda` or `mps` |
|`--epoch`|  `-e` | number of epochs |
|`--batch`| `-b`| batch size. default is `4`, decrase this if not enough memory |
|`--learning-rate`| `-lr` | learning rate |
|`--length`|`-len` | data length. default is `65536` |
|`--max-data`| `-m` | max number of data file. |
|  |`-fp16 True`| use 16-bit floating point |

## realtime_inference.py
| Option name | Alias | Description |
|---| --- | ---|
|`--target`|`-t`| target speaker's speaking file |
|`-device`| `-d` | set inferencing device. you can use `cpu`, `cuda` or `mps` |
|`--output`| `-o` | output audio device ID |
|`--input`| `-i` | input audio device ID |
|`--loopback`| `-l` | loopback(second output) audio device ID |
|`--gain`| `-g` | output gain(dB) |
|`--threshold`| `-thr` | conversion threshold |
|`--chunk`| `-c` | chunk size. default is `4096` |
|`--buffersize`,| `-b` | buffer size. default is `8`|
|| `-fp16 True`| use 16-bit floatation point (deprecated)|
|`--pitch` | `-p`| pitch shift |
|`--alpha`| `-a` | bypass level. default is `0`. |
||`-k`| k of kNN regression. |

## inference.py
| Option name | Alias | Description |
|---| --- | ---|
|`--target`|`-t`| target speaker's speaking audio file (supports `wav`, `mp3`, `ogg`) |
|`--audio-library-path`| `-lib` | path to audio library |
|`-device`| `-d` | set inferencing device. you can use `cpu`, `cuda` or `mps` |
|| `-fp16 True`| use 16-bit floatation point (deprecated)|
|`--pitch`| `-p` | pitch shift |
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