# UaVirBASE Baseline Reproduction Report

Date: 2026-06-03  
Workspace: `D:\UaVirBASE_reproduction`  
Official repo: `https://gitlab.com/g.jekaterynczuk/uavirbase_ssl`  
Repo commit: `cd3b21ce83d810b269a2038351b2fe2afbf483db`  
Dataset: `https://zenodo.org/records/15391924`

## Executive Summary

The UaVirBASE baseline pipeline was reproduced end to end: dataset download, checksum verification, extraction, official preprocessing options 3 and 4, clean environment setup, training, checkpoint generation, and final checkpoint evaluation.

No pretrained weights were present in the official GitLab repository, so immediate official-weight inference was not possible. A 10-epoch smallest practical training run was executed on the official active configuration. The final checkpoint achieved strong distance/height/azimuth performance but weaker UAV orientation prediction.

Final checkpoint metrics on 1,340 test clips:

| Target | MAE | RMSE |
|---|---:|---:|
| Distance `a` | 1.073 m | 1.541 m |
| Height `b` | 1.273 m | 1.729 m |
| Azimuth `c` | 3.457 deg | 5.899 deg |
| Orientation `d` | 25.187 deg | 41.556 deg |

## Commands Executed

### Workspace

```powershell
New-Item -ItemType Directory -Force -Path D:\UaVirBASE_reproduction,D:\UaVirBASE_reproduction\dataset,D:\UaVirBASE_reproduction\logs,D:\UaVirBASE_reproduction\artifacts
```

### Repository

```powershell
git clone https://gitlab.com/g.jekaterynczuk/uavirbase_ssl.git D:\UaVirBASE_reproduction\uavirbase_ssl
git rev-parse HEAD
```

Result: `cd3b21ce83d810b269a2038351b2fe2afbf483db`.

### Environment

```powershell
python -m venv D:\UaVirBASE_reproduction\.venv
D:\UaVirBASE_reproduction\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Error: PyPI could not resolve `torch==2.7.0+cu126`.

Fix:

```powershell
D:\UaVirBASE_reproduction\.venv\Scripts\python.exe -m pip install --upgrade pip
D:\UaVirBASE_reproduction\.venv\Scripts\python.exe -m pip install torch==2.7.0+cu126 torchaudio==2.7.0+cu126 --index-url https://download.pytorch.org/whl/cu126
D:\UaVirBASE_reproduction\.venv\Scripts\python.exe -m pip install filelock==3.13.1 fsspec==2024.6.1 Jinja2==3.1.4 MarkupSafe==2.1.5 mpmath==1.3.0 networkx==3.3 numpy==2.1.2 pillow==11.0.0 sympy==1.13.3 typing_extensions==4.12.2 wandb==0.19.11 albumentations librosa matplotlib pydub tqdm soundfile scipy
```

Smoke test:

```powershell
D:\UaVirBASE_reproduction\.venv\Scripts\python.exe -c "import torch, torchaudio, librosa, albumentations, pydub; import model, operation, trainer; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"
```

Result: `torch 2.7.0+cu126 cuda True NVIDIA GeForce RTX 4070`.

### Dataset

```powershell
curl.exe -L --retry 5 --retry-delay 10 --continue-at - --output D:\UaVirBASE_reproduction\dataset\Microphone_array.zip "https://zenodo.org/records/15391924/files/Microphone_array.zip?download=1"
Get-FileHash D:\UaVirBASE_reproduction\dataset\Microphone_array.zip -Algorithm MD5
tar -xf D:\UaVirBASE_reproduction\dataset\Microphone_array.zip -C D:\UaVirBASE_reproduction\dataset
```

Zenodo file: `Microphone_array.zip`, 13,783,664,678 bytes.  
MD5 matched: `3C4081511630F8045FDA0347CAC45534`.

### Official Preprocessing

`postprocess.py` was path-edited per the README instruction:

- `base_folder_path = r"D:\UaVirBASE_reproduction\dataset\Microphone_array"`
- `processed_path = r"D:\UaVirBASE_reproduction\processed_datasets"`
- `divided_path = r"D:\UaVirBASE_reproduction\divided_datasets"`

Commands:

```powershell
cmd.exe /c "echo 3| D:\UaVirBASE_reproduction\.venv\Scripts\python.exe postprocess.py"
cmd.exe /c "echo 4| D:\UaVirBASE_reproduction\.venv\Scripts\python.exe postprocess.py"
```

Option 3 splits each raw `output.wav` into 75% train and 25% test long WAVs and names folders from labels.  
Option 4 slices train audio into 1.5 s clips with 0.5 s stride and samples up to 20 one-second test clips per folder.

### Training

No official pretrained weights were present. Training was required.

`config.py` was set for the smallest meaningful reproducible run:

- `dataset_path = r"D:\UaVirBASE_reproduction\divided_datasets"`
- `epochs = 10`
- `batch_size = 16`
- active variation: `mel`, 96 kHz, `n_fft=2048`, `hop_length=1024`, `n_mels=128`
- WandB disabled.

Command:

```powershell
D:\UaVirBASE_reproduction\.venv\Scripts\python.exe main.py
```

Captured logs:

- `D:\UaVirBASE_reproduction\logs\training_10epochs_stdout.log`
- `D:\UaVirBASE_reproduction\logs\training_10epochs_stderr.log`

Generated checkpoints:

- `D:\UaVirBASE_reproduction\uavirbase_ssl\saved_models\mel_96000_2048_1024_0_128_0_0\5_end_unet.pth`
- `D:\UaVirBASE_reproduction\uavirbase_ssl\saved_models\mel_96000_2048_1024_0_128_0_0\10_end_unet.pth`

### Evaluation

The official repo has no standalone inference script, so `scripts\evaluate_checkpoint.py` was added. It loads the checkpoint config, rebuilds the official Dataset and model, computes test metrics, and saves figures/tables.

```powershell
D:\UaVirBASE_reproduction\.venv\Scripts\python.exe scripts\evaluate_checkpoint.py --checkpoint D:\UaVirBASE_reproduction\uavirbase_ssl\saved_models\mel_96000_2048_1024_0_128_0_0\10_end_unet.pth --out-dir D:\UaVirBASE_reproduction\artifacts --training-log D:\UaVirBASE_reproduction\logs\training_10epochs_stdout.log --batch-size 16 --examples 24
```

## A. Repository Analysis

Top-level official files:

| File | Purpose |
|---|---|
| `README.md` | Minimal usage instructions: download dataset, edit `postprocess.py`, run options 3 and 4, then run `main.py`. |
| `requirements.txt` | Partial dependency list. Fails for PyTorch CUDA wheel without PyTorch index and omits several imported packages. |
| `config.py` | Dataset path, feature extraction settings, model/training hyperparameters, and active feature variation list. |
| `postprocess.py` | Interactive preprocessing and metadata summary script. |
| `operation.py` | PyTorch Dataset, audio loading, feature extraction, normalization, resizing, and spectrogram plotting helper. |
| `model.py` | Residual CNN classifier architecture. |
| `trainer.py` | Training loop, test loop, loss, metric computation, checkpoint saving, and optional WandB logging. |
| `main.py` | Training entry point. Builds Dataset/DataLoader, Trainer, and iterates over active config variations. |

Identified scripts:

- Training: `main.py`, `trainer.py`
- Inference: none official; evaluation logic exists inside `Trainer.test_loop`
- Preprocessing: `postprocess.py`
- Configuration: `config.py`
- Model architecture: `model.py`

Architecture:

![Model architecture](figures/model_architecture.png)

The model is a residual convolutional classifier:

1. Input tensor: `[batch, 8, 256, 256]`.
2. Encoder `conv_in`.
3. Six residual downsampling blocks with channel multiplier blocks `(4, 8, 16, 24, 32, 64)` and base channel count `8`.
4. Final residual block.
5. Global average pooling.
6. Fully connected layer, GELU, fully connected output layer.
7. Six output values.

Input features:

- 8-channel waveform loaded with `librosa.load(..., mono=False)`.
- Official active run uses mel spectrograms:
  - sample rate: 96,000 Hz
  - `n_fft=2048`
  - `hop_length=1024`
  - `n_mels=128`
- Per-channel spectrograms are normalized and resized to 256 x 256, then stacked as 8 input channels.

Output labels:

The network predicts six regression outputs:

- `a`: distance normalized by `/50`
- `b`: height normalized by `/50`
- `c`: azimuth encoded as normalized sine and cosine
- `d`: UAV orientation encoded as normalized sine and cosine

Orientation mapping:

- folder code `1`: front, 0 deg
- folder code `2`: left, 270 deg
- folder code `3`: back, 180 deg
- folder code `4`: right, 90 deg

Loss:

`trainer.py` uses summed MSE over six sigmoid-normalized outputs:

```text
MSE(distance) + MSE(height) + MSE(azimuth_sin) + MSE(azimuth_cos) + MSE(orientation_sin) + MSE(orientation_cos)
```

Training configuration used:

- Optimizer: Adam
- learning rate: `0.0002`
- betas: `(0.5, 0.9)`
- epochs: `10`
- batch size: `16`
- checkpoint interval: every 5 epochs
- GPU: NVIDIA GeForce RTX 4070

## B. Dataset Analysis

Downloaded dataset:

- Archive: `Microphone_array.zip`
- Size: 13,783,664,678 bytes
- MD5: `3C4081511630F8045FDA0347CAC45534`
- Extracted raw folders: 132
- Raw `output.wav` files: 132
- Raw `label.json` files: 132
- Raw duration: 5,536.50 s, or 1.538 h

Audio format from example file:

- Format: WAV
- Subtype: PCM_32
- Channels: 8
- Sampling rate: 96,000 Hz
- Example duration: 60 s

Microphone array:

- 8 microphones.
- Example label metadata lists a Behringer UMC1820 interface and 8 Rode NTG-2 channels.
- Microphones are arranged around the center with azimuths 0, 45, 90, 135, 180, 225, 270, and 315 degrees.
- Example microphone distance from array center: 1.72 m.

Label definitions:

- `distance`: UAV distance class, mainly 10 m or 20 m.
- `height`: UAV height class, mainly 10 m or 20 m.
- `azimuth`: source bearing class, `0, 45, 90, 135, 180, 225, 270, 315`.
- `rotation`: UAV orientation relative to array, encoded into side code `1-4`.
- Ambient noise examples are encoded as `none_none_none_none`.

Official divided dataset:

| Split | Folders | WAV clips |
|---|---:|---:|
| Train | 132 | 8,040 |
| Test | 132 | 1,340 |

Example files:

- `D:\UaVirBASE_reproduction\divided_datasets\train\10_10_0_1\clip_0_1500.wav`
- `D:\UaVirBASE_reproduction\divided_datasets\test\10_10_0_1\random_clip_1.wav`
- `D:\UaVirBASE_reproduction\dataset\Microphone_array\20241115_093611\label.json`

## C. Reproduction Results

Training curves:

![Training curves](figures/training_curves.png)

Validation metrics printed during training:

| Epoch | Loss | Distance MAE | Height MAE | Azimuth MAE | Orientation MAE |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.6543 | 11.9742 | 10.6252 | 94.2723 | 91.8700 |
| 2 | 0.2620 | 2.8046 | 3.1828 | 9.0944 | 81.8995 |
| 3 | 0.2551 | 1.7215 | 2.7991 | 5.8820 | 74.7728 |
| 4 | 0.1960 | 1.7435 | 2.3720 | 5.3467 | 54.7905 |
| 5 | 0.1645 | 1.5554 | 1.7693 | 5.5887 | 46.1463 |
| 6 | 0.1429 | 1.5464 | 2.0000 | 5.4987 | 38.9675 |
| 7 | 0.1140 | 1.4164 | 1.8327 | 7.3708 | 32.5317 |
| 8 | 0.1101 | 1.3674 | 1.5673 | 4.1911 | 31.4684 |
| 9 | 0.1229 | 1.4569 | 2.2640 | 4.3868 | 33.5391 |
| 10 | 0.0871 | 1.2083 | 1.4128 | 4.7157 | 25.5358 |

Final checkpoint evaluation:

| Target | MAE | RMSE |
|---|---:|---:|
| Distance | 1.073 m | 1.541 m |
| Height | 1.273 m | 1.729 m |
| Azimuth | 3.457 deg | 5.899 deg |
| Orientation | 25.187 deg | 41.556 deg |

Prediction examples:

![Prediction examples](figures/prediction_examples.png)

Example rows are saved in:

- `tables\prediction_examples.csv`
- `tables\final_checkpoint_metrics.json`
- `tables\training_metrics.csv`

## D. Practical Understanding

### 1. How the microphone array estimates UAV direction

The array records the same UAV sound at 8 microphones placed at known positions. A UAV sound reaches microphones at slightly different times and with different spectral/level patterns. The neural network learns those multi-channel differences from examples. It is not explicitly solving time-difference-of-arrival equations; it learns a mapping from 8-channel spectrogram patterns to source labels.

### 2. How azimuth is predicted

Azimuth is encoded as sine and cosine rather than a direct degree value. This avoids the 0/360 degree discontinuity. The model predicts normalized sine and cosine values, then the angle is reconstructed with `atan2`.

### 3. How distance is predicted

Distance is a scalar regression target. The label distance is divided by 50 during training and multiplied by 50 during evaluation. The model learns distance cues from amplitude, spectrum, reverberation, and multi-channel patterns, but in this dataset the primary distances are discrete 10 m and 20 m classes.

### 4. What position is predicted

The model predicts:

- direction/bearing: yes, azimuth
- distance: yes
- height: yes
- UAV orientation: yes
- full 3D Cartesian position: not directly

It predicts polar-style labels relative to the microphone array: distance, height, azimuth, and orientation. A full 3D relative position can be derived approximately from distance, height, and azimuth if coordinate conventions are defined, but that conversion is not implemented in the official repo.

### 5. Adaptability

Crazyflie:

- Possible, but not plug-and-play.
- Crazyflie has different propeller size, motor speed, acoustic spectrum, and quieter sound than a DJI Mavic 3 Cine.
- Fine-tuning or recollection is needed.

ReSpeaker microphone arrays:

- Possible in concept, but the model currently assumes exactly 8 channels.
- A 4-channel ReSpeaker would require architecture input-channel change and retraining.
- An 8-channel ReSpeaker could fit the input shape but array geometry differs, so retraining or calibration is still needed.

USB microphone arrays:

- Possible if synchronized multi-channel audio is available.
- Independent unsynchronized USB microphones are problematic because timing differences are the key localization cue.

Vicon-assisted experiments:

- Very suitable for label generation.
- Vicon can provide ground-truth relative pose while microphones provide sound.
- For robust experiments, synchronize audio timestamps with Vicon timestamps.

## E. Research Assessment

Replacing the dataset UAV with Crazyflie is moderately to highly difficult. The pipeline is usable, but the learned acoustic distribution is DJI Mavic 3 Cine, not Crazyflie. Crazyflie acoustic signatures are smaller, higher pitched, lower amplitude, and more sensitive to room acoustics and background noise.

Current model generalization to Crazyflie propeller noise is unlikely to be reliable without fine-tuning. The model may still learn generic direction cues if signal-to-noise ratio is high, but the distance and orientation labels are especially dataset-specific.

Additional data needed:

- 8-channel synchronized Crazyflie recordings.
- Multiple distances, heights, azimuths, and yaw/orientation states.
- Hover, translation, and formation-flight conditions.
- Background-only/noise recordings.
- Different rooms or outdoor conditions.
- Ground-truth positions from Vicon or motion capture.
- Calibration data for the target microphone array geometry.

Support for "Acoustic Relative Bearing Estimation for Crazyflie Swarms":

- Feasible as a research direction.
- The current UaVirBASE model supports the bearing-estimation framing, but a Crazyflie-specific dataset is required.
- For swarms, source separation or multi-source localization becomes the central challenge; the current baseline is single-source.

Support for "Acoustic-Assisted Formation Recovery under Vicon Degradation":

- Feasible as an assistive cue, not as a full replacement initially.
- Acoustic bearing can help maintain relative direction when Vicon degrades.
- Distance from acoustics alone is less robust and will need fusion with inertial/control priors.
- A practical system should fuse audio bearing with degraded Vicon, onboard state estimates, and formation constraints.

## Errors and Fixes

1. `pip install -r requirements.txt` failed because PyPI did not provide `torch==2.7.0+cu126`.
   - Fix: install PyTorch/torchaudio from `https://download.pytorch.org/whl/cu126`.

2. Initial PyTorch install failed with old pip due to `typing_extensions` metadata handling.
   - Fix: upgrade pip to `26.1.2`.

3. Official requirements omitted imported packages.
   - Fix: install `albumentations`, `librosa`, `matplotlib`, `pydub`, `tqdm`, `soundfile`, and `scipy`.

4. `postprocess.py` had hard-coded paths.
   - Fix: edit paths as the official README instructs.

5. Initial 10-epoch launch with JSON override failed because PowerShell/Start-Process quoting passed malformed JSON.
   - Fix: set the small-run parameters in `config.py`.

6. First custom evaluation run used base `config.py` feature settings instead of checkpoint settings.
   - Fix: evaluator now loads `checkpoint["config"]` before building the Dataset.

## Reproducibility Notes and Limitations

- Official option 4 uses `random.sample` without a fixed seed for test clip selection. Exact test clips can change if preprocessing is rerun.
- The repository does not include pretrained weights, so published-paper metrics cannot be checked directly without full training.
- The official `main.py` tests at the beginning of each epoch, then trains, then saves checkpoints every 5 epochs. The final checkpoint therefore needs a separate evaluation pass.
- `postprocess.py` option 2 has confusing metadata parsing names and is not used for training.
- Ambient/noise labels are encoded as zeros for all six targets, which maps to an artificial angle under sine/cosine decoding. This can affect angle metrics if included.
- The model predicts single-source labels and is not directly a multi-UAV swarm localizer.

## Deliverables

- README: `D:\UaVirBASE_reproduction\README.md`
- Report: `D:\UaVirBASE_reproduction\artifacts\reproduction_report.md`
- Environment: `D:\UaVirBASE_reproduction\artifacts\requirements_reproduction.txt`
- Generated figures list: `D:\UaVirBASE_reproduction\artifacts\generated_figures.txt`
- Generated models list: `D:\UaVirBASE_reproduction\artifacts\generated_models.txt`
- Logs:
  - `D:\UaVirBASE_reproduction\logs\training_10epochs_stdout.log`
  - `D:\UaVirBASE_reproduction\logs\training_10epochs_stderr.log`
- Figures:
  - `figures\model_architecture.png`
  - `figures\training_curves.png`
  - `figures\prediction_examples.png`
- Models:
  - `saved_models\mel_96000_2048_1024_0_128_0_0\5_end_unet.pth`
  - `saved_models\mel_96000_2048_1024_0_128_0_0\10_end_unet.pth`

## Strengths and Limitations

Strengths:

- Public dataset with 8-channel synchronized recordings and rich metadata.
- Simple end-to-end baseline.
- Sine/cosine angle encoding is appropriate.
- The reproduced 10-epoch run learned azimuth well.

Limitations:

- No pretrained weights in the repo.
- Dependency file is incomplete.
- Paths are hard-coded.
- Preprocessing has non-deterministic test clip sampling.
- Orientation remains much harder than azimuth.
- The model is tied to an 8-channel geometry and a DJI Mavic-style acoustic source.
