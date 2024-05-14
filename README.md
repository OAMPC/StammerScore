# StammerScore

## Description
StammerScore is a comprehensive Python toolkit designed to facilitate the end-to-end process of speech fluency analysis. This project encompasses the procurement and preprocessing of datasets, training various machine learning models, saving these models, and utilizing them to assess the fluency of audio inputs. It also includes educational tools to help participants understand stuttering and evaluate fluency in audio samples. Additional utilities for data processing and visualization are provided to support iterative data exploration and model refinement.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Features](#features)
- [Dependencies](#dependencies)
- [Contributors](#contributors)
- [License](#license)

## Installation
To install StammerScore, follow these steps (please include any necessary commands or scripts):

1. Clone the repository:
git clone https://github.com/OAMPC/StammerScore

2. Install required Python packages:
pip install -r requirements.txt

## Usage
StammerScore does not have a linear workflow; instead, it offers various scripts and notebooks to perform specific tasks:
#### **Dataset Creation**: Scripts to download and extract audio clips, and Jupyter notebooks for audio augmentation and preprocessing.
- `Data/download_audio_updated.py`
- `Data/extract_clips_updated.py`
- `AudioAugmenter.ipynb`
#### **Model Training**: Notebooks and scripts for training machine learning models.
- `BinaryClassification.ipynb`
- `model_trainer.py`
- `MultiLabelClassification.ipynb`
- `feature_extractor.py`
#### **Model Evaluation**: Tools and scripts for evaluating models and educating users.
- `Evaluation/Tutorial/Tutorial.py`
- `Evaluation/DysfluencyMarkerApp.py`
- `Evaluation/KappaValueCalculator.ipynb`
- `Evaluation/CSVProcessor.ipynb`
- `generate_fluency_score.py`
- `Demo UI.py`

## Features
- **Dataset Management**: Tools for downloading, preprocessing, and augmenting speech datasets.
- **Model Training**: Flexible scripts for training different types of ML models on speech data.
- **Fluency Scoring**: Functionality to assess audio files and generate fluency scores.
- **Educational Tools**: Components to teach users about stuttering and evaluate speech fluency.

## Dependencies
StammerScore requires the following Python libraries:
- pandas
- numpy
- matplotlib
- joblib
- sklearn
- seaborn
- tqdm
- pathlib
- lightgbm
- scipy
- subprocess
- soundfile
- librosa
- shutil
- csv
- os
- argparse
- tkinter
- threading
- statsmodels
- krippendorff
- mutagen
- wave
- pygame
- time
- re

## Contributors
The original dataset and dataset download code are from the SEP-28k dataset, which can be found here: [SEP-28k on GitHub](https://github.com/apple/ml-stuttering-events-dataset).


## License
Specify the license under which the project is released. If the project includes third-party datasets or code, ensure their licenses are respected and documented.
