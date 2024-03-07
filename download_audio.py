#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2021 Apple Inc. All Rights Reserved.
#

"""
For each podcast episode:
* Download the raw mp3/m4a file
* Convert it to a 16k mono wav file
* Remove the original file
"""

import os
import pathlib
from pydub import AudioSegment
import numpy as np
import argparse
import requests

parser = argparse.ArgumentParser(description='Download raw audio files for SEP-28k or FluencyBank and convert to 16k Hz mono wavs.')
parser.add_argument('--episodes', type=str, required=True,
                    help='Path to the labels csv files (e.g., SEP-28k_episodes.csv)')
parser.add_argument('--wavs', type=str, default="wavs",
                    help='Path where audio files are saved')

args = parser.parse_args()

# Ensure FFmpeg is accessible by pydub/AudioSegment
# AudioSegment.converter = "/path/to/ffmpeg"  # Uncomment and specify if ffmpeg is not in PATH

# Load episode data
table = np.loadtxt(args.episodes, dtype=str, delimiter=",")
n_items = len(table)

for i in range(n_items):
    show_abrev, ep_idx, episode_url = table[i, [-2, -1, 2]]
    episode_dir = pathlib.Path(args.wavs, show_abrev)
    episode_dir.mkdir(parents=True, exist_ok=True)

    audio_path_orig = episode_dir / f"{ep_idx}.wav"  # Directly working with .wav for simplicity

    if not audio_path_orig.exists():
        print("Downloading and processing", show_abrev, ep_idx)

        # Download audio file
        response = requests.get(episode_url)
        if response.status_code == 200:
            with open(audio_path_orig, 'wb') as audio_file:
                audio_file.write(response.content)

            # Convert audio to desired format and specs with pydub
            audio = AudioSegment.from_file_using_temporary_files(audio_path_orig)
            audio = audio.set_channels(1).set_frame_rate(16000)
            audio.export(audio_path_orig, format="wav")
        else:
            print(f"Failed to download {episode_url}")

print("Processing complete.")
