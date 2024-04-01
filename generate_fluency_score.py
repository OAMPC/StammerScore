import argparse
import os
import shutil
import numpy as np
import pandas as pd
import joblib
import librosa
import soundfile as sf
from tqdm import tqdm
import subprocess
import feature_extractor as fe


def split_audio_signal(y, sr, chunk_length=3):
    """Splits an audio signal into fixed-length chunks."""
    chunks = [(y[i:i + chunk_length * sr], i // (chunk_length * sr))
              for i in range(0, len(y), chunk_length * sr) if i + chunk_length * sr <= len(y)]
    return chunks

def save_chunks_and_predict(chunks, chunks_dir, sr, model, scaler):
    """Saves audio chunks to disk in 'fluent chunks' or 'dysfluent chunks' directories based on predictions."""
    fluent_dir = os.path.join(chunks_dir, "fluent chunks")
    dysfluent_dir = os.path.join(chunks_dir, "dysfluent chunks")

    # Create directories for fluent and dysfluent chunks
    os.makedirs(fluent_dir, exist_ok=True)
    os.makedirs(dysfluent_dir, exist_ok=True)

    predictions, chunk_names = [], []
    for chunk, index in tqdm(chunks, desc="Processing chunks"):
        features = fe.extract_features_advanced(chunk, sr)
        scaled_features = scaler.transform(np.array(features).reshape(1, -1))
        prediction = model.predict(scaled_features)[0]
        
        # Determine the target directory based on the prediction
        target_dir = fluent_dir if prediction == 1 else dysfluent_dir
        chunk_name = f"chunk_{index}.wav"
        chunk_path = os.path.join(target_dir, chunk_name)
        
        # Save the chunk in the respective directory
        sf.write(chunk_path, chunk, sr)
        y, chunk_sr = librosa.load(chunk_path, sr=None)
        assert chunk_sr == 16000, "Chunk sample rate is not 16 kHz"
        
        predictions.append(prediction)
        chunk_names.append(chunk_name)

    return predictions, chunk_names


def setup_output_directories(output_dir, audio_name):
    """Sets up output directories for predictions and chunks."""
    prediction_output_dir = os.path.join(output_dir, audio_name)
    os.makedirs(prediction_output_dir, exist_ok=True)
    chunks_dir = os.path.join(prediction_output_dir, "chunks")
    os.makedirs(chunks_dir, exist_ok=True)
    return prediction_output_dir, chunks_dir

def generate_fluency_score(predictions, chunk_names, prediction_output_dir):
    """Generates a CSV of predictions and calculates the fluency score."""
    predictions_df = pd.DataFrame({'ChunkName': chunk_names, 'Prediction': predictions})
    predictions_df.to_csv(os.path.join(prediction_output_dir, 'chunk_predictions.csv'), index=False)
    return np.mean(predictions)

def predict_and_score(audio_path, model_path, scaler_path, output_dir):
    """Main function to process audio and generate fluency score."""
    y, sr = librosa.load(audio_path, sr=None)
    chunks = split_audio_signal(y, sr)
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    audio_name = os.path.basename(audio_path).replace('.wav', '')

    prediction_output_dir, chunks_dir = setup_output_directories(output_dir, audio_name)
    predictions, chunk_names = save_chunks_and_predict(chunks, chunks_dir, sr, model, scaler)
    fluency_score = generate_fluency_score(predictions, chunk_names, prediction_output_dir)
    
    return fluency_score

def convert_audio_to_mono_wav_safe(audio_path_orig):
    """Safely converts an audio file to 16kHz mono WAV format, replacing the original file."""
    # Create a temporary path for the converted file
    
    print("Converting the audio file to the correct format")
    temp_wav_path = audio_path_orig + ".temp.wav"
    
    # Construct the ffmpeg command line for conversion
    ffmpeg_cmd = f"ffmpeg -y -i \"{audio_path_orig}\" -ac 1 -ar 16000 \"{temp_wav_path}\" -loglevel error"
    
    # Execute the command using subprocess
    try:
        subprocess.check_call(ffmpeg_cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        # If conversion was successful, replace the original file with the converted one
        shutil.move(temp_wav_path, audio_path_orig)
    except subprocess.CalledProcessError:
        print("Error during conversion. The original file has not been modified.")
        # Remove the temporary file if the conversion failed
        os.remove(temp_wav_path)

def is_audio_in_target_format(audio_path, target_sr=16000, target_channels=1):
    """
    Checks if the audio file is already in the target sample rate and channel configuration.
    
    Returns:
    - True if the audio file meets the target specifications, False otherwise.
    """
    y, sr = librosa.load(audio_path, sr=None, mono=False)
    current_channels = 1 if len(y.shape) == 1 else y.shape[0]
    
    return sr == target_sr and current_channels == target_channels

def setup_arguments():
    """Sets up command-line arguments for the script."""
    parser = argparse.ArgumentParser(description="Generate fluency score from an audio clip.")
    parser.add_argument("model_path", help="Path to the ML model file.")
    parser.add_argument("scaler_path", help="Path to the ML scaler file.")
    parser.add_argument("audio_clip_path", help="Path to the audio clip.")
    parser.add_argument("output_dir", help="Directory to save output.")
    return parser.parse_args()

def testArgs():    
    args = argparse.Namespace()
    args.model_path = "C:\\Users\\ojmar\\Documents\\Uni\\Synoptic Project\\StammerScore\\ML Models\\Strict-Binary-RandF-gpu-optimised\\model.joblib"
    args.scaler_path = "C:\\Users\\ojmar\\Documents\\Uni\\Synoptic Project\\StammerScore\\ML Models\Strict-Binary-RandF-gpu-optimised\\scaler.joblib"
    # args.audio_clip_path = "C:\\Users\\ojmar\\Documents\\Uni\\Synoptic Project\\StammerScore\\AudioFiles\\test_audio\\Creating-a-Safe-Space-at-Work-For People-Who-Stutter.wav"
    # args.audio_clip_path = "C:\\Users\\ojmar\\Documents\\Uni\\Synoptic Project\\StammerScore\\AudioFiles\\test_audio\\do-schools-kill-creativity-sir-ken-robinson-ted.wav"
    # args.audio_clip_path = "C:\\Users\\ojmar\\Documents\\Uni\\Synoptic Project\\StammerScore\\AudioFiles\\test_audio\\6-steps-to-overcoming-self-doubt-and-conquering-your-fears-lewis-howes-short.wav"
    # args.audio_clip_path = "C:\\Users\\ojmar\\Documents\\Uni\\Synoptic Project\\StammerScore\\AudioFiles\\test_audio\\My Stuttering Life Podcast Presents - My Journey From PWS To PWSS.wav"
    args.audio_clip_path = "C:\\Users\\ojmar\\Documents\\Uni\\Synoptic Project\\StammerScore\\AudioFiles\\test_audio\\How Placebo Effects Work to Change Our Biology & Psychology - short.wav"
    args.output_dir = "C:\\Users\\ojmar\\Documents\\Uni\\Synoptic Project\\StammerScore\\ML Models\\Strict-Binary-RandF-gpu-optimised"
    return args

if __name__ == "__main__":
    # args = setup_arguments()
    args = testArgs()
    if not is_audio_in_target_format(args.audio_clip_path):
        convert_audio_to_mono_wav_safe(args.audio_clip_path)
    else:
        print("Audio file is already in the target format. No conversion needed.")

    fluency_score = predict_and_score(args.audio_clip_path, args.model_path, args.scaler_path, args.output_dir)
    print(f"Fluency Score: {fluency_score}")