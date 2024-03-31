import argparse
import os
import numpy as np
import pandas as pd
import joblib
import librosa
import soundfile as sf

def split_audio_signal(y, sr, chunk_length=3):
    chunk_size = chunk_length * sr  # Chunk size in samples
    chunks = [(y[i:i + chunk_size], i // chunk_size) for i in range(0, len(y), chunk_size) if i + chunk_size <= len(y)]
    return chunks

# Feature extraction function
def extract_features(audio_signal, sr):
    mfccs = np.mean(librosa.feature.mfcc(y=audio_signal, sr=sr, n_mfcc=13), axis=1)
    # You can add more features here
    return mfccs

def predict_and_score(audio_path, model_path, scaler_path, output_dir):
    # Load the audio file
    y, sr = librosa.load(audio_path, sr=None)
    # Split the audio signal into 3-second chunks
    chunks = split_audio_signal(y, sr)
    
    # Load the saved model and scaler
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    predictions = []
    chunk_names = []
    for chunk, index in chunks:
        # Save each chunk as an audio file
        chunk_name = f"chunk_{index}.wav"
        chunk_path = os.path.join(output_dir, chunk_name)
        sf.write(chunk_path, chunk, sr)
        chunk_names.append(chunk_name)
        
        features = extract_features(chunk, sr)
        features = scaler.transform(np.array(features).reshape(1, -1))  # Scale the features
        prediction = model.predict(features)[0]
        predictions.append(prediction)
    
    # Create a DataFrame with chunk names and predictions
    predictions_df = pd.DataFrame({
        'ChunkName': chunk_names,
        'Prediction': predictions
    })
    predictions_df.to_csv(os.path.join(output_dir, 'chunk_predictions.csv'), index=False)
    
    # Calculate the final fluency score (optional)
    fluency_score = np.mean(predictions)
    
    return fluency_score

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process audio clip and generate fluency score.")
    parser.add_argument("model_path", type=str, help="Path to the ML model joblib file.")
    parser.add_argument("scaler_path", type=str, help="Path to the ML scaler joblib file.")
    parser.add_argument("audio_clip_path", type=str, help="Path to the audio clip.")
    parser.add_argument("output_dir", type=str, help="Directory to save chunks and predictions.")
    
    args = parser.parse_args()
    
    # Calculate and print the fluency score
    fluency_score = predict_and_score(args.audio_clip_path, args.model_path, args.scaler_path, args.output_dir)
    print(f"Fluency Score: {fluency_score}")
