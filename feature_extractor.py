import numpy as np
import librosa

N_MFCC = 13

def extract_features_basic(audio_signal, sr, n_mfcc=N_MFCC):
    """Extracts MFCC features from an audio signal."""
    mfccs = librosa.feature.mfcc(y=audio_signal, sr=sr, n_mfcc=n_mfcc)
    return np.mean(mfccs, axis=1)

def extract_features_advanced(audio_signal, sr, n_fft=2048, hop_length=None, n_mels=40, fmax=8000):
    """Extracts advanced features including Mel-filterbank energy features and pitch features from an audio signal."""
    if hop_length is None:
        hop_length = int(sr * 0.01)
    n_fft = int(sr * 0.025)
    
    mfb_features = librosa.feature.melspectrogram(y=audio_signal, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, fmax=fmax)
    pitches, magnitudes = librosa.piptrack(y=audio_signal, sr=sr, n_fft=n_fft, hop_length=hop_length)
    pitch_mean = np.mean(pitches, axis=1)
    pitch_delta = np.diff(pitch_mean)
    voicing_feature = np.mean(magnitudes, axis=1)
    
    features = np.concatenate([np.mean(mfb_features, axis=1), pitch_mean[:-1], pitch_delta, voicing_feature[:-1]])
    return features

def load_audio_and_extract_features(audio_path, extraction_type='basic', n_mfcc=N_MFCC):
    """Loads an audio file and extracts features based on the specified extraction type."""
    y, sr = librosa.load(audio_path, sr=None)
    if extraction_type == 'basic':
        return extract_features_basic(y, sr, n_mfcc)
    elif extraction_type == 'advanced':
        return extract_features_advanced(y, sr)
    else:
        raise ValueError("Invalid extraction type specified. Choose either 'basic' or 'advanced'.")
