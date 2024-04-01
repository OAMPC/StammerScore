import numpy as np
import librosa

N_MFCC = 13

def extract_features_basic(audio_signal, sr, n_mfcc=N_MFCC):
    """
    Extracts MFCC features from an audio signal.

    Parameters:
    - audio_signal: The audio signal array.
    - sr: The sample rate of the audio signal.
    - n_mfcc: The number of Mel-frequency cepstral coefficients to extract.

    Returns:
    The mean of the extracted MFCC features across the time axis.
    """
    mfccs = librosa.feature.mfcc(y=audio_signal, sr=sr, n_mfcc=n_mfcc)
    return np.mean(mfccs, axis=1)

def extract_features_advanced(audio_signal, sr, n_fft=2048, hop_length=None, n_mels=40, fmax=8000):
    """
    Extracts advanced features including Mel-filterbank energy features and pitch features from an audio signal.

    Parameters:
    - audio_signal: The audio signal array.
    - sr: The sample rate of the audio signal.
    - n_fft: The length of the FFT window.
    - hop_length: Number of samples between successive frames.
    - n_mels: Number of Mel bands to generate.
    - fmax: Highest frequency (in Hz).

    Returns:
    A combined feature array including Mel-filterbank energies, pitch mean, pitch delta, and voicing features.
    """
    if hop_length is None:
        hop_length = int(sr * 0.01)  # Default hop_length to frame rate of 100 Hz
    n_fft = int(sr * 0.025)  # Default n_fft to a 25 ms window
    
    mfb_features = librosa.feature.melspectrogram(y=audio_signal, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, fmax=fmax)
    pitches, magnitudes = librosa.piptrack(y=audio_signal, sr=sr, n_fft=n_fft, hop_length=hop_length)
    pitch_mean = np.mean(pitches, axis=1)
    pitch_delta = np.diff(pitch_mean)
    voicing_feature = np.mean(magnitudes, axis=1)
    
    features = np.concatenate([np.mean(mfb_features, axis=1), pitch_mean[:-1], pitch_delta, voicing_feature[:-1]])
    return features

def load_audio_and_extract_features(audio_path, extraction_type='basic', n_mfcc=N_MFCC):
    """
    Loads an audio file and extracts features based on the specified extraction type.

    Parameters:
    - audio_path: The path to the audio file.
    - extraction_type: Type of feature extraction ('basic' or 'advanced').
    - n_mfcc: The number of Mel-frequency cepstral coefficients to extract (only for 'basic').

    Returns:
    The extracted features as specified by the extraction type.
    """
    y, sr = librosa.load(audio_path, sr=None)
    if extraction_type == 'basic':
        return extract_features_basic(y, sr, n_mfcc)
    elif extraction_type == 'advanced':
        return extract_features_advanced(y, sr)
    else:
        raise ValueError("Invalid extraction type specified. Choose either 'basic' or 'advanced'.")
