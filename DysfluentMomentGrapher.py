import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_ML_fluency_chart(csv_path):
    df = pd.read_csv(csv_path)
    
    chunk_duration = 3
    df['Start'] = df.index * chunk_duration
    
    fig, ax = plt.subplots(figsize=(10, 1))
    
    for _, row in df.iterrows():
        color = 'green' if row['Prediction'] == 1 else 'red'
        ax.broken_barh([(row['Start'], chunk_duration)], (0, 1), facecolors=color)
    
    ax.set_title('Audio Fluency Chart')
    ax.set_yticks([])
    ax.set_xlabel('Time (seconds)')
    
    total_duration = len(df) * chunk_duration
    ax.set_xlim(0, total_duration)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    plt.show()

csv_path = 'ML Models\Strict-Binary-RandF-basic\My Stuttering Life Podcast Presents - My Journey From PWS To PWSS\chunk_predictions.csv'
plot_ML_fluency_chart(csv_path)
