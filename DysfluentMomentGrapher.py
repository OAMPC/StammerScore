import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Function to plot the fluency chart
def plot_ML_fluency_chart(csv_path):
    # Load the CSV into a DataFrame
    df = pd.read_csv(csv_path)
    
    # Assume each chunk represents 3 seconds
    chunk_duration = 3
    
    # Calculate the start time of each chunk
    df['Start'] = df.index * chunk_duration
    
    # Set up the plot
    fig, ax = plt.subplots(figsize=(10, 1)) # Adjust the figsize to fit your needs
    
    # For each chunk, plot a segment in the bar
    for _, row in df.iterrows():
        color = 'green' if row['Prediction'] == 1 else 'red'
        ax.broken_barh([(row['Start'], chunk_duration)], (0, 1), facecolors=color)
    
    # Set plot title and labels
    ax.set_title('Audio Fluency Chart')
    ax.set_yticks([])
    ax.set_xlabel('Time (seconds)')
    
    # Set the x-axis limits to match the total duration of the audio
    total_duration = len(df) * chunk_duration
    ax.set_xlim(0, total_duration)
    
    # Hide the spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    plt.show()

# Example usage
# csv_path = 'ML Models/Strict-Binary-RandF_optimised/How Placebo Effects Work to Change Our Biology & Psychology - short/chunk_predictions.csv
# csv_path = 'Evaluation\Marks\Processed\MSLP - Ollie.csv'
csv_path = 'ML Models\Strict-Binary-RandF-basic\My Stuttering Life Podcast Presents - My Journey From PWS To PWSS\chunk_predictions.csv'
plot_ML_fluency_chart(csv_path)
