import tkinter as tk
from tkinter import filedialog, ttk
import pygame
import csv
import time
from tkinter.simpledialog import askstring
from mutagen.mp3 import MP3
import wave
import pandas as pd

class DysfluencyMarkerApp:
    def __init__(self, master):
        self.master = master
        master.title("Dysfluency Marker")
        pygame.mixer.init()
        self.audio_loaded = False
        self.is_playing = False
        self.pause_time = 0  # Add an attribute to track pause time

        # Layout
        self.play_button = tk.Button(master, text="Play Audio", command=self.toggle_playback)
        self.play_button.pack()

        self.mark_button = tk.Button(master, text="Mark Dysfluency", command=self.mark_dysfluency, state=tk.DISABLED)
        self.mark_button.pack()

        self.progress = ttk.Progressbar(master, orient="horizontal", length=200, mode="determinate")
        self.progress.pack()

        self.status_label = tk.Label(master, text="Load an audio file to get started.", wraplength=300)
        self.status_label.pack()

        # Load audio file
        self.load_audio_file()

    def reset_ui(self):
        self.progress["value"] = 0
        self.is_playing = False
        self.audio_loaded = False
        self.mark_button.config(state=tk.DISABLED)
        self.play_button.config(state=tk.NORMAL, text="Play Audio")
        self.status_label.config(text="Load an audio file to get started.")

    def load_audio_file(self):
        self.audio_file = filedialog.askopenfilename(title="Select an Audio File",
                                             filetypes=(("Audio Files", "*.mp3;*.wav"),))

        if self.audio_file:
            self.status_label.config(text=f"Loaded: {self.audio_file.split('/')[-1]}")
            pygame.mixer.music.load(self.audio_file)
            self.audio_loaded = True
            
            # Determine the duration of the audio file
            if self.audio_file.endswith(".mp3"):
                audio = MP3(self.audio_file)
                duration = audio.info.length
            elif self.audio_file.endswith(".wav"):
                with wave.open(self.audio_file, 'r') as wav_file:
                    frames = wav_file.getnframes()
                    rate = wav_file.getframerate()
                    duration = frames / float(rate)
            else:
                # Placeholder for unsupported formats
                self.status_label.config(text="Unsupported file format.")
                duration = 0
            
            # Set the progress bar's maximum value to the duration of the audio file
            self.progress["maximum"] = duration

            # Ask for the test name
            test_name = askstring("Test Name", "Enter the name of the test:")
            if test_name:  # Ensure a name was entered
                print("Create CSV")
                self.csv_file = f"Evaluation\Marks\Raw\{test_name}.csv"
                self.output_csv_file = f"Evaluation\Marks\Processed\{test_name}.csv"
                with open(self.csv_file, mode='w', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow(["Timestamp", "Mark"])
            else:
                self.status_label.config(text="No test name provided. Default CSV name will be used.")
        else:
            self.status_label.config(text="No file loaded. Please restart and select a file.")


    def toggle_playback(self):
        if self.audio_loaded and not self.is_playing:
            pygame.mixer.music.play()
            self.is_playing = True
            self.mark_button.config(state=tk.NORMAL)
            self.play_button.config(state=tk.DISABLED)
            self.update_progress_bar()
        elif not self.audio_loaded:
            # If no audio is loaded, open the file dialog to select a new file
            self.load_audio_file()

    def update_progress_bar(self):
        if self.is_playing:
            current_pos = pygame.mixer.music.get_pos() / 1000
            duration = self.progress["maximum"]
            if current_pos >= 0 and current_pos < duration:
                self.progress["value"] = current_pos
                self.master.after(100, self.update_progress_bar)
            else:
                self.progress["value"] = duration
                self.play_button.config(text="Play Again?")
                self.is_playing = False
                self.mark_button.config(state=tk.DISABLED)
                self.convert_csv_format(self.csv_file, self.output_csv_file)
                self.reset_ui()

    def mark_dysfluency(self):
        if self.audio_loaded:
            current_time = pygame.mixer.music.get_pos() / 1000
            total_duration = self.progress["maximum"]
            with open(self.csv_file, mode='a', newline='') as file:                
                print(f"Newline added to {file}")
                writer = csv.writer(file)
                timestamp_ratio = f"{current_time:.2f}/{total_duration:.2f}"
                writer.writerow([timestamp_ratio, "Dysfluency"])
            self.status_label.config(text=f"Dysfluency marked at {current_time:.2f} seconds.")
    
    def convert_csv_format(self, input_csv_path, output_csv_path):
        # Read the CSV into a DataFrame
        df = pd.read_csv(input_csv_path)
        
        # Extract the total duration from the first row's Timestamp
        total_duration = float(df['Timestamp'].iloc[0].split('/')[1])
        
        # Calculate the number of chunks
        num_chunks = int(total_duration // 3) + (1 if total_duration % 3 > 0 else 0)
        
        # Initialize the prediction dictionary with all chunks as fluent (1)
        predictions = {f"chunk_{i}.wav": 1 for i in range(num_chunks)}
        
        # Mark chunks containing a dysfluent moment as 0
        for index, row in df.iterrows():
            dysfluent_time = float(row['Timestamp'].split('/')[0])
            chunk_index = int(dysfluent_time // 3)
            predictions[f"chunk_{chunk_index}.wav"] = 0
        
        # Convert the dictionary to a DataFrame
        predictions_df = pd.DataFrame(list(predictions.items()), columns=['ChunkName', 'Prediction'])
        
        # Write the new CSV file
        predictions_df.to_csv(output_csv_path, index=False)

if __name__ == "__main__":
    root = tk.Tk()
    app = DysfluencyMarkerApp(root)
    root.mainloop()
