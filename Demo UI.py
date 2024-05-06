import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from generate_fluency_score import ui_integrator
import threading

def update_progress(current, total):
    progress = int((current / total) * 100)
    progress_bar['value'] = progress
    root.update_idletasks()

def process_file(file_path):
    score = ui_integrator(file_path, update_progress_callback=update_progress)
    def update_gui():
        score_label.config(text=f"The score for your audio file is: {score}")
        # Reset progress bar after processing is complete
        progress_bar['value'] = 0  
    root.after(0, update_gui)

def browse_and_process():
    file_path = filedialog.askopenfilename(filetypes=[("Audio Files", "*.mp3 *.wav *.flac")])
    if file_path:
        progress_bar['value'] = 0 
        # Start the long-running task in a separate thread
        threading.Thread(target=process_file, args=(file_path,), daemon=True).start()
    else:
        messagebox.showwarning("Warning", "No file was selected. Please select a file.")

root = tk.Tk()
root.title("Audio File Score Calculator")
root.configure(bg='#d7f5dd')

btn_browse = tk.Button(root, text="Browse and Process Audio File", command=browse_and_process, bg='#e0e0e0', padx=10, pady=5)
btn_browse.pack(pady=20, padx=20)

progress_bar = ttk.Progressbar(root, orient='horizontal', mode='determinate', length=280)
progress_bar.pack(pady=10, padx=20)

score_label = tk.Label(root, text="Score will be displayed here", bg='#d7f5dd')
score_label.pack(pady=20)

root.minsize(400, 200)

root.mainloop()
