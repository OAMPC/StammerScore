import os
import random
import tkinter as tk
from tkinter import messagebox
from pygame import mixer

# Path and category settings
BASE_PATH = "C:/Users/ojmar/Documents/Uni/Synoptic Project/StammerScore/Evaluation/Tutorial/Clips"
stuttering_categories = {
    'Prolongation': ('Prolongation', 'Elongated syllable (e.g., M[mmm]ommy)'),
    'Block': ('Block', 'Gasps for air or stuttered pauses'),
    'Sound Repetition': ('Sound Repetition', 'Repeated syllables (e.g., I [pr-pr-pr-]prepared dinner)'),
    'Word Repetition': ('Word Repetition', 'The same word or phrase is repeated (e.g., I made [made] dinner)'),
    'Interjection': ('Interjection', 'Common filler words such as "um" or "uh" or person-specific filler words')
}

class StutteringTestApp:
    def __init__(self, master):
        self.master = master
        master.title("Stuttering Event Classification Test")
        mixer.init()

        self.start_tutorial()

    def start_tutorial(self):
        self.tutorial_frame = tk.Frame(self.master, padx=10, pady=10)
        self.tutorial_frame.pack(padx=10, pady=10)

        self.label = tk.Label(self.tutorial_frame, text="Welcome to the Stuttering Event Classification Tutorial.\n\nThis will teach you about what is classed as a stutter/dysfluent moment in regards to this experiment.\n\nClick 'Next' to learn about each category.")
        self.label.pack(pady=10)

        self.next_button = tk.Button(self.tutorial_frame, text="Next", command=self.show_categories)
        self.next_button.pack()

        self.category_index = 0

    def show_categories(self):
        # Remove previous tutorial content
        for widget in self.tutorial_frame.winfo_children():
            widget.destroy()

        if self.category_index < len(stuttering_categories):
            category, description = list(stuttering_categories.values())[self.category_index]
            self.label = tk.Label(self.tutorial_frame, text=f"{category}: {description}")
            self.label.pack(pady=(10, 20))

            play_button = tk.Button(self.tutorial_frame, text="Play Example", command=lambda: self.play_example(category))
            play_button.pack(pady=5)

            next_button = tk.Button(self.tutorial_frame, text="Next", command=self.show_categories)
            next_button.pack(pady=10)

            self.category_index += 1
        else:
            self.start_test()

    def play_example(self, category):
        folder_path = os.path.join(BASE_PATH, category)
        files = os.listdir(folder_path)
        chosen_file = random.choice(files)
        mixer.music.load(os.path.join(folder_path, chosen_file))
        mixer.music.play()

    def start_test(self):
        # Clear the tutorial frame and setup the testing environment
        self.tutorial_frame.destroy()
        self.setup_test_frame()

    def setup_test_frame(self):
        self.test_frame = tk.Frame(self.master, padx=10, pady=10)
        self.test_frame.pack(padx=10, pady=10)

        label = tk.Label(self.test_frame, text="Now, listen to the audio and select the correct stuttering event type:")
        label.pack(pady=(0, 20))

        play_button = tk.Button(self.test_frame, text="Play Random Clip", command=self.play_random_clip)
        play_button.pack(side=tk.TOP, padx=5, pady=5)

        self.variable = tk.StringVar(self.master)
        self.variable.set("Choose type")
        self.variable.trace("w", self.enable_submit_button)  # Trace changes to this variable

        menu = tk.OptionMenu(self.test_frame, self.variable, *stuttering_categories.keys())
        menu.pack(pady=(0, 20))

        self.submit_button = tk.Button(self.test_frame, text="Submit Answer", command=self.check_answer, state=tk.DISABLED)
        self.submit_button.pack(side=tk.TOP, padx=5, pady=5)

        self.correct_count = 0
        self.correct_label = tk.Label(self.test_frame, text="Correct Answers: 0/10")
        self.correct_label.pack()

        self.current_category = None
        self.current_clip_path = None

    def enable_submit_button(self, *args):
        if self.variable.get() != "Choose type":
            self.submit_button.config(state=tk.NORMAL)
        else:
            self.submit_button.config(state=tk.DISABLED)

    def check_answer(self):
        user_answer = self.variable.get().strip()
        if user_answer.lower() == self.current_category.lower():
            self.correct_count += 1
            self.correct_label.config(text=f"Correct Answers: {self.correct_count}/10")
            messagebox.showinfo("Result", "Correct!")
        else:
            messagebox.showinfo("Result", f"Incorrect! Correct answer was {self.current_category}")
        self.variable.set("Choose type")
        self.submit_button.config(state=tk.DISABLED)
        self.current_category = None
        self.current_clip_path = None

        if self.correct_count >= 10:
            self.show_completion_screen()

    def play_random_clip(self):
        if not self.current_clip_path:
            category = random.choice(list(stuttering_categories.keys()))
            folder_path = os.path.join(BASE_PATH, stuttering_categories[category][0])
            files = os.listdir(folder_path)
            chosen_file = random.choice(files)
            self.current_category = category
            self.current_clip_path = os.path.join(folder_path, chosen_file)

        mixer.music.load(self.current_clip_path)
        mixer.music.play()

    def show_completion_screen(self):
        # Clear the test frame and show the completion message
        self.test_frame.destroy()
        completion_frame = tk.Frame(self.master, padx=10, pady=10)
        completion_frame.pack(padx=10, pady=10)
        completion_label = tk.Label(completion_frame, text="Congratulations! You have completed the tutorial.\nYou can now move on to the next activity.")
        completion_label.pack(pady=20)

def main():
    root = tk.Tk()
    app = StutteringTestApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
