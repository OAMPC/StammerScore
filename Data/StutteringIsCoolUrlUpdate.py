import csv
import re

# Define your input and output file paths
input_file_path = r"C:\Users\ojmar\Documents\Uni\StammerScore\ml-stuttering-events-dataset\stutteringiscool.csv"
output_file_path = r"C:\Users\ojmar\Documents\Uni\StammerScore\ml-stuttering-events-dataset\stutteringiscool_modified.csv"

# Function to convert URLs
def convert_url(original_url):
    episode_part = re.search(r'/([^/]+\.mp3)$', original_url)
    if episode_part:
        new_url = f"http://media.blubrry.com/stutteringiscool/stutteringiscool.com/sound/{episode_part.group(1)}"
        return new_url
    return original_url

# Read from the input file, modify the data, and write to the output file
with open(input_file_path, mode='r', encoding='utf-8') as infile, open(output_file_path, mode='w', newline='', encoding='utf-8') as outfile:
    csv_reader = csv.reader(infile)
    csv_writer = csv.writer(outfile)

    for row in csv_reader:
        if row:  # Check if the row is not empty
            original_url = row[2]  # Assuming URLs are in the third column
            row[2] = convert_url(original_url)  # Convert and replace the URL
            csv_writer.writerow(row)  # Write the modified row to the output file

# The modified data is now saved to 'stutteringiscool_modified.csv'
