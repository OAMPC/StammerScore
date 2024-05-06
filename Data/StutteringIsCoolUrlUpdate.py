import csv
import re

'''Requires as the StutteringIsCool data has moved to a different endpoint from what the original dataset states'''

input_file_path = r"C:\Users\ojmar\Documents\Uni\StammerScore\ml-stuttering-events-dataset\stutteringiscool.csv"
output_file_path = r"C:\Users\ojmar\Documents\Uni\StammerScore\ml-stuttering-events-dataset\stutteringiscool_modified.csv"

def convert_url(original_url):
    episode_part = re.search(r'/([^/]+\.mp3)$', original_url)
    if episode_part:
        new_url = f"http://media.blubrry.com/stutteringiscool/stutteringiscool.com/sound/{episode_part.group(1)}"
        return new_url
    return original_url

with open(input_file_path, mode='r', encoding='utf-8') as infile, open(output_file_path, mode='w', newline='', encoding='utf-8') as outfile:
    csv_reader = csv.reader(infile)
    csv_writer = csv.writer(outfile)

    for row in csv_reader:
        if row:
            original_url = row[2]
            row[2] = convert_url(original_url)
            csv_writer.writerow(row)

