import json
from numpy import linspace

# Load the data from the JSON file
with open('data/experiment/analysis/noun_analysis.json', 'r') as file:
    data = json.load(file)

# Function to parse the data string and extract detailed information
def parse_data(entry):
    parts = entry.split(":")
    word_details = parts[0].strip().split()
    word = word_details[0]
    senses = int(word_details[2])
    frequency = int(parts[1].split(",")[0].strip())
    scores = parts[2].strip().split(", ")
    score_dict = {score.split()[0]: float(score.split()[1]) for score in scores}
    return word, senses, frequency, score_dict

# Parse each entry to get detailed information
parsed_data = [parse_data(entry) for entry in data]

# Sort by sense count to evenly sample
parsed_data.sort(key=lambda x: x[1], reverse=True)

# Sample evenly across the range of sense counts
sample_size = 20
indices = [int(i) for i in linspace(0, len(parsed_data) - 1, sample_size)]
sampled_data = [parsed_data[i] for i in indices]

# Format for Google Docs
formatted_table = "Word | Senses | Frequency | HOS | SC | HOA | HOAO\n"
formatted_table += "-" * 60 + "\n"
for word, senses, frequency, scores in sampled_data:
    formatted_table += f"{word} | {senses} | {frequency} | {scores['higher_order_selective']:.3f} | {scores['simple_combination']:.3f} | {scores['higher_order_aggregated']:.3f} | {scores['higher_order_adjs_only']:.3f}\n"

print(formatted_table)
