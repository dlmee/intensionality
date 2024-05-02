import os
import json
import numpy as np

def load_polysemy_and_frequency_data(filepath):
    """Loads polysemy counts and frequency data from a JSON file."""
    with open(filepath, 'r') as file:
        data = json.load(file)
    return {item[0]: {'polysemy': item[1], 'frequency': item[2]} for item in data}

def extract_values_and_store_by_file(directory, key, polysemy_data):
    """Extracts values from JSON files and organizes them by word, keeping one score per file source."""
    word_data = {}

    for filename in os.listdir(directory):
        if filename.endswith('.json'):
            filepath = os.path.join(directory, filename)
            file_key = filename.split('.')[0]  # Use the part of the filename before the extension as the key
            with open(filepath, 'r') as file:
                data = json.load(file)
                for adj, nouns in data.items():
                    for noun, metrics in nouns.items():
                        if noun in polysemy_data:
                            if key in metrics:
                                score = metrics[key]
                                if noun not in word_data:
                                    word_data[noun] = {
                                        'polysemy': polysemy_data[noun]['polysemy'],
                                        'frequency': polysemy_data[noun]['frequency'],
                                        'scores': {}
                                    }
                                if file_key not in word_data[noun]['scores']:
                                    word_data[noun]['scores'][file_key] = []
                                word_data[noun]['scores'][file_key].append(score)

    # Sort words by polysemy count, descending
    sorted_words = sorted(word_data.items(), key=lambda x: x[1]['polysemy'], reverse=True)
    sorted_word_data = {word: info for word, info in sorted_words}

    return sorted_word_data

def extract_adj_values_and_store_by_file(directory, key, polysemy_data):
    """Extracts values from JSON files, treating adjectives as primary keys and nouns as subkeys, keeping one score per file source."""
    adj_data = {}

    for filename in os.listdir(directory):
        if filename.endswith('.json'):
            filepath = os.path.join(directory, filename)
            file_key = filename.split('.')[0]  # Use the part of the filename before the extension as the key
            with open(filepath, 'r') as file:
                data = json.load(file)
                for adj, nouns in data.items():
                    if adj not in polysemy_data:
                        continue  # Skip if the adjective is not in the polysemy data
                    if adj not in adj_data:
                        adj_data[adj] = {
                            'polysemy': polysemy_data[adj]['polysemy'],
                            'frequency': polysemy_data[adj]['frequency'],
                            'scores': {}
                        }
                    for noun, metrics in nouns.items():
                        if key in metrics:
                            score = metrics[key]
                            if file_key not in adj_data[adj]['scores']:
                                adj_data[adj]['scores'][file_key] = []
                            adj_data[adj]['scores'][file_key].append(score)

    # Sort adjectives by polysemy count, descending
    sorted_adjs = sorted(adj_data.items(), key=lambda x: x[1]['polysemy'], reverse=True)
    sorted_adj_data = {adj: info for adj, info in sorted_adjs}

    return sorted_adj_data

# Usage example
directory = 'data/experiment/results'
polysemy_noun_file = 'data/experiment/nn_final_counts.json'
polysemy_adj_file = 'data/experiment/adj_final_counts.json'
key = 'distance_between_composition_and_observed_phrase'

# Load polysemy data for nouns and adjectives
polysemy_noun_data = load_polysemy_and_frequency_data(polysemy_noun_file)
polysemy_adj_data = load_polysemy_and_frequency_data(polysemy_adj_file)

# Extract values sorted by polysemy for both nouns and adjectives
sorted_noun_data = extract_values_and_store_by_file(directory, key, polysemy_noun_data)
sorted_adj_data = extract_adj_values_and_store_by_file(directory, key, polysemy_adj_data)

# Print the sorted data
def print_word_data(sorted_data, word_type):
    table = []
    for i, (word, details) in enumerate(sorted_data.items()):
        #if i % 5 != 0: continue
        #if word not in ['white', 'alleged', 'impossible', 'black', 'former', 'likely', 'putative','natural', 'solid', 'active']: continue
        # Prepare to display the average scores for each file
        average_scores = {file_key: np.mean(scores) for file_key, scores in details['scores'].items()}
        # Format the scores as a string for better readability
        scores_str = ', '.join([f"{file_key}: {avg_score:.3f}" for file_key, avg_score in average_scores.items()])
        table.append(f"{word} ({word_type}): Senses: {details['polysemy']}, Frequency: {details['frequency']}, Average Scores: {scores_str}")
    return table
# Print the sorted and averaged data for nouns and adjectives
t1 = print_word_data(sorted_noun_data, "Noun")
t2 = print_word_data(sorted_adj_data, "Adj")

with open('data/experiment/analysis/noun_analysis.json', 'w') as outj:
    json.dump(t1, outj, indent=4, ensure_ascii=False)
with open('data/experiment/analysis/adj_analysis.json', 'w') as outj:
    json.dump(t2, outj, indent=4, ensure_ascii=False)

