import json
import numpy as np


def normalize_matrices(matrix_dict):
    """Normalize each row in each matrix within the dictionary.

    Parameters:
        matrix_dict (dict): A dictionary where each key is associated with a matrix
                            represented as a list of lists (each list being a row of the matrix).

    Returns:
        dict: A dictionary with the same keys but with each matrix row normalized.
    """
    normalized_dict = {}
    for key, matrix in matrix_dict.items():
        normalized_matrix = []
        for row in matrix:
            arr = np.array(row)
            norm = np.linalg.norm(arr)
            if norm == 0:
                print(f"Warning: Zero-length vector in matrix '{key}', row cannot be normalized.")
                normalized_row = row  # or handle this however you deem appropriate
            else:
                normalized_row = (arr / norm).tolist()
            normalized_matrix.append(normalized_row)
        normalized_dict[key] = normalized_matrix
    return normalized_dict

def normalize_vectors(vector_dict):
    """Normalize each vector in the dictionary to have a unit norm."""
    normalized_dict = {}
    for key, vector in vector_dict.items():
        arr = np.array(vector)
        norm = np.linalg.norm(arr)
        if norm == 0:
            print(f"Warning: Zero-length vector for key {key}, cannot normalize.")
            normalized_dict[key] = vector  # or handle this however you deem appropriate
        else:
            normalized_dict[key] = (arr / norm).tolist()
    return normalized_dict

def normalize_nested_vectors(nested_vector_dict):
    """Normalize each vector in a nested dictionary to have a unit norm.

    Parameters:
        nested_vector_dict (dict): A dictionary of dictionaries, where inner dictionaries contain lists as vectors.

    Returns:
        dict: A nested dictionary with all vectors normalized.
    """
    normalized_dict = {}
    for outer_key, sub_dict in nested_vector_dict.items():
        normalized_dict[outer_key] = {}
        for inner_key, vector in sub_dict.items():
            arr = np.array(vector)
            norm = np.linalg.norm(arr)
            if norm == 0:
                print(f"Warning: Zero-length vector for key {outer_key}-{inner_key}, cannot normalize.")
                normalized_dict[outer_key][inner_key] = vector  # or handle this however you deem appropriate
            else:
                normalized_dict[outer_key][inner_key] = (arr / norm).tolist()
    return normalized_dict

def write_dict_to_json(data, file_path):
    """Write the dictionary to a JSON file."""
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)

with open('data/experiment/semantic_space/vectors_bncwac20m.json', 'r') as inj:
    vector_dict = json.load(inj)

normalized_vectors = normalize_vectors(vector_dict)
output_file_path = 'data/experiment/semantic_space/vectors_normalized.json'
write_dict_to_json(normalized_vectors, output_file_path)
print(f"Normalized vectors written to {output_file_path}")