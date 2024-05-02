import xml.etree.ElementTree as ET
import os
from tqdm import tqdm
import json
import numpy as np
import random
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import csr_matrix
import math


class Intension:
    def __init__(self, root, initial = True, first=None, second=None, third=None, index=None) -> None:
        if initial:
            paths = self.find_xml_files(root)
            if not first:
                self.first = self.first_pass(paths)
                with open("first_pass.json", "w") as outj:
                    json.dump(self.first, outj, indent=4, ensure_ascii=False)
            else:
                with open(first, 'r') as inj:
                    self.first = json.load(inj)
            if not second:
                self.second = self.second_pass(self.first, paths)
                with open("second_pass.json", "w") as outj:
                    json.dump(self.second, outj, indent=4, ensure_ascii=False)
            else:
                with open(second, 'r') as inj:
                    self.second = json.load(inj)
            if not third:
                self.third = self.third_pass(self.second, paths, index)
                with open("third_pass.json", "w") as outj:
                    json.dump(self.third, outj, indent=4, ensure_ascii=False)
            else:
                with open(third, 'r') as inj:
                    self.third = json.load(inj)
                with open(index, "r") as inj:
                    self.index = json.load(inj)
            print("and done!")

    def first_pass(self, paths):
        target_pos = ['SUBST', 'VERB', 'ADJ', 'ADV']
        fp = {pos:{} for pos in target_pos}
        for path in tqdm(paths, desc="Getting word counts"):
            if path.startswith('Texts'):
                sentences = self.read_sentences_from_bnc(path)
            else:
                sentences = self.read_sentences_from_ukwac(path)
            if sentences:
                for sentence in sentences:
                    for word, pos in sentence:
                        if pos in target_pos:
                            if word not in fp[pos]:
                                fp[pos][word] = 1
                            else:
                                fp[pos][word] += 1
        return fp
    
    def second_pass(self, first_stem, paths):
        #First a little rearranging, we only want the top 10k across all categories.
        target_pos = ['SUBST', 'VERB', 'ADJ', 'ADV']
        first = {pos:{} for pos in target_pos}
        second = {pos:{} for pos in target_pos}
        word_counts = sorted([(word, k, count) for k,v in first_stem.items() for word, count in v.items()], key= lambda x:x[2], reverse = True)
        for word, pos, count in word_counts[1000:11000]:
            if word.lower() not in first[pos]:
                first[pos][word.lower()] = count
        
        #Curious about the balance there. 
        for k,v in first.items():
            print(f"The amount of words in {k} is {len(v)}")
        for path in tqdm(paths, desc="Getting collocations, i.e. second pass"):
            sentences = self.read_sentences_from_bnc(path)
            for sentence in sentences:
                for word, pos in sentence:
                    if pos in target_pos:
                        if word in first[pos]:
                            if word not in second[pos]:
                                second[pos][word] = {pos:{} for pos in target_pos}
                            for w2, p2 in sentence:
                                if p2 in target_pos:
                                    if (word, pos) == (w2,p2): continue
                                    if w2 in first[p2]:
                                        if w2 in second[pos][word][p2]:
                                            second[pos][word][p2][w2] += 1
                                        else:
                                            second[pos][word][p2][w2] = 1
        return second

    def third_pass(self, second, paths, idx):
        third = {}
        if not idx:
            idx = {}
            counter = 0
            for k,v in second.items():
                for k2, v2 in v.items():
                    for v3 in v2.values():
                        for vec in v3:
                            if vec not in idx:
                                idx[vec] = counter
                                counter +=1
            with open("index.json", "w") as outj:
                json.dump(idx, outj, indent=4, ensure_ascii=False)
        else:
            with open(idx, 'r') as inj:
                idx = json.load(inj)
        for k,v in second.items():
            for k2, v2 in tqdm(v.items(), desc = "building out a section"):
                third[k2] = [0] * len(idx)
                for v3 in v2.values():
                    for vec, count in v3.items():
                        third[k2][idx[vec]] = count

        print("and done!")

        return third

    def find_phrasal_vectors(self, tgts, root, tenk, previous = None):
        target_pos = ['SUBST', 'VERB', 'ADJ', 'ADV']
        paths = []
        for r in root:
            paths = paths + self.find_xml_files(r)
        targets = []
        for tgt in tgts:
            with open(tgt, "r") as inj:
                targets = targets + json.load(inj)
        with open(tenk, "r") as inj:
            tenk = json.load(inj)
        tenk = [word for v in tenk.values() for word in v]
        if previous:
            with open(previous[0], "r") as inj:
                an_vectors = json.load(inj)
            with open(previous[1], "r") as inj:
                an_matrices = json.load(inj)
        else:
            an_vectors = {word:{} for word in targets}
            an_matrices = {word:{} for word in targets}
        
        for path in tqdm(paths, desc="Building Phrasal Vectors"):
            counter = 0
            
            if path.startswith('Texts'):
                sentences = self.read_sentences_from_bnc(path)
            else:
                sentences = self.read_sentences_from_ukwac(path)
            if not sentences: continue
            for sentence in tqdm(sentences, total= 1000000):
                counter += 1
                try:
                    for i, (word, pos) in enumerate(sentence):
                        if word in an_vectors:
                            for w2, p2 in sentence:
                                if w2 == word: continue
                                if w2 in tenk:
                                    if w2 not in an_vectors[word]:
                                        an_vectors[word][w2] = 1
                                    else:
                                        an_vectors[word][w2] += 1
                        if pos in ['ADJ', 'JJ']:
                            if word in an_matrices:
                                if i + 1 < len(sentence):
                                    w2, p2 = sentence[i+1]
                                    if p2 in ['SUBST', 'NN', 'NP']:
                                        if w2 in an_matrices:
                                            if w2 not in an_matrices[word]:
                                                an_matrices[word][w2] = {}
                                            for w3, p3 in sentence:
                                                if w3 != word and w3 != w2 and w3 in tenk:
                                                    if w3 in an_matrices[word][w2]:
                                                        an_matrices[word][w2][w3] += 1
                                                    else:
                                                        an_matrices[word][w2][w3] = 1
                except KeyboardInterrupt:
                    with open("vectors/AN_vectors.json", "w") as outj:
                        json.dump(an_vectors, outj, indent=4, ensure_ascii=False)
                    with open("vectors/AN_matrices.json", "w") as outj:
                        json.dump(an_matrices, outj, indent=4, ensure_ascii=False)
                    raise  # Re-raise KeyboardInterrupt to ensure it's not caught by the next except block
                except KeyError as e:
                    print(f"Something broke here! {e}")
                if counter % 1000000 == 0:
                    p = path.split("/")
                    with open(f"vectors/{p[0]}_AN_vectors.json", "w") as outj:
                        json.dump(an_vectors, outj, indent=4, ensure_ascii=False)
                    with open(f"vectors/{p[0]}AN_matrices.json", "w") as outj:
                        json.dump(an_matrices, outj, indent=4, ensure_ascii=False)
                    break
        with open(f"vectors/final_AN_vectors.json", "w") as outj:
            json.dump(an_vectors, outj, indent=4, ensure_ascii=False)
        with open(f"vectors/final_AN_matrices.json", "w") as outj:
            json.dump(an_matrices, outj, indent=4, ensure_ascii=False)

    def build_sparce_vecs(self, index, vectors = None, matrices = None):
        def apply_svd_to_matrices(matrices, max_components=5):
            transformed_matrices = {}

            # Iterate over each key in the matrices
            for k, adj_matrices in tqdm(matrices.items(), desc= "Running SVD"):
                if not adj_matrices:
                    print(f"Skipping SVD for {k} due to no data!")
                    continue
                # Create a matrix from the vectors
                data_matrix = np.array([v for v in adj_matrices.values()]).T

                        # Check if the data matrix is large enough for SVD
                if data_matrix.shape[1] < 2:  # Check if there are at least two features
                    print(f"Skipping SVD for {k} due to insufficient data. Shape: {data_matrix.shape}")
                    continue  # Skip this iteration if not enough data

                        # Adjust n_components based on the number of available features
                n_components = min(max_components, data_matrix.shape[1] - 1)
                if n_components < 1:
                    print(f"Skipping SVD for {k} as n_components must be at least 1. Available features: {data_matrix.shape[1]}")
                    continue

                sparse_matrix = csr_matrix(data_matrix)  # Convert to CSR format for efficient processing

                # Apply SVD
                svd = TruncatedSVD(n_components=n_components)
                transformed_data = svd.fit_transform(sparse_matrix).T  # Fit and transform the data

                # Store the transformed data back as a list
                transformed_matrices[k] = transformed_data.tolist()

            return transformed_matrices
        
        
        
        with open(index, "r") as inj:
            index = json.load(inj)
        
        if vectors:
            with open(vectors, "r") as inj:
                vectors = json.load(inj)
            vecs = {k:[0]*len(index) for k in vectors.keys()}
            for k,v in tqdm(vectors.items(), desc="building simple vectors"):
                for word, count in v.items():
                    vecs[k][index[word]] = count
                norm = math.sqrt(sum(y ** 2 for y in vecs[k]))
                vecs[k] = [x / norm for x in vecs[k]]

            with open("data/experiment/semantic_space/vectors_bncwac20m.json", "w") as outj:
                json.dump(vecs, outj, indent=4, ensure_ascii=False)
        
        if matrices:
            with open(matrices, "r") as inj:
                matrices = json.load(inj)
            
            svd = TruncatedSVD(n_components=5)
            # Initialize empty dictionaries for both adj_matrices and noun_matrices
            adj_matrices = {}
            noun_matrices = {}

            # Iterate through each adjective and its associated noun dictionary from 'matrices'
            for adj, noun_dict in tqdm(matrices.items(), desc="building matrices"):
                if adj not in adj_matrices:
                    adj_matrices[adj] = {}

                for noun, adj_dict in noun_dict.items():
                    # Initialize vectors in adj_matrices if not already done
                    if noun not in adj_matrices[adj]:
                        adj_matrices[adj][noun] = [0.1] * len(index)

                    # Check if there is actual data to process
                    if adj_dict:
                        # Populate the adj_matrices
                        for k2, count in adj_dict.items():
                            adj_matrices[adj][noun][index[k2]] = count

                        # Initialize and populate noun_matrices
                        if noun not in noun_matrices:
                            noun_matrices[noun] = {}
                        if adj not in noun_matrices[noun]:
                            noun_matrices[noun][adj] = [0.1] * len(index)
                        
                        # Since adj_dict might already have counts, use the same to populate noun_matrices
                        for k2, count in adj_dict.items():
                            noun_matrices[noun][adj][index[k2]] = count
            print("and done!")
            normalize = False
            if normalize:
                normalized_matrices = {}
                for adj, noun_vecs in adj_matrices.items():
                    # Initialize the dictionary for this adjective if not already present
                    if adj not in normalized_matrices:
                        normalized_matrices[adj] = {}
                    
                    for noun, vec in noun_vecs.items():
                        # Calculate the norm of the vector
                        norm = math.sqrt(sum(y ** 2 for y in vec))
                        # Normalize the vector if the norm is not zero
                        if norm != 0:
                            normalized_matrices[adj][noun] = [x / norm for x in vec]
                        else:
                            # Handle the case where the vector norm is zero (to avoid division by zero)
                            normalized_matrices[adj][noun] = vec  # Optionally, you could handle it differently, like setting to vec or [0]*len(vec)

                # Now `normalized_matrices` contains the normalized vectorscs[k]]


                with open("data/experiment/semantic_space/AN_Observed_Vectors.json", "w") as outj:
                    json.dump(normalized_matrices, outj, indent=4, ensure_ascii=False)
            
            adj_matrices = apply_svd_to_matrices(adj_matrices)
            noun_matrices = apply_svd_to_matrices(noun_matrices)

            # Save the transformed n_matrices to JSON
            with open("data/experiment/semantic_space/adj_higher_order.json", "w") as outj:
                json.dump(adj_matrices, outj, indent=4, ensure_ascii=False)
            with open("data/experiment/semantic_space/noun_higher_order.json", "w") as outj:
                json.dump(noun_matrices, outj, indent=4, ensure_ascii=False)
    

    
    def cosine_similarity(self, vec1, vec2):
        """Compute the cosine similarity between two vectors."""
        dot_product = np.dot(vec1, vec2)
        norm_a = np.linalg.norm(vec1)
        norm_b = np.linalg.norm(vec2)
        return dot_product / (norm_a * norm_b)

    def find_nearest_neighbors(self, word_vec_dict, target_vector, n):
        """Find the n nearest neighbors of target_vector in word_vec_dict based on cosine similarity."""
        target_vector = word_vec_dict[target_vector]
        similarities = {}
        # Calculate similarity between target vector and each vector in the dictionary
        for word, vector in word_vec_dict.items():
            similarity = self.cosine_similarity(target_vector, vector)
            similarities[word] = similarity
        
        # Sort by similarity descending
        sorted_similarities = sorted(similarities.items(), key=lambda item: item[1], reverse=True)
        
        # Return the top n neighbors
        return sorted_similarities[:n]


    
    def read_sentences_from_bnc(self, file_path):
        # Load and parse the XML file
        tree = ET.parse(file_path)
        root = tree.getroot()

        # Extract sentences enclosed in <s> tags and yield them one by one
        for sentence in root.findall('.//s'):  # Find all <s> tags, which denote sentences
            current_sentence = []
            for word in sentence.findall('.//w'):  # Find all <w> tags within the sentence
                text = word.text.strip() if word.text else ""
                pos = word.attrib.get('pos', 'N/A')  # Get the part of speech if available
                current_sentence.append((text, pos))
            yield current_sentence  # Yield each sentence after it's fully constructed

    
    def read_sentences_from_ukwac(self, file_path):
        # Load and parse the XML file
        try:
            with open(file_path, 'r', encoding='iso-8859-1') as file:
                tree = ET.parse(file)
            root = tree.getroot()
            if root:
                print(f"Was able to process {file_path}, hurray!")
        except Exception as e:
            print(f"Unable to process {file_path}: {e}")
            return

        # Extract sentences enclosed in <s> tags and yield them one by one
        for sentence in root.findall('.//s'):
            current_sentence = []
            for element in sentence.itertext():
                words = element.strip().split('\n')
                for word in words:
                    if word.strip():
                        parts = word.split('\t')
                        if len(parts) >= 3:
                            text = parts[2].strip()
                            pos = parts[1].strip()
                            current_sentence.append((text, pos))
            yield current_sentence

    
    def find_xml_files(self, root_folder):
        xml_files = []
        # Walk through each directory in the root folder
        for dirpath, dirnames, filenames in os.walk(root_folder):
            # Check each file to see if it ends with '.xml'
            for file in filenames:
                if file.endswith('.xml'):
                    # Construct full path relative to the root folder
                    full_path = os.path.relpath(os.path.join(root_folder, dirpath, file), start=root_folder)
                    xml_files.append(full_path)
        return xml_files

if __name__ == "__main__":

    myintensionality = Intension('Texts', initial=False, first = 'first_pass.json', second = None, third='third_pass.json', index="index.json")
    #myintensionality.find_phrasal_vectors(('data/experiment/my_adj_candidates.json', 'data/experiment/my_nn_candidates.json'), ('UKWAC_clean',), 'data/second_pass.json', previous = ('vectors/final_AN_vectors_V1.json', 'vectors/final_AN_matrices_V1.json'))
    myintensionality.build_sparce_vecs('data/bnc/index.json', matrices = 'vectors/final_AN_matrices.json') # vectors = 'vectors/final_AN_vectors.json', 
    """mytarget = random.choice(list(myintensionality.third.keys()))
    print(f"My target is {mytarget}")
    result = myintensionality.find_nearest_neighbors(myintensionality.third, mytarget, 5)
    print(result)"""