import string

import numpy as np


def read_letters(fileName):
    dict_data = {}
    with open(fileName, 'r') as file:
        for line in file:
            line = line.strip()  # Remove leading/trailing whitespace
            if line and line != '#REF!':
                value, key = line.split('\t')
                dict_data[key.lower()] = float(value)
    return dict_data


def read_words(fileName):
    array_data = []
    with open(fileName, 'r') as file:
        for line in file:
            line = line.strip()  # Remove leading/trailing whitespace
            if line:
                array_data.append(line)
    return array_data


def read_enc_file(fileName):
    with open(fileName, "r") as file:
        text = ''
        for line in file:
            line = line.strip()  # Remove leading/trailing whitespace
            line = line + '\n'
            text += ''.join(line)
    return text


def write_file(file_name, dec):
    file = open(file_name, 'w')
    file.write(dec)
    file.close()


def write_key(file_name, key):
    mutated_key = np.array(list(key))
    alphabet = np.array(list(string.ascii_lowercase))
    sort = np.argsort(mutated_key)
    mutated_key_sorted = mutated_key[sort]
    alphabet_sorted = alphabet[sort]
    with open(file_name, "w") as file:
        for i in range(len(mutated_key_sorted)):
            line = mutated_key_sorted[i] + '\t' + alphabet_sorted[i] + '\n'
            file.write(line)

