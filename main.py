from GA import GA
from files import read_letters, read_words, read_enc_file, write_file, write_key

# Define the genetic algorithm parameters
population_size = 200
mutation_rate = 0.3
max_generations = 200

ciphertext = read_enc_file('enc.txt')
letter_frequency = read_letters('Letter_Freq.txt')
bigram_frequency = read_letters('Letter2_Freq.txt')
common_words = read_words('dict.txt')

ga = GA(mutation_rate=mutation_rate)    # add lemark=True or lemark=False
population = ga.initialPopulation(population_size)
ga.setFitnessParameters(ciphertext=ciphertext, letter_frequency=letter_frequency,
        bigram_frequency=bigram_frequency, common_words=common_words)
key, dec, steps = ga.runSectionB(max_generations) # change here to run other section

write_file('plain.txt', dec)
write_key('perm.txt', key)
print("number of steps: ", steps)