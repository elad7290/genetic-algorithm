import copy
import string
import collections
import random
import numpy as np
import matplotlib.pyplot as plt

# hyper parameters
TOURNAMENT_SIZE = 8
BEST_KEY_DUPLICATION = 0.05
N = 1


# Generate an initial population of random keys
def _generateRandomKey():
    alphabet = list(string.ascii_lowercase)
    random.shuffle(alphabet)
    return ''.join(alphabet)


def showGraph(y1, y2, x):
    plt.plot(x, y1, color='blue', label='avg')
    plt.plot(x, y2, color='red', label='best')
    plt.xlabel('fitness value')
    plt.ylabel('iterations')
    plt.legend()
    plt.show()


class GA:
    fitness_counter = 0

    def __init__(self, mutation_rate, lemark=False):
        self.ciphertext = None
        self.expected_frequency = None
        self.expected_bigram_frequency = None
        self.word_frequencies = None
        self.population = None
        self.mutation_rate = mutation_rate
        self.lemark = lemark

    def initialPopulation(self, population_size):
        population = [_generateRandomKey() for _ in range(population_size)]
        self.population = population
        return population

    def setFitnessParameters(self, ciphertext, letter_frequency, bigram_frequency, common_words):
        self.ciphertext = ciphertext
        self.letter_frequency = letter_frequency
        self.bigram_frequency = bigram_frequency
        self.common_words = common_words

    def fitness(self, key):
        self.fitness_counter += 1
        decrypted_text = self.ciphertext.translate(str.maketrans(key, string.ascii_lowercase))
        fitness = 0
        # Calculate the frequency distribution of letters in the decrypted text
        decrypted_letter_count = collections.Counter(decrypted_text.lower())
        total_letters = sum(decrypted_letter_count.values())
        decrypted_letter_frequencies = {letter: decrypted_letter_count[letter] / total_letters for letter in self.letter_frequency}
        value = sum(abs(decrypted_letter_frequencies[letter] - self.letter_frequency[letter]) for letter in self.letter_frequency)
        fitness += value / len(self.letter_frequency)
        # Calculate the frequency distribution of two letters in the decrypted text
        decrypted_bigram_count = collections.Counter([decrypted_text[i:i + 2].lower() for i in range(len(decrypted_text) - 1)
                                                      if len(decrypted_text[i:i + 2].replace(" ", "").replace(".", "").replace(",", "").replace(";", "").replace("\n", "")) == 2])
        total_bigrams = sum(decrypted_bigram_count.values())
        decrypted_bigram_frequencies = {bigram: decrypted_bigram_count[bigram] / total_bigrams for bigram in self.bigram_frequency}
        value = sum(abs(decrypted_bigram_frequencies[bigram] - self.bigram_frequency[bigram]) for bigram in self.bigram_frequency)
        fitness += value / len(self.bigram_frequency)
        # Calculate the presence of common words in the decrypted text
        decrypted_words = decrypted_text.lower().split()
        value = sum(1 for word in decrypted_words if word not in self.common_words)
        fitness += value / len(self.common_words)
        return fitness

    def _crossover(self, parent1, parent2):
        crossover_point = random.randint(1, len(parent1) - 1)
        child = parent1[:crossover_point] + parent2[crossover_point:]
        return child

    def _selectParent(self, fitness_scores):
        #need to change -> maybe we dont need the for loop at all ?
        for _ in range(len(self.population)):
            tournament_indices = random.sample(range(len(self.population)), TOURNAMENT_SIZE)
            tournament_scores = [fitness_scores[i] for i in tournament_indices]
            best_index = tournament_indices[tournament_scores.index(min(tournament_scores))]
        return self.population[best_index]

    def _nextGeneration(self, best_key, fitness_scores):
        next_generation = [best_key for _ in range(int(round(len(self.population)*BEST_KEY_DUPLICATION)))]
        population_left = len(self.population) - len(next_generation)
        for _ in range(population_left):
            parent1 = self._selectParent(fitness_scores)
            parent2 = self._selectParent(fitness_scores)
            child = self._crossover(parent1, parent2)
            next_generation.append(child)
        return next_generation

    def _mutation(self, next_generation):
        k = int(round(self.mutation_rate * len(self.population)))
        indices_for_mutation = random.choices(range(0, len(self.population)), k=k)
        for i in indices_for_mutation:
            mutated_key = list(next_generation[i])
            swap_indices = random.sample(range(26), 2)
            mutated_key[swap_indices[0]], mutated_key[swap_indices[1]] = mutated_key[swap_indices[1]], mutated_key[
                swap_indices[0]]
            next_generation[i] = ''.join(mutated_key)
        return next_generation

    def _repair(self, next_generation):
        alphabet = set(string.ascii_lowercase)
        for i in range(len(self.population)):
            while len(set(next_generation[i])) != len(next_generation[i]):
                mutated_key = list(next_generation[i])
                duplicates = [letter for letter, count in collections.Counter(mutated_key).items() if count > 1]
                missings = list(alphabet - set(mutated_key))
                random.shuffle(missings)
                for j in range(len(duplicates)):
                    indices = [k for k, letter in enumerate(mutated_key) if letter == duplicates[j]]
                    mutated_key[indices[0]] = missings[0]
                next_generation[i] = ''.join(mutated_key)
        return next_generation

    def _isImproved(self, best_solutions, num):
        if len(best_solutions) < num:
            return True
        for i in reversed(range(len(best_solutions)-10, len(best_solutions))):
            if best_solutions[i] < best_solutions[i-1]:
                return True
        return False

    def run(self, max_generations):
        avg_solutions = []
        best_solutions = []

        min_fitness = float('inf')
        min_key = ''
        min_dec = ""

        # Perform the genetic algorithm
        for generation in range(max_generations):
            # Calculate fitness for each key in the population
            fitness_scores = [self.fitness(key) for key in self.population]
            avg = np.mean(fitness_scores)
            avg_solutions.append(avg)
            best_fitness = min(fitness_scores)
            best_solutions.append(best_fitness)
            best_key = self.population[fitness_scores.index(best_fitness)]
            decrypted_text = self.ciphertext.translate(str.maketrans(best_key, string.ascii_lowercase))
            print("Generation:", generation+1)
            print("Key:", best_key)
            print("Decrypted text:", decrypted_text.replace("\n", " "))

            next_generation = self._nextGeneration(best_key, fitness_scores)
            # Repair invalid children
            next_generation = self._repair(next_generation)
            # Perform mutation on the next generation
            next_generation = self._mutation(next_generation)

            self.population = next_generation

            if best_fitness < min_fitness:
                min_fitness = best_fitness
                min_key = best_key
                min_dec = decrypted_text

            if not self._isImproved(best_solutions, 10):
                break

        showGraph(avg_solutions, best_solutions, range(1,len(best_solutions)+1))
        return min_key, min_dec, self.fitness_counter

    def runSectionB(self, max_generations):
        avg_solutions = []
        best_solutions = []

        min_fitness = float('inf')
        min_key = ''
        min_dec = ""

        # Perform the genetic algorithm
        for generation in range(max_generations):
            # Calculate fitness for each key in the population

            fitness_scores = self._optimization()
            avg = np.mean(fitness_scores)
            avg_solutions.append(avg)
            best_fitness = min(fitness_scores)
            best_key = self.population[fitness_scores.index(best_fitness)]
            decrypted_text = self.ciphertext.translate(str.maketrans(best_key, string.ascii_lowercase))
            best_solutions.append(best_fitness)

            print("Generation:", generation+1)
            print("Key:", best_key)
            print("Decrypted text:", decrypted_text.replace("\n", " "))

            next_generation = self._nextGeneration(best_key, fitness_scores)
            # Repair invalid children
            next_generation = self._repair(next_generation)
            # Perform mutation on the next generation
            next_generation = self._mutation(next_generation)

            self.population = next_generation

            if best_fitness < min_fitness:
                min_fitness = best_fitness
                min_key = best_key
                min_dec = decrypted_text

            if not self._isImproved(best_solutions, 10):
                break

        showGraph(avg_solutions, best_solutions, range(1,len(best_solutions)+1))
        return min_key, min_dec, self.fitness_counter

    def _optimization(self):
        optimized_population = copy.deepcopy(self.population)
        # optimizations
        for j in range(len(optimized_population)):
            mutated_key = list(optimized_population[j])
            for i in range(N):
                swap_indices = random.sample(range(26), 2)
                mutated_key[swap_indices[0]], mutated_key[swap_indices[1]] = mutated_key[swap_indices[1]], mutated_key[
                    swap_indices[0]]
            optimized_population[j] = ''.join(mutated_key)
        # find best
        fitness_scores = [self.fitness(key) for key in optimized_population]
        # lemark or darvin
        if self.lemark:
            self.population = optimized_population
        return fitness_scores




