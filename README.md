# Genetic Algorithm - Monoalphabetic Cipher Decryption

A **monoalphabetic cipher** is a substitution cipher where each letter is replaced by another letter. For example, the Atbash cipher is an ancient monoalphabetic cipher in which the letter 'A' is substituted with 'Z', 'B' with 'Y', and so on. Essentially, a permutation of the alphabet substitutes the actual letters. In this exercise, you'll be tasked with developing a genetic algorithm to decrypt text encoded with a monoalphabetic cipher.

In the provided model, you'll find a file named `enc.txt` containing a text segment where words have been encrypted using a monoalphabetic cipher. Your goal is to write a genetic algorithm to break the cipher and decrypt the text.

Your program should generate two files: `plain.txt`, containing the decrypted text, and `perm.txt`, containing the permutation table. Additionally, to facilitate comparison between different solutions, your program should print the number of steps (i.e., the total calls to the fitness function) it takes to halt.

## Part A: Genetic Algorithm Implementation

Write a genetic algorithm to solve the problem and decrypt the cipher. The program should produce two files: `plain.txt` with the decrypted text and `perm.txt` with the permutation table. Also, print the number of steps taken by the program until termination to enable comparison between different solutions.

## Part B: Darwinian and Lamarckian Variants

Implement two variations of the genetic algorithm:

1. **Darwinian Version:** After generating a solution using the genetic algorithm, apply a small number of relatively local optimizations. For example, randomly select N pairs of letters in the code table, swap their values, and accept the swap if it improves the fitness. Experiment with different values of N, and include the fitness function calls in the runtime calculations.

2. **Lamarckian Version:** Similar to the Darwinian version, but pass the genome after local optimization to the next generation. The fitness function should be applied after optimization. Include fitness function calls in the runtime calculations.

Consider experimenting with various values for N and measure the runtime performance of the regular, Darwinian, and Lamarckian algorithms.
