import random
from string import punctuation
import math
from collections import OrderedDict
import re

# TRAIN AND TEST FILES
hamilton_train_files = ['1.txt', '6.txt', '7.txt', '8.txt', '13.txt', '15.txt', '16.txt', '17.txt', '21.txt',
                        '22.txt', '23.txt', '24.txt', '25.txt', '26.txt', '27.txt', '28.txt', '29.txt']
hamilton_test_files = ['9.txt', '11.txt', '12.txt']
madison_train_files = ['10.txt', '14.txt', '37.txt', '38.txt', '39.txt', '40.txt', '41.txt', '42.txt',
                       '43.txt', '44.txt', '45.txt', '46.txt']
madison_test_files = ['47.txt', '48.txt', '58.txt']
unknown_test_files = ['49.txt', '50.txt', '51.txt', '52.txt', '53.txt', '54.txt', '55.txt', '56.txt',
                      '57.txt', '62.txt', '63.txt']


# REMOVE PUNCTUATIONS
def strip_punctuation(s):
    return ''.join(c for c in s if c not in punctuation)


# GENEREATE N-GRAMS ACCORDING TO N
def generate_ngrams(s, n):
    # Convert to lowercases
    s = s.lower()

    # Replace all none alphanumeric characters with spaces
    s = strip_punctuation(s)

    # Break sentence in the token, remove empty tokens
    tokens = [token for token in s.split(" ") if token != ""]

    # Use the zip function to help us generate n-grams
    # Concatentate the tokens into ngrams and return
    ngrams = zip(*[tokens[i:] for i in range(n)])
    return [" ".join(ngram) for ngram in ngrams]


# READ TEST FILES AND STORE THEIR SENTENCES
def file_sentence(filename):
    file = open("../data/" + filename, "r")

    lines = []
    for line in file:
        line = re.split("r'(((?<=[a-z0-9][.?!])|(?<=[a-z0-9][.?!]\"))(\s|\r\n)(?=\"?[A-Z]))'", line)
        for sentence in line:
            sentence = sentence.lower()
            sentence = strip_punctuation(sentence)
            liner = [token for token in sentence.split(" ") if token != ""]
            lines.append(liner)

    return lines[1:]


# READ TRAIN FILES AND BUILD N-GRAMS
def read_and_build_ngram(file_list, ngram):
    result = []
    for files in file_list:
        file = open("../data/" + files, "r")

        lines = []
        if ngram == 2 or ngram == 3:
            for line in file:
                line = re.split("r'(((?<=[a-z0-9][.?!])|(?<=[a-z0-9][.?!]\"))(\s|\r\n)(?=\"?[A-Z]))'", line)
                for sentence in line:
                    unigram = generate_ngrams(sentence, ngram)
                    lines.append(unigram)

        if ngram == 1:
            for line in file:
                unigram = generate_ngrams(line[:-1], ngram)
                lines.append(unigram)

        for sentences in lines:
            for sentence in sentences:
                result.append(sentence)
    return result


# CREATE DICTIONARY FOR UNIQUE WORDS OR WORD PAIRS, STORED THEIR FREQUENCIES
def create_dict(ngram_list):
    result = {}

    for item in ngram_list:
        result[item] = result.get(item, 0) + 1

    return result


# FIND THE TOTAL FREQUENCIES OF WORDS OR WORD PAIRS IN N-GRAM
def get_ngram_frequency(ngram_dict):
    result = 0

    for word, frequency in ngram_dict.items():
        result += frequency

    return result


# CALCULATE THE PROBABILITY OF UNIQUE WORDS FOR UNIGRAM ACCORDING TO THEIR FREQUENCIES
def calculate_unigram_probability_unsmoothed(unigram_dict):
    result = {}

    for word, frequency in unigram_dict.items():
        prob = frequency / get_ngram_frequency(unigram_dict)
        result[word] = prob

    return result


# CALCULATE THE PROBABILITY OF ONE SENTENCE FOR UNIGRAM
def calculate_unigram_sentence_probability(sentence, unigram):
    sentence_probability_log_sum = 0
    for word in sentence:
        word_probability = float(unigram.get(word))
        sentence_probability_log_sum += math.log(word_probability, 2)
    return sentence_probability_log_sum


# CALCULATE THE PROBABILITY OF UNIQUE WORD PAIRS FOR BIGRAM ACCORDING TO THEIR FREQUENCIES
def calculate_bigram_probability_unsmoothed(bigram_dict, unigram_dict):
    result = {}

    for word, frequency in bigram_dict.items():
        words = word.split()
        prob = frequency / int(unigram_dict[words[0]])
        result[word] = prob

    return result


# CALCULATE THE PROBABILITY OF ONE SENTENCE FOR BIGRAM
def calculate_bigram_sentence_probability(sentence, bigram, unigram_freq, unigram_prob):
    bigram_sentence_probability_log_sum = 0
    for i in range(len(sentence)):
        if i == 0:
            if sentence[i] not in unigram_prob.keys():
                bigram_word_probability = 1 / (len(unigram_freq) + len(unigram_freq))
            else:
                bigram_word_probability = float(unigram_prob.get(sentence[i]))
        else:
            if sentence[i - 1] + " " + sentence[i] not in bigram.keys():
                bigram_word_probability = 1 / (int(unigram_freq.get(sentence[i - 1], 0)) + len(unigram_freq))
            else:
                bigram_word_probability = float(bigram.get(sentence[i - 1] + " " + sentence[i]))
        bigram_sentence_probability_log_sum += math.log(bigram_word_probability, 2)
    return bigram_sentence_probability_log_sum


# CALCULATE THE PERPLEXITY OF SENTENCES FOR BIGRAM
def calculate_bigram_perplexity(essay, bigram, unigram_freq, unigram_prob):
    bigram_sentence_probability_log_sum = 0
    count = 0
    for sentence in essay:
        count += len(sentence)
        bigram_sentence_probability_log_sum -= calculate_bigram_sentence_probability(sentence, bigram,
                                                                                     unigram_freq, unigram_prob)
    return math.pow(2, (bigram_sentence_probability_log_sum / count))


# CALCULATE THE PROBABILITY OF UNIQUE WORD PAIRS FOR TRIGRAM ACCORDING TO THEIR FREQUENCIES
def calculate_trigram_probability_unsmoothed(trigram_dict, bigram_dict):
    result = {}

    for word, frequency in trigram_dict.items():
        words = word.split()
        prob = frequency / int(bigram_dict[words[0] + " " + words[1]])
        result[word] = prob

    return result


# CALCULATE THE PROBABILITY OF ONE SENTENCE FOR TRIGRAM
def calculate_trigram_sentence_probability(sentence, trigram, bigram_freq, bigram_prob, unigram_freq, unigram_prob):
    trigram_sentence_probability_log_sum = 0
    for i in range(len(sentence)):
        if i == 0:
            if sentence[i] not in unigram_prob.keys():
                trigram_word_probability = 1 / (get_ngram_frequency(unigram_freq) + len(unigram_prob))
            else:
                trigram_word_probability = float(unigram_prob.get(sentence[i]))
        if i == 1:
            if sentence[i - 1] + " " + sentence[i] not in bigram_prob.keys():
                trigram_word_probability = 1 / (int(unigram_freq.get(sentence[i - 1], 0)) + len(unigram_freq))
            else:
                trigram_word_probability = float(bigram_prob.get(sentence[i - 1] + " " + sentence[i]))
        else:
            if sentence[i - 2] + " " + sentence[i - 1] + " " + sentence[i] not in trigram.keys():
                trigram_word_probability = 1 / (int(bigram_freq.get(sentence[i - 2] + " " + sentence[i - 1], 0)) +
                                                len(bigram_freq))
            else:
                trigram_word_probability = float(trigram.get(sentence[i - 2] + " " + sentence[i - 1] + " " +
                                                             sentence[i]))
        trigram_sentence_probability_log_sum += math.log(trigram_word_probability, 2)
    return trigram_sentence_probability_log_sum


# CALCULATE THE PERPLEXITY OF SENTENCES FOR TRIGRAM
def calculate_trigram_perplexity(essay, trigram, bigram_freq, bigram_prob, unigram_freq, unigram_prob):
    trigram_sentence_probability_log_sum = 0
    count = 0
    for sentence in essay:
        count += len(sentence)
        trigram_sentence_probability_log_sum -= calculate_trigram_sentence_probability(sentence, trigram,
                                                                                       bigram_freq, bigram_prob,
                                                                                       unigram_freq, unigram_prob)
    return math.pow(2, (trigram_sentence_probability_log_sum / count))


# CHOICE WORD OR WORD PAIR FOR GENERATE RANDOMLY SENTENCE
def weighted_choice(choices):
   total = sum(float(probability) for word, probability in choices.items())
   r = random.uniform(0, total)
   upto = 0
   for word, probability in choices.items():
      if upto + float(probability) > r:
         return word
      upto += float(probability)


# GENERATE RANDOM SENTENCE ACCORDING TO UNIGRAM
def create_unigram_sentence(unigram, n):
    result = []
    for i in range(n):
        result.append(weighted_choice(OrderedDict(sorted(unigram.items(), key=lambda x: x[1]))))
    return result


# GENERATE RANDOM SENTENCE ACCORDING TO BIGRAM
def create_bigram_sentence(word, ngram, n):
    result = []
    result.append(word)
    for i in range(n):
        # Get all possible elements ((first word, second word), frequency)
        choices = dict()
        for element, probability in ngram.items():
            if element.split(" ")[0] == word:
                choices.update({element: probability})
        if not choices:
            break

        # Choose a pair with weighted probability from the choice list
        word = weighted_choice(OrderedDict(sorted(choices.items(), key=lambda x: x[1]))).split(" ")[1]
        result.append(word)
    return result


# GENERATE RANDOM SENTENCE ACCORDING TO TRIGRAM
def create_trigram_sentence(word, nextword, ngram, n):
    result = []
    result.append(word)
    result.append(nextword)
    for i in range(n):
        # Get all possible elements ((first word, second word), frequency)
        choices = dict()
        for element, probability in ngram.items():
            if element.split(" ")[0] == word and element.split(" ")[1] == nextword:
                choices.update({element: probability})
        if not choices:
            break

        # Choose a pair with weighted probability from the choice list
        wordpair = weighted_choice(OrderedDict(sorted(choices.items(), key=lambda x: x[1])))
        word = wordpair.split(" ")[1]
        nextword = wordpair.split(" ")[2]
        result.append(nextword)
    return result


# PRINT THE WORDS OF GIVEN SENTENCE
def print_sentence(sentence):
    for word in sentence:
        print(word, end=" ")
    print()


# TASK 1 - BUILD N-GRAM LANGUAGE MODELS

unigram_hamilton_list = read_and_build_ngram(hamilton_train_files, 1)
bigram_hamilton_list = read_and_build_ngram(hamilton_train_files, 2)
trigram_hamilton_list = read_and_build_ngram(hamilton_train_files, 3)

unigram_madison_list = read_and_build_ngram(madison_train_files, 1)
bigram_madison_list = read_and_build_ngram(madison_train_files, 2)
trigram_madison_list = read_and_build_ngram(madison_train_files, 3)

unigram_hamilton = create_dict(unigram_hamilton_list)
unigram_hamilton_probability_unsmoothed = calculate_unigram_probability_unsmoothed(unigram_hamilton)

unigram_madison = create_dict(unigram_madison_list)
unigram_madison_probability_unsmoothed = calculate_unigram_probability_unsmoothed(unigram_madison)

bigram_hamilton = create_dict(bigram_hamilton_list)
bigram_hamilton_probability_unsmoothed = calculate_bigram_probability_unsmoothed(bigram_hamilton, unigram_hamilton)

bigram_madison = create_dict(bigram_madison_list)
bigram_madison_probability_unsmoothed = calculate_bigram_probability_unsmoothed(bigram_madison, unigram_madison)

trigram_hamilton = create_dict(trigram_hamilton_list)
trigram_hamilton_probability_unsmoothed = calculate_trigram_probability_unsmoothed(trigram_hamilton, bigram_hamilton)

trigram_madison = create_dict(trigram_madison_list)
trigram_madison_probability_unsmoothed = calculate_trigram_probability_unsmoothed(trigram_madison, bigram_madison)


# TASK 2 - GENERATE SENTENCES ACCORDING TO UNIGRAM
print("*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*\n")
print("GENERATING RANDOM SENTENCES ACCORDING TO UNIGRAM")
print("\n*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*\n")
print("Sentence1: (Unigram Hamilton)")
sentence1 = create_unigram_sentence(unigram_hamilton_probability_unsmoothed, 30)
print_sentence(sentence1)
print("Probability of sentence:\t", calculate_unigram_sentence_probability(sentence1,
                                                                           unigram_hamilton_probability_unsmoothed))

print("Sentence2: (Unigram Hamilton)")
sentence2 = create_unigram_sentence(unigram_hamilton_probability_unsmoothed, 30)
print_sentence(sentence2)
print("Probability of sentence:\t", calculate_unigram_sentence_probability(sentence2,
                                                                           unigram_hamilton_probability_unsmoothed))

print("Sentence3: (Unigram Madison)")
sentence3 = create_unigram_sentence(unigram_madison_probability_unsmoothed, 30)
print_sentence(sentence3)
print("Probability of sentence:\t", calculate_unigram_sentence_probability(sentence3,
                                                                           unigram_madison_probability_unsmoothed))

print("Sentence4: (Unigram Madison)")
sentence4 = create_unigram_sentence(unigram_madison_probability_unsmoothed, 30)
print_sentence(sentence4)
print("Probability of sentence:\t", calculate_unigram_sentence_probability(sentence4,
                                                                           unigram_madison_probability_unsmoothed))


# TASK 2 - GENERATE SENTENCES ACCORDING TO BIGRAM
print("\n*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*\n")
print("GENERATING RANDOM SENTENCES ACCORDING TO BIGRAM")
print("\n*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*\n")
print("Sentence5: (Bigram Hamilton)")
sentence5 = create_bigram_sentence(weighted_choice(unigram_hamilton_probability_unsmoothed),
                                   bigram_hamilton_probability_unsmoothed, 29)
print_sentence(sentence5)
print("Probability of sentence:\t", calculate_bigram_sentence_probability(sentence5,
                                                                          bigram_hamilton_probability_unsmoothed,
                                                                          unigram_hamilton,
                                                                          unigram_hamilton_probability_unsmoothed))

print("Sentence6: (Bigram Hamilton)")
sentence6 = create_bigram_sentence(weighted_choice(unigram_hamilton_probability_unsmoothed),
                                   bigram_hamilton_probability_unsmoothed, 29)
print_sentence(sentence6)
print("Probability of sentence:\t", calculate_bigram_sentence_probability(sentence6,
                                                                          bigram_hamilton_probability_unsmoothed,
                                                                          unigram_hamilton,
                                                                          unigram_hamilton_probability_unsmoothed))

print("Sentence7: (Bigram Madison)")
sentence7 = create_bigram_sentence(weighted_choice(unigram_madison_probability_unsmoothed),
                                   bigram_madison_probability_unsmoothed, 29)
print_sentence(sentence7)
print("Probability of sentence:\t", calculate_bigram_sentence_probability(sentence7,
                                                                          bigram_madison_probability_unsmoothed,
                                                                          unigram_madison,
                                                                          unigram_madison_probability_unsmoothed))


print("Sentence8: (Bigram Madison)")
sentence8 = create_bigram_sentence(weighted_choice(unigram_madison_probability_unsmoothed),
                                   bigram_madison_probability_unsmoothed, 29)
print_sentence(sentence8)
print("Probability of sentence:\t", calculate_bigram_sentence_probability(sentence8,
                                                                          bigram_madison_probability_unsmoothed,
                                                                          unigram_madison,
                                                                          unigram_madison_probability_unsmoothed))


# TASK 2 - GENERATE ESSAYS ACCORDING TO TRIGRAM
print("\n*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*\n")
print("GENERATING RANDOM SENTENCES ACCORDING TO TRIGRAM")
print("\n*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*\n")
print("Sentence9: (Trigram Hamilton)")
start_word9 = weighted_choice(unigram_hamilton_probability_unsmoothed)
second_word9 = create_bigram_sentence(start_word9, bigram_hamilton_probability_unsmoothed, 1)[1]
sentence9 = create_trigram_sentence(start_word9, second_word9, trigram_hamilton_probability_unsmoothed, 28)
print_sentence(sentence9)
print("Probability of sentence:\t", calculate_trigram_sentence_probability(sentence9,
                                                                           trigram_hamilton_probability_unsmoothed,
                                                                           bigram_hamilton,
                                                                           bigram_hamilton_probability_unsmoothed,
                                                                           unigram_hamilton,
                                                                           unigram_hamilton_probability_unsmoothed))

print("Sentence10: (Trigram Hamilton)")
start_word10 = weighted_choice(unigram_hamilton_probability_unsmoothed)
second_word10 = create_bigram_sentence(start_word10, bigram_hamilton_probability_unsmoothed, 1)[1]
sentence10 = create_trigram_sentence(start_word10, second_word10, trigram_hamilton_probability_unsmoothed, 28)
print_sentence(sentence10)
print("Probability of sentence:\t", calculate_trigram_sentence_probability(sentence10,
                                                                           trigram_hamilton_probability_unsmoothed,
                                                                           bigram_hamilton,
                                                                           bigram_hamilton_probability_unsmoothed,
                                                                           unigram_hamilton,
                                                                           unigram_hamilton_probability_unsmoothed))

print("Sentence11: (Trigram Madison)")
start_word11 = weighted_choice(unigram_madison_probability_unsmoothed)
second_word11 = create_bigram_sentence(start_word11, bigram_madison_probability_unsmoothed, 1)[1]
sentence11 = create_trigram_sentence(start_word11, second_word11, trigram_madison_probability_unsmoothed, 28)
print_sentence(sentence11)
print("Probability of sentence:\t", calculate_trigram_sentence_probability(sentence11,
                                                                           trigram_madison_probability_unsmoothed,
                                                                           bigram_madison,
                                                                           bigram_madison_probability_unsmoothed,
                                                                           unigram_madison,
                                                                           unigram_madison_probability_unsmoothed))

print("Sentence12: (Trigram Madison)")
start_word12 = weighted_choice(unigram_madison_probability_unsmoothed)
second_word12 = create_bigram_sentence(start_word12, bigram_madison_probability_unsmoothed, 1)[1]
sentence12 = create_trigram_sentence(start_word12, second_word12, trigram_madison_probability_unsmoothed, 28)
print_sentence(sentence12)
print("Probability of sentence:\t", calculate_trigram_sentence_probability(sentence12,
                                                                           trigram_madison_probability_unsmoothed,
                                                                           bigram_madison,
                                                                           bigram_madison_probability_unsmoothed,
                                                                           unigram_madison,
                                                                           unigram_madison_probability_unsmoothed))


# TASK 3 - CLASSIFICATION AND EVALUATION FROM BIGRAM PERPLEXITY
print("\n*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*\n")
print("TEST FILES ACCORDING TO BIGRAM MODELS")
print("\n*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*\n")
for files in hamilton_test_files:
    unk_sentence = file_sentence(files)
    print(files, "HAMILTON TEST")
    ham_perp = calculate_bigram_perplexity(unk_sentence, bigram_hamilton_probability_unsmoothed, unigram_hamilton,
                                           unigram_hamilton_probability_unsmoothed)
    mad_perp = calculate_bigram_perplexity(unk_sentence, bigram_madison_probability_unsmoothed, unigram_madison,
                                           unigram_madison_probability_unsmoothed)
    print("hamilton perplexity:\t", ham_perp)
    print("madison perplexity:\t", mad_perp)

    if ham_perp < mad_perp:
        print("Author of ", files, " is HAMILTON")
    else:
        print("Author of ", files, " is MADISON")
    print("-----------------")

for files in madison_test_files:
    unk_sentence = file_sentence(files)
    print(files, "MADISON TEST")
    ham_perp = calculate_bigram_perplexity(unk_sentence, bigram_hamilton_probability_unsmoothed, unigram_hamilton,
                                           unigram_hamilton_probability_unsmoothed)
    mad_perp = calculate_bigram_perplexity(unk_sentence, bigram_madison_probability_unsmoothed, unigram_madison,
                                           unigram_madison_probability_unsmoothed)
    print("hamilton perplexity:\t", ham_perp)
    print("madison perplexity:\t", mad_perp)

    if ham_perp < mad_perp:
        print("Author of ", files, " is HAMILTON")
    else:
        print("Author of ", files, " is MADISON")
    print("-----------------")

for files in unknown_test_files:
    unk_sentence = file_sentence(files)
    print(files, "UNKNOWN TEST")
    ham_perp = calculate_bigram_perplexity(unk_sentence, bigram_hamilton_probability_unsmoothed, unigram_hamilton,
                                           unigram_hamilton_probability_unsmoothed)
    mad_perp = calculate_bigram_perplexity(unk_sentence, bigram_madison_probability_unsmoothed, unigram_madison,
                                           unigram_madison_probability_unsmoothed)
    print("hamilton perplexity:\t", ham_perp)
    print("madison perplexity:\t", mad_perp)

    if ham_perp < mad_perp:
        print("Author of ", files, " is HAMILTON")
    else:
        print("Author of ", files, " is MADISON")
    print("-----------------")

# TASK 3 - CLASSIFICATION AND EVALUATION FROM TRIGRAM PERPLEXITY
print("\n*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*\n")
print("TEST FILES ACCORDING TO TRIGRAM MODELS")
print("\n*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*\n")
for files in hamilton_test_files:
    unk_sentence = file_sentence(files)
    print(files, "HAMILTON TEST")
    ham_perp = calculate_trigram_perplexity(unk_sentence, trigram_hamilton_probability_unsmoothed, bigram_hamilton,
                                            bigram_hamilton_probability_unsmoothed, unigram_hamilton,
                                            unigram_hamilton_probability_unsmoothed)
    mad_perp = calculate_trigram_perplexity(unk_sentence, trigram_madison_probability_unsmoothed, bigram_madison,
                                            bigram_madison_probability_unsmoothed, unigram_hamilton,
                                            unigram_madison_probability_unsmoothed)
    print("hamilton perplexity:\t", ham_perp)
    print("madison perplexity:\t", mad_perp)

    if ham_perp < mad_perp:
        print("Author of ", files, " is HAMILTON")
    else:
        print("Author of ", files, " is MADISON")
    print("-----------------")

for files in madison_test_files:
    unk_sentence = file_sentence(files)
    print(files, "MADISON TEST")
    ham_perp = calculate_trigram_perplexity(unk_sentence, trigram_hamilton_probability_unsmoothed, bigram_hamilton,
                                            bigram_hamilton_probability_unsmoothed, unigram_hamilton,
                                            unigram_hamilton_probability_unsmoothed)
    mad_perp = calculate_trigram_perplexity(unk_sentence, trigram_madison_probability_unsmoothed, bigram_madison,
                                            bigram_madison_probability_unsmoothed, unigram_hamilton,
                                            unigram_madison_probability_unsmoothed)
    print("hamilton perplexity:\t", ham_perp)
    print("madison perplexity:\t", mad_perp)

    if ham_perp < mad_perp:
        print("Author of ", files, " is HAMILTON")
    else:
        print("Author of ", files, " is MADISON")
    print("-----------------")

for files in unknown_test_files:
    unk_sentence = file_sentence(files)
    print(files, "UNKNOWN TEST")
    ham_perp = calculate_trigram_perplexity(unk_sentence, trigram_hamilton_probability_unsmoothed, bigram_hamilton,
                                            bigram_hamilton_probability_unsmoothed, unigram_hamilton,
                                            unigram_hamilton_probability_unsmoothed)
    mad_perp = calculate_trigram_perplexity(unk_sentence, trigram_madison_probability_unsmoothed, bigram_madison,
                                            bigram_madison_probability_unsmoothed, unigram_hamilton,
                                            unigram_madison_probability_unsmoothed)
    print("hamilton perplexity:\t", ham_perp)
    print("madison perplexity:\t", mad_perp)

    if ham_perp < mad_perp:
        print("Author of ", files, " is HAMILTON")
    else:
        print("Author of ", files, " is MADISON")
    print("-----------------")
