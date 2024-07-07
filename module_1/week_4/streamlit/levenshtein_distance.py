import streamlit as st


def calculate_distance(token1, token2, distance, t1, t2):
    if token1[t1-1] == token2[t2-1]:
        return distance[t1-1][t2-1]
    else:
        a = distance[t1-1][t2-1]
        b = distance[t1][t2-1]
        c = distance[t1-1][t2]

        if a <= b and a <= c:
            return a + 1
        elif b <= a and b <= c:
            return b + 1
        else:
            return c + 1


def levenshtein_distance(token1, token2):
    distance = [[0]*(len(token2)+1) for _ in range(len(token1)+1)]

    for t1 in range(len(token1)+1):
        distance[t1][0] = t1

    for t2 in range(len(token2)+1):
        distance[0][t2] = t2

    for t1 in range(1, len(token1)+1):
        for t2 in range(1, len(token2)+1):
            distance[t1][t2] = calculate_distance(
                token1, token2, distance, t1, t2)

    return distance[len(token1)][len(token2)]


def load_vocab(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    words = sorted(set([line.strip().lower() for line in lines]))
    return words


def main():
    st.title('Word Correction using Levenshtein Distance')
    word = st.text_input('Enter a word: ')
    vocabs = load_vocab('vocab.txt')

    if st.button('Compute'):

        # Compute the distance between the word and the vocabularies
        distances = dict()
        for vocab in vocabs:
            distance = levenshtein_distance(word, vocab)
            distances[vocab] = distance

        # Sorted by distance
        sorted_distances = dict(
            sorted(distances.items(), key=lambda item: item[1]))
        correct_word = list(sorted_distances.keys())[0]
        st.write('Correct word: ', correct_word)

        col1, col2 = st.columns(2)
        col1.write('Vocabularies')
        col1.write(vocabs)

        col2.write('Distances')
        col2.write(sorted_distances)


if __name__ == '__main__':
    main()
