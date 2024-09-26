import os
import math
from collections import defaultdict

# Preprocessing function
def preprocess(text):
    """ Tokenizes and normalizes the input text. """
    return text.lower().split()  # Can be enhanced with more advanced techniques

# Build dictionary and postings list
def build_index(corpus):
    """ Builds the index for the given corpus. """
    dictionary = defaultdict(int)
    postings = defaultdict(list)
    doc_lengths = {}

    for doc_id, text in corpus.items():
        terms = preprocess(text)
        term_freqs = defaultdict(int)

        for term in terms:
            term_freqs[term] += 1

        for term, freq in term_freqs.items():
            dictionary[term] += 1  # Document frequency
            postings[term].append((doc_id, freq))  # (document_id, term_frequency)

        # Calculate document length for normalization
        doc_lengths[doc_id] = math.sqrt(sum((1 + math.log10(freq)) ** 2 for freq in term_freqs.values()))

    return dictionary, postings, doc_lengths

# Calculate tf-idf
def calculate_tf_idf(term, freq, df, N):
    """ Computes the TF-IDF score for a term. """
    tf = 1 + math.log10(freq)
    idf = math.log10(N / df) if df > 0 else 0  # Handle division by zero
    return tf * idf

# Cosine similarity
def cosine_similarity(query_vector, doc_vector, doc_length):
    """ Computes the cosine similarity between query and document vectors. """
    dot_product = sum(query_vector[term] * doc_vector.get(term, 0) for term in query_vector)
    query_length = math.sqrt(sum(weight ** 2 for weight in query_vector.values()))
    return dot_product / (query_length * doc_length) if query_length > 0 and doc_length > 0 else 0

# Search function
def search(query, dictionary, postings, doc_lengths, N):
    """ Searches for the query in the index and ranks documents by relevance. """
    query_terms = preprocess(query)
    query_vector = {term: calculate_tf_idf(term, query_terms.count(term), dictionary[term], N) for term in query_terms}

    scores = defaultdict(float)
    for term, weight in query_vector.items():
        for doc_id, freq in postings.get(term, []):
            # Calculate TF-IDF for the document term
            doc_tf_idf = calculate_tf_idf(term, freq, dictionary[term], N)
            scores[doc_id] += weight * doc_tf_idf

    # Normalize scores by document lengths
    for doc_id in scores:
        scores[doc_id] /= doc_lengths.get(doc_id, 1)

    ranked_docs = sorted(scores.items(), key=lambda x: (-x[1], x[0]))[:10]
    return ranked_docs

# Load documents from the corpus folder
def load_corpus(corpus_folder):
    """ Loads text documents from the specified folder. """
    corpus = {}
    for filename in os.listdir(corpus_folder):
        if filename.endswith('.txt'):
            try:
                with open(os.path.join(corpus_folder, filename), 'r', encoding='utf-8') as file:
                    corpus[filename] = file.read()
            except Exception as e:
                print(f"Error reading {filename}: {e}")
    return corpus

# Example usage
if __name__ == "__main__":
# if __name__ == "__main__":
    # Use relative path to the Corpus folder
    corpus_folder = os.path.join(os.getcwd(), 'Corpus')
    corpus = load_corpus(corpus_folder)

    dictionary, postings, doc_lengths = build_index(corpus)
    N = len(corpus)

    while True:
        # Ask user for query
        query = input("Enter your search query (or type 'exit' to quit): ")

        if query.strip().lower() == 'exit':  # Check for exit condition
            print("Exiting the search.")
            break

        if not query.strip():  # Basic input validation
            print("Query cannot be empty.")
        else:
            results = search(query, dictionary, postings, doc_lengths, N)

            # Print results
            if results:
                for doc, score in results:
                    print(f"{doc}: {score:.4f}")
            else:
                print("No results found.")
