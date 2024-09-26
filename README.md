IR_Assignment2
##Overview
Our project implements a simple Information Retrieval (IR) system that builds an inverted index from a collection of text documents, allowing users to perform keyword-based searches. The system ranks documents based on their relevance to the search query using the TF-IDF (Term Frequency-Inverse Document Frequency) metric and cosine similarity.

##Features
1.Preprocessing: Normalizes and tokenizes input text.
2.Index Building: Constructs a dictionary and postings list for efficient retrieval.
3.TF-IDF Calculation: Computes TF-IDF scores for terms in both documents and queries.
4.Cosine Similarity: Measures the similarity between query and document vectors.
5.Search Functionality: Allows users to input search queries and retrieves the top relevant documents.

##Requirements
Python 3.x
os, math, and collections modules (included in standard library)

##Project Structure
/ir_assignment2
│
├── code.py                
├── Corpus/                
│   ├── adobe.txt
│   ├── apple.txt     
│   └── ...          
└── README.md        

##Installation
1.Clone the repository or download the project files.
2.Ensure you have Python installed on your machine.
3.Run the application:
python code.py
Input your search query when prompted. Type 'exit' to quit the application.

Example
After running the program, you can enter a query like:

Enter your search query (or type 'exit' to quit): machine learning
The program will output relevant documents and their corresponding relevance scores.
