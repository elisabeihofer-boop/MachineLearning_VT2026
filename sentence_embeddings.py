'''create sentence embeddings (from word embedding means) for each .tsv file'''

# importing
from train_fasttext import tokenizer
import argparse
import pandas as pd
import numpy as np
from gensim.models import FastText

def sentence_vectors(sentence, model, dim):
    '''takes a sentence and returns the mean of the vectors of its tokens'''

    # getting a list of tokens
    tokens = tokenizer(sentence)

    vectors = []

    # checking if the token is in the model => appending its vector value to a list 
    for token in tokens:
        if token in model.wv:
            vectors.append(model.wv[token])
    
    # handling unkown tokens (returning netural zero vector) 
    if len(vectors) == 0:
        return np.zeros(dim)
    
    return np.mean(vectors, axis=0)

def main():
    # adding the arguments
    parser = argparse.ArgumentParser(description="Create sentence embeddings")

    # input file
    parser.add_argument("--input_file", required=True, help="TSV input file")

    # model output path
    parser.add_argument("--output_path", type=str, required=True, help="specify the output path for the model")

    # specify output file
    parser.add_argument("--output_file", required=True, help="specify the name of the output file") 

    args = parser.parse_args()

    # loading the trained FastText model
    print("* Loading model *")
    model = FastText.load(args.output_path)
    dim = model.vector_size

    # access vocabulary and vectors
    #vocab = model.wv.key_to_index # keyed vectors (dict with words and their indices)

    #reading the input file
    print("* Reading data *")
    df = pd.read_csv(args.input_file, sep="\t")

    # creating a list with the sentence vector embeddings
    print("* Generating sentence embeddings *")
    embeddings = []

    for sentence in df["text"]:
        vector_embd = sentence_vectors(sentence, model, dim)
        embeddings.append(vector_embd)

    # saving the embeddings as a numpy array
    print(f"* Saving vector embeddings to {args.output_file} *")
    embeddings = np.array(embeddings)
    np.save(args.output_file, embeddings)

    print("All done!")

if __name__ == "__main__":
    main()
