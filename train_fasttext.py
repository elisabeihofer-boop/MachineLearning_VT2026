'''training word (in this case: character) embeddings over all .tsv files => outputs a trained FastText model'''

# importing
import argparse
import pandas as pd
from gensim.models import FastText

# inspecting the data: text (third column), category (second column), index_id (first column)
#df = pd.read_csv("train.tsv", sep = "\t")
#print(df)

def tokenizer(sentence):
    # Handling Latin words and Chinese characters
    tokens = []
    temp = ""

    for char in sentence:
        # adding character to the temporary string if it's not a Chinese character
        if (char.isascii() and char.isalpha() or char.isdigit()) and char.strip() != "":
            temp += char
        else:
            # if the character is Chinese ...
            if temp != "":
            # ... append the non-Chinese temp string ...
                tokens.append(temp)
                temp = ""
            # ... and the current Chinese character
                tokens.append(char)
            else:
                tokens.append(char)
    # appending a possible last non-Chinese word
    if temp != "":
        tokens.append(temp)

    return tokens

def load_files(files):
    '''This function loads the data from each file and returns a list of tokenized sentences'''

    sentences = []

    # reading the file and extracting the sentence (text) column
    for file in files:
        df = pd.read_csv(file, sep = "\t")
        file_sents = df["text"]

    # converting each sentences to a list of characters and saving them in a list of tokenized sentences
        for sent in file_sents:

            tokens = tokenizer(sent)    
            sentences.append(tokens)

            # original solution (handling latin characters as single characters too):
            #chars = list(sent)
            #sentences.append(chars)

    return sentences 

def main():
    # adding the arguments
    parser = argparse.ArgumentParser(description="Train FastText character embeddings")

    # takes any number of .tsv files
    parser.add_argument("--input_files", nargs="+", required=True, help="state one or more TSV files")

    # specify vector size
    parser.add_argument("--dim", type=int, required=True, help="Vector size (embedding dimension)")

    # specify filepath to save model
    parser.add_argument("--output_path", type=str, required=True, help="file path to save the model")
    
    # other parameters: window, min_count, epochs
    parser.add_argument("--win", type=int, default=5, help="window size")
    parser.add_argument("--min_count", type=int, default=1, help="minimum count")
    parser.add_argument("--epochs", type=int, default=10, help="number of epochs")

    args = parser.parse_args()

    # getting the tokenized sentences for all the input files
    print("* Loading data *")
    sentences = load_files(args.input_files)

    # Training FastText embeddings
    print("* Training FastText *")
    model = FastText(sentences=sentences,
                 vector_size=args.dim,
                 window=args.win,
                 min_count=args.min_count,
                 workers=4 # how many CPU cores (threads) are used during training
                )

    model.train(sentences, total_examples=len(sentences), epochs=args.epochs)

    # saving the model
    print("* Saving model *")
    model.save(args.output_path)

    print("All done!")

if __name__ == "__main__":
    main()
