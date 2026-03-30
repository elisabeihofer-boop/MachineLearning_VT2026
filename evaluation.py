'''Loads the saved (and trained) model and evaluates its performance on the test set'''

# imports
import argparse
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import LabelEncoder
import pickle
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt


def main():
    # adding the arguments
    parser = argparse.ArgumentParser(description="Evaluate classifier model")

    # input file
    parser.add_argument("--input_file", default="test.tsv", help="input TSV testing file")

    # model path
    parser.add_argument("--model_path", type=str, required=True, help="specify the path for the previously saved classifier model")

    # sentence embeddings
    parser.add_argument("--embeddings_file", default="test_embeddings.npy", help="file containing the sentence embeddings for the training data")

    # batch size
    parser.add_argument("--batch_size", type=int, required=True, help="enter batch size")
    
    args = parser.parse_args()

    # loading the sentence embeddings
    print("* Loading sentence embeddings *")
    embds = np.load(args.embeddings_file)
    
    # loading labels from train.tsv
    df = pd.read_csv(args.input_file, sep = "\t")
    labels = df["category"] # strings
    #print(labels)

    # encoding labels
    le = pickle.load(open("encoder.pkl", "rb")) # reusing the same encoder from training
    #print(le.classes_)

    print("* Encoding labels *")
    labels_enc = le.transform(labels) # only transform, not fit

    # creating dataset
    print("* Creating dataset *")

    class MyDataset(Dataset):
        def __init__(self, embds, labels_enc):
            # converting embeddings (X) and labels (y) to tensors
            self.X = torch.tensor(embds, dtype=torch.float32)
            self.y = torch.tensor(labels_enc, dtype=torch.long) # integers => CrossEntropyLoss
        
        def __len__(self):
            # returning the size of the dataset
            return len(self.X)

        def __getitem__(self, index):
            # returning input and label for a given index
            input_X = self.X[index]
            label_y = self.y[index]
            return (input_X, label_y)

    test_dataset = MyDataset(embds, labels_enc)

    # creating the dataloader (will access the dataset)
    print("* Creating dataloader *")
    dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle = False)

    # re-creating the feed-forward neural network model

    input_size = embds.shape[1] # embedding dimension
    num_classes = len(le.classes_) # number of classes (labels)
    hidden_size = 64 # number of neurons in the hidden layer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # input size: embedding dimension, output size: number of topics
    class NeuralNet(nn.Module):
        def __init__(self, input_size, hidden_size, num_classes):
            super(NeuralNet, self).__init__()
            self.l1 = nn.Linear(input_size, hidden_size) # first layer
            self.relu = nn.ReLU() # activation function
            self.l2 = nn.Linear(hidden_size, num_classes) # second layer

        def forward(self, X):
            out = self.l1(X)
            out = self.relu(out)
            out = self.l2(out)

            return out


    # creating the model
    model = NeuralNet(input_size, hidden_size, num_classes)

    # loading the saved weights
    print("* Loading trained weights *")
    model.load_state_dict(torch.load(args.model_path))

    # moving the model to device
    model.to(device)

    # setting the model to evaluating mode
    model.eval()

    # getting the predictions
    print("* Predicting *")

    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for X_batch, y_batch in dataloader:

            # moving data to device
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            # forward pass
            outputs = model(X_batch)
            predictions = torch.argmax(outputs, dim=1)

            # converting tensors to lists and storing to lists
            all_predictions.append(predictions.cpu().numpy()) # predicted labels
            all_labels.append(y_batch.cpu().numpy()) # true label

    # flattening lists
    predictions = np.concatenate(all_predictions).tolist()
    true_labels = np.concatenate(all_labels).tolist() # integers

    # computing accuracy
    print("* Computing accuracy *")

    correct = 0
    for i in range(len(predictions)):
        if predictions[i] == true_labels[i]:
            correct += 1

    total = len(predictions)

    accuracy = correct / total
    print(f"Accuracy: {accuracy:.4f}")

    # printing confusion matrix
    cm = confusion_matrix(true_labels, predictions, labels=range(num_classes))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                                  display_labels=le.classes_)
    disp.plot()

    plt.savefig("confusion_matrix.png")
    plt.show()

    print("All done!")

if __name__ == "__main__":
    main()
