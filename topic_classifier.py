'''Training a feed-forward classifier (input: train.tsv, output: trained model)'''

# importing
import argparse
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import pickle
import matplotlib.pyplot as plt


def main():
    # adding the arguments
    parser = argparse.ArgumentParser(description="Train classifier")

    # input file
    parser.add_argument("--input_file", default="train.tsv", help="input TSV training file")

    # sentence embeddings
    parser.add_argument("--embeddings_file", default="train_embeddings.npy", help="file containing the sentence embeddings for the training data")

    # number of epochs
    parser.add_argument("--epochs", type=int, required=True, help="specify number of epochs")

    # batch size
    parser.add_argument("--batch_size", type=int, required=True, help="enter batch size")

    # model output path
    parser.add_argument("--output_path", type=str, required=True, help="specify the output path for the model")

    # giving the option to use validation set
    parser.add_argument("--dev_file", type=str, help="optionally add the tsv dev file to use the validation set at every epoch")
    parser.add_argument("--dev_embeddings", type=str, help="optionally add the dev embeddings to use the validaiton set at every epoch") 

    args = parser.parse_args()

    # loading sentence embeddings (for train.tsv)
    print("* Loading sentence embeddings *")
    embds = np.load(args.embeddings_file)
    #print(embds)

    # loading labels from train.tsv
    df = pd.read_csv(args.input_file, sep = "\t")
    labels = df["category"]
    #print(labels)

    # encoding labels
    le = LabelEncoder()

    #le.fit(labels)
    #classes = list(le.classes_)
    #print(classes)

    print("* Encoding labels *")
    labels_enc = le.fit_transform(labels)
    #print(labels_enc)

    # saving the label encoder
    pickle.dump(le, open("encoder.pkl", "wb"))

    # creating dataset
    print("* Creating dataset *")

    class MyDataset(Dataset):
        def __init__(self, embds, labels_enc):
            # converting embeddings (X) and labels (y) to tensors
            self.X = torch.tensor(embds, dtype=torch.float32)
            self.y = torch.tensor(labels_enc, dtype=torch.long) # long (integers) for labels due to CrossEntropyLoss
        
        def __len__(self):
            # returning the size of the dataset
            return len(self.X)

        def __getitem__(self, index):
            # returning input and label for a given index
            input_X = self.X[index]
            label_y = self.y[index]
            return (input_X, label_y)

    train_dataset = MyDataset(embds, labels_enc)
    #print(len(dataset))
    #print(dataset[0])

    # creating the dataloader (will access the dataset)
    print("* Creating dataloader *")

    dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle = True)

    # in case of validation (processing dev data)
    if args.dev_file:
        # lodading the embeddings
        dev_embds = np.load(args.dev_embeddings)

        # loading the labels
        dev_df = pd.read_csv(args.dev_file, sep = "\t")
        dev_labels = dev_df["category"]

        # encoding the labels (using the same encoder)
        dev_labels_enc = le.transform(dev_labels)

        # reusing MyDataset and DataLoader
        valid_dataset = MyDataset(dev_embds, dev_labels_enc)
        valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle = False)
    
    # building the feed-forward neural network model
    print("* Building model *")

    input_size = embds.shape[1] # embedding dimension = number of features for each input
    num_classes = len(le.classes_) # number of classes (labels)
    hidden_size = 64 # neurons in the hidden layer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # input size: embedding dimension, output size: number of topics
    class NeuralNet(nn.Module):
        def __init__(self, input_size, hidden_size, num_classes):
            super(NeuralNet, self).__init__()
            self.l1 = nn.Linear(input_size, hidden_size) # first layer
            self.relu = nn.ReLU() # activation function
            self.l2 = nn.Linear(hidden_size, num_classes) # second layer

        # defining the forward pass
        def forward(self, X):
            out = self.l1(X)
            out = self.relu(out)
            out = self.l2(out)

            return out
        
    model = NeuralNet(input_size, hidden_size, num_classes).to(device)
    model.train() # setting the model to training mode

    # adding an optimizer to update weights
    print("* Adding optimizer *")

    optimizer = torch.optim.Adam(model.parameters(), # mostly default values
                           lr=0.001, 
                           betas=(0.9, 0.999), 
                           eps=1e-08, 
                           weight_decay=0, 
                           amsgrad=False, 
                           foreach=None, 
                           maximize=False, 
                           capturable=False, 
                           differentiable=False, 
                           fused=None) 
                           #decoupled_weight_decay=False) # did not work on the server
    
    # defining the loss function
    criterion = nn.CrossEntropyLoss()

    # adding lists for the validation training loop
    train_losses = []
    valid_accuracies = []

    # setting up the training loop (with modifications for Bonus Part 1)
    print("* Training data *")

    for epoch in range(args.epochs):

        # tracking training loss
        total_loss = 0

        for X_batch, y_batch in dataloader:
            # X_batch = (batch_size, embedding_dim)
            # y_batch = (batch, size,)

            # moving data to device
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            # forward pass
            outputs = model(X_batch) # (batch_size, num_classes)

            # computing loss
            loss = criterion(outputs, y_batch)
            total_loss += loss.item()

            # backward pass
            optimizer.zero_grad() # resetting gradients
            loss.backward() # computing gradients
            optimizer.step() # updating weights

        # computing average loss
        avg_loss = total_loss / len(dataloader)
        train_losses.append(avg_loss)

        print(f"Epoch: {epoch + 1}")
        print(f"Loss (average): {avg_loss}")
        print()

        # checking shapes
        #if epoch == 0:
            #print(X_batch.shape)
            #print(outputs.shape)

        # evaluating the model on the dev set (if the user chose to add it)
        if args.dev_file:
            model.eval()

            valid_predictions = []
            valid_labels = []
            with torch.no_grad():
                for X_batch, y_batch in valid_loader:

                    # moving data to device (same as model)
                    X_batch = X_batch.to(device)
                    y_batch = y_batch.to(device)

                    # forward pass
                    outputs = model(X_batch)
                    predictions = torch.argmax(outputs, dim=1)

                    # converting tensors to lists and storing to lists
                    valid_predictions.append(predictions.cpu().numpy()) # predicted labels
                    valid_labels.append(y_batch.cpu().numpy()) # true label

            # flattening lists
            predictions = np.concatenate(valid_predictions).tolist()
            true_labels = np.concatenate(valid_labels).tolist() # integers

            # computing accuracy
            correct = 0
            for i in range(len(predictions)):
                if predictions[i] == true_labels[i]:
                    correct += 1

            total = len(predictions)

            accuracy = correct / total
            print("* Computing accuracy *")
            print(f"Accuracy: {accuracy}")
            print()
            valid_accuracies.append(accuracy)

            # switching back to training mode
            model.train()

    # saving the trained model as a file
    torch.save(model.state_dict(), args.output_path) # state_dict() = learned weights

    # using matplotlib to create a plot of the performance of the model
    if args.dev_file:
        print("* Evaluating performance *")

        x = range(1, args.epochs + 1)
        y = valid_accuracies

        plt.plot(x, y, color="r", label="Accuracy")
        plt.plot(x, train_losses, color="g", label="Loss")

        plt.title("Performance of the model per training epoch")
        plt.xlabel("Epochs")
        plt.ylabel("Performance")
        plt.legend()

        # saving the figure
        plt.savefig("training_performance.png")
        plt.show() # showing in console

    print("All done!")


if __name__ == "__main__":
    main()
