dataset_size = 10000
embedding_size = 100
batch_size = 10

train_path = "aclImdb/train/"
test_path = "aclImdb/test/"

import warnings

warnings.filterwarnings("ignore")

import time
import random

import os
import pandas as pd

import numpy as np

from gensim.parsing.preprocessing import remove_stopwords
from gensim.utils import tokenize
from gensim.models import Word2Vec

import torch
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

torch.autograd.set_detect_anomaly(True)

from lstm_model import SentimentLSTM

pd.set_option('display.width', 320)
pd.set_option('display.max_columns', 10)

writer = SummaryWriter('runs/experiment_2')


def imdb_data_preprocess(inpath, outpath="./", name="imdb_tr.csv", mix=False):
    indices = []
    text = []
    rating = []

    i = 0

    for filename in os.listdir(inpath + "pos"):
        data = open(inpath + "pos/" + filename, 'r', encoding="ISO-8859-1").read()
        indices.append(i)
        remove_stopwords(data)
        text.append(list(tokenize(data)))
        rating.append("1")
        i = i + 1

        if i > (dataset_size - 1) / 2:
            break

    for filename in os.listdir(inpath + "neg"):
        data = open(inpath + "neg/" + filename, 'r', encoding="ISO-8859-1").read()
        indices.append(i)
        remove_stopwords(data)
        text.append(list(tokenize(data)))
        rating.append("0")
        i = i + 1

        if i > dataset_size - 1:
            break

    Dataset = list(zip(indices, text, rating))

    if mix:
        np.random.shuffle(Dataset)

    df = pd.DataFrame(data=Dataset, columns=['row_Number', 'text', 'polarity'])
    df.to_csv(outpath + name)

    return df


def create_word2vec_model(df, outpath="./embedding"):
    # Parameters
    size = embedding_size
    window = 3
    min_count = 1
    workers = 3
    sg = 1

    word2vec_model_file = outpath + 'word2vec_' + str(size) + '.model'

    start_time = time.time()

    tokens = pd.Series(df['text']).values

    w2v_model = Word2Vec(tokens, min_count=min_count, size=size, workers=workers, window=window, sg=sg)
    print("Time taken to train word2vec model: " + str(time.time() - start_time))
    w2v_model.save(word2vec_model_file)

    # Load the model from the model file
    sg_w2v_model = Word2Vec.load(word2vec_model_file)

    return sg_w2v_model


def generate_embedding(df, model):
    embedding_df = pd.DataFrame()

    start_time = time.time()

    for index, row in df.iterrows():
        model_vector = (np.mean([model[token] for token in row['text']], axis=0)).tolist()

        if type(model_vector) is list:
            vector = [float(vector_element) for vector_element in model_vector]
            embedding_df = embedding_df.append(pd.DataFrame(vector).T)

    print("Time taken to vectorize data: " + str(time.time() - start_time))

    # Set index of DataFrame
    embedding_df.index = [i for i in range(dataset_size)]
    return embedding_df


def train_model(model, input_data, label, epoch=50):

    X_train = torch.tensor(input_data.values.astype(np.float32))
    X_train = X_train.reshape(dataset_size, 1, embedding_size)
    Y_train = torch.tensor(label.values.astype(np.float32))
    Y_train = Y_train.reshape(dataset_size)

    train_tensor = TensorDataset(X_train, Y_train)
    train_loader = DataLoader(dataset=train_tensor, batch_size=batch_size, shuffle=True)

    start_time = time.time()

    # Train model
    for e in range(epoch):
        h, cell = model.init_hidden()
        for i, data in enumerate(train_loader, 0):
            X, Y = data
            optimizer.zero_grad()

            h = torch.tensor(h.clone())
            cell = torch.tensor(cell.clone())

            Y_hat, h, cell = model(X, h, cell)
            loss = criterion(Y_hat, Y)
            loss.backward()
            optimizer.step()

            writer.add_scalar('loss', loss)

            avg_loss = loss.item()
            avg_loss += loss / len(train_loader)

        print('[Epoch: {:>4}] loss = {:>.9}'.format(e + 1, avg_loss))

    print("Time taken to train model: " + str(time.time() - start_time))
    print('Learning Finished!')

def test_model(model, input_data, label):

    # Change the data to tensor and reshape
    X_test = torch.tensor(input_data.values.astype(np.float32))
    X_test = X_test.reshape(dataset_size, 1, embedding_size)
    Y_test = torch.tensor(label.values.astype(np.float32))
    Y_test = Y_test.reshape(dataset_size)

    # Define DataLoader
    test_tensor = TensorDataset(X_test, Y_test)
    test_loader = DataLoader(dataset=test_tensor, batch_size=batch_size, shuffle=True)

    test_loss = 0
    correct = 0
    h, cell = model.init_hidden()
    for X, Y in test_loader:
        Y_hat, h, cell = model(X, h, cell)
        test_loss += criterion(Y_hat, Y).item()

        final_label = list()
        for i, predicted in enumerate(Y_hat):
            if predicted.item() > 0.5:
                final_label.append(1)
            else:
                final_label.append(0)

        compare = torch.tensor(final_label)
        correct += (compare == Y).sum()

    test_loss /= len(test_loader.dataset)
    print(
        f'==================\nTest set: Average loss : {test_loss:.4f}, Accuracy : {correct}/{len(test_loader.dataset)}'
        f'({100. * correct / len(test_loader.dataset):.0f}%)')


if __name__ == "__main__":
    random.seed(0)

    print("Preprocessing the training_data--")
    train_df = imdb_data_preprocess(inpath=train_path)
    test_df = imdb_data_preprocess(inpath=test_path)

    corpus_df = pd.concat([train_df, test_df])

    sg_w2v_model = create_word2vec_model(df=corpus_df)

    # df that has embedding vector, doesn't contain label
    em_train_df = generate_embedding(df=train_df, model=sg_w2v_model)
    em_train_df.to_csv('./train.csv')

    # Define model
    net = SentimentLSTM(dataset_size, embedding_size, 256, 2, batch_size)

    print(net)

    # Define loss function and optimizer
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.003)

    train_model(model=net, input_data=em_train_df, label=train_df['polarity'])

    print("Test the model")
    em_test_df = generate_embedding(df=test_df, model=sg_w2v_model)
    em_test_df.to_csv('./test.csv')
    test_model(net, em_test_df, label=test_df['polarity'])
