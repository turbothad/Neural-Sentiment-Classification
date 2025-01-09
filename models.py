# models.py

import torch
import torch.nn as nn
from torch import optim
import numpy as np
import random
from sentiment_data import *
from collections import defaultdict


class SentimentClassifier(object):
    """
    Sentiment classifier base type
    """

    def predict(self, ex_words: List[str], has_typos: bool) -> int:
        """
        Makes a prediction on the given sentence
        :param ex_words: words to predict on
        :param has_typos: True if we are evaluating on data that potentially has typos, False otherwise. If you do
        spelling correction, this parameter allows you to only use your method for the appropriate dev eval in Q3
        and not otherwise
        :return: 0 or 1 with the label
        """
        raise Exception("Don't call me, call my subclasses")

    def predict_all(self, all_ex_words: List[List[str]], has_typos: bool) -> List[int]:
        """
        You can leave this method with its default implementation, or you can override it to a batched version of
        prediction if you'd like. Since testing only happens once, this is less critical to optimize than training
        for the purposes of this assignment.
        :param all_ex_words: A list of all exs to do prediction on
        :param has_typos: True if we are evaluating on data that potentially has typos, False otherwise.
        :return:
        """
        return [self.predict(ex_words, has_typos) for ex_words in all_ex_words]


class TrivialSentimentClassifier(SentimentClassifier):
    def predict(self, ex_words: List[str], has_typos: bool) -> int:
        """
        :param ex:
        :return: 1, always predicts positive class
        """
        return 1
    
class PrefixEmbeddings:
    def __init__(self, word_embeddings: WordEmbeddings, prefix_length: int = 3):
        self.prefix_length = prefix_length
        self.prefix_to_index = {}
        self.index_to_prefix = []
        self.vectors = []
        self.build_prefix_embeddings(word_embeddings)

    def build_prefix_embeddings(self, word_embeddings: WordEmbeddings):
        prefix_to_vectors = defaultdict(list)
        for i in range(len(word_embeddings.word_indexer)):
            word = word_embeddings.word_indexer.get_object(i)
            vector = word_embeddings.vectors[i]
            prefix = word[:self.prefix_length]
            prefix_to_vectors[prefix].append(vector)

        for prefix, vectors in prefix_to_vectors.items():
            avg_vector = np.mean(vectors, axis=0)
            self.prefix_to_index[prefix] = len(self.index_to_prefix)
            self.index_to_prefix.append(prefix)
            self.vectors.append(avg_vector)

        self.vectors = np.array(self.vectors)

    def index_of(self, prefix: str) -> int:
        return self.prefix_to_index.get(prefix, self.prefix_to_index.get("UNK"))


def get_initialized_prefix_embedding_layer(prefix_embeddings: PrefixEmbeddings, frozen: bool = True) -> nn.Embedding:
    num_embeddings, embedding_dim = prefix_embeddings.vectors.shape
    embedding_layer = nn.Embedding(num_embeddings, embedding_dim, padding_idx=0)
    embedding_layer.weight = nn.Parameter(torch.tensor(prefix_embeddings.vectors, dtype=torch.float32))
    embedding_layer.weight.requires_grad = not frozen
    return embedding_layer

class NeuralSentimentClassifier(SentimentClassifier, nn.Module):
    def __init__(self, embeddings: PrefixEmbeddings, hidden_dim: int = 300, use_prefix: bool = False):
        SentimentClassifier.__init__(self)
        nn.Module.__init__(self)
        self.use_prefix = use_prefix
        self.embedding = get_initialized_prefix_embedding_layer(embeddings)
        self.indexer = embeddings if use_prefix else embeddings.word_indexer
        self.hidden_dim = hidden_dim
        self.fc1 = nn.Linear(self.embedding.embedding_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 2)  # binary classification
        self.dropout = nn.Dropout(0.5)  # dropout for regularization
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, ex_indices: List[List[int]]):
        lengths = [len(seq) for seq in ex_indices]
        max_len = max(lengths)
        padded_ex_indices = [seq + [0] * (max_len - len(seq)) for seq in ex_indices]
        
        embeds = self.embedding(torch.tensor(padded_ex_indices))
        avg_embeds = torch.mean(embeds, dim=1)
        hidden = torch.relu(self.fc1(avg_embeds))
        hidden = self.dropout(hidden)  # dropout
        output = self.fc2(hidden)
        return self.softmax(output)

    def predict(self, ex_words: List[str], has_typos: bool) -> int:
        self.eval()
        with torch.no_grad():
            if self.use_prefix:
                indices = [self.indexer.index_of(word[:self.indexer.prefix_length]) for word in ex_words]
            else:
                indices = [self.indexer.index_of(word) if self.indexer.index_of(word) != -1 else self.indexer.index_of("UNK") for word in ex_words]
            output = self.forward([indices])
            return torch.argmax(output, dim=1).item()

    def predict_all(self, all_ex_words: List[List[str]], has_typos: bool) -> List[int]:
        self.eval()
        predictions = []
        with torch.no_grad():
            for ex_words in all_ex_words:
                if self.use_prefix:
                    indices = [self.indexer.index_of(word[:self.indexer.prefix_length]) for word in ex_words]
                else:
                    indices = [self.indexer.index_of(word) if self.indexer.index_of(word) != -1 else self.indexer.index_of("UNK") for word in ex_words]
                output = self.forward([indices])
                predictions.append(torch.argmax(output, dim=1).item())
        return predictions

def train_deep_averaging_network(args, train_exs: List[SentimentExample], dev_exs: List[SentimentExample],
                                 word_embeddings: WordEmbeddings, train_model_for_typo_setting: bool) -> NeuralSentimentClassifier:
    if train_model_for_typo_setting:
        embeddings = PrefixEmbeddings(word_embeddings)
        model = NeuralSentimentClassifier(embeddings, use_prefix=True)
    else: 
        model = NeuralSentimentClassifier(word_embeddings, use_prefix=False)
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    loss_function = nn.CrossEntropyLoss()

    epochs = args.num_epochs 
    batch_size = args.batch_size

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        random.shuffle(train_exs)
        for i in range(0, len(train_exs), batch_size):
            batch_exs = train_exs[i:i+batch_size]
            batch_words = [ex.words for ex in batch_exs]
            batch_labels = [ex.label for ex in batch_exs]

            if train_model_for_typo_setting:
                batch_indices = [[model.indexer.index_of(word[:model.indexer.prefix_length]) for word in words] for words in batch_words]
            else:
                batch_indices = [[model.indexer.index_of(word) if model.indexer.index_of(word) != -1 else model.indexer.index_of("UNK") for word in words] for words in batch_words]

            model.zero_grad()
            output = model.forward(batch_indices)
            loss = loss_function(output, torch.tensor(batch_labels))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_exs)}")

    return model