import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import os
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from sklearn.model_selection import train_test_split


class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=2, dropout=0):
        super(GRUModel, self).__init__()
        # self.gru = nn.GRU(input_size, hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout, bidirectional=True)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout, bidirectional=True)
        self.fc1 = nn.Linear(hidden_size * 2, hidden_size) 
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x, lengths):
        x_packed = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        # out_packed, _ = self.gru(x_packed)
        out_packed, _ = self.lstm(x_packed)
        out, _ = pad_packed_sequence(out_packed, batch_first=True)

        #maxpooling
        out, _ = torch.max(out, dim=1)
        out = self.fc1(out)
        out = self.relu(out)
        #out = self.dropout(out)
        out = self.fc2(out)
        return out
    
    def softmax(self, x):
        return torch.softmax(x, dim=-1)


class CustomDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels = labels

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, index):
        # length should be the length of the sequence without nan values at the end
        length = np.argwhere(np.isnan(self.sequences[index]))[0, 0] if np.isnan(self.sequences[index]).any() else len(self.sequences[index])
        length = 600 #1000 #2459 # the minimal length observed in the dataset
        sequence = torch.tensor(self.sequences[index]).unsqueeze(1)
        label = torch.tensor(self.labels[index])
        return sequence, length, label




train_X = pd.read_csv('X_train.csv')
train_y = pd.read_csv('y_train.csv')
train_X = train_X.drop('id', axis=1)
train_y = train_y.drop('id', axis=1)
train_df = pd.concat([train_X, train_y], axis=1)

train_df_0 = train_df[train_df['y'] == 0]
train_df_1 = train_df[train_df['y'] == 1]
train_df_2 = train_df[train_df['y'] == 2]
train_df_3 = train_df[train_df['y'] == 3]

train_df_0, val_df_0 = train_test_split(train_df_0, test_size=0.2, random_state=42)
train_df_1, val_df_1 = train_test_split(train_df_1, test_size=0.2, random_state=42)
train_df_2, val_df_2 = train_test_split(train_df_2, test_size=0.2, random_state=42)
#train_df_3, val_df_3 = train_test_split(train_df_3, test_size=0.2, random_state=42)

# overample each class to have the same number of samples as the majority class
len_0 = len(train_df_0)
len_1 = len(train_df_1)
len_2 = len(train_df_2)
#len_3 = len(train_df_3)
biggest_len = max(len_0, len_1, len_2) #, len_3)
smallest_len = min(len_0, len_1, len_2) #, len_3)

#ovaersampling
#train_df_0 = pd.concat([train_df_0] * (biggest_len // len_0 + 1), axis=0)
#train_df_1 = pd.concat([train_df_1] * (biggest_len // len_1 + 1), axis=0)
#train_df_2 = pd.concat([train_df_2] * (biggest_len // len_2 + 1), axis=0)
#train_df_3 = pd.concat([train_df_3] * (biggest_len // len_3 + 1), axis=0)

# cut the overampled classes to have the same number of samples as the smallest class
train_df_0 = train_df_0[:smallest_len]
train_df_1 = train_df_1[:smallest_len]
train_df_2 = train_df_2[:smallest_len]
#train_df_3 = train_df_3[:smallest_len]

#class_weights = [len(train_df) / len(train_df_0), len(train_df) / len(train_df_1), len(train_df) / len(train_df_2)] #, len(train_df) / len(train_df_3)]
#class_weights_normalized = [i / sum(class_weights) for i in class_weights]

train_df = pd.concat([train_df_0, train_df_1, train_df_2], axis=0) #, train_df_3], axis=0)
val_df = pd.concat([val_df_0, val_df_1, val_df_2], axis=0) # val_df_3], axis=0)
train_y = train_df.pop('y')
val_y = val_df.pop('y')

train_df = train_df.values.astype(float)
val_df = val_df.values.astype(float)
train_y = train_y.values.astype(int)
val_y = val_y.values.astype(int)


input_size = 1
hidden_size = 256
output_size = 3 
batch_size = 64
num_epochs = 10
learning_rate = 0.01


train_dataset = CustomDataset(train_df, train_y)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataset = CustomDataset(val_df, val_y)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GRUModel(input_size, hidden_size, output_size)
#class_weights = torch.FloatTensor(class_weights_normalized).to(device)
criterion = nn.CrossEntropyLoss() #(weight=class_weights)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
model.to(device)

weights_dir = 'gru_weights'
os.makedirs(weights_dir, exist_ok=True)


for epoch in range(num_epochs):
    running_loss = 0.0
    model.train()
    for sequences, lengths, labels in tqdm(train_loader):
        sequences, lengths, labels = sequences.to(device), lengths, labels.to(device)
        outputs = model(sequences.float(), lengths)
        outputs = model.softmax(outputs)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for sequences, lengths, labels in val_loader:
            #print(labels)
            sequences, lengths, labels = sequences.to(device), lengths, labels.to(device)
            outputs = model(sequences.float(), lengths)
            test_loss = criterion(outputs, labels)
            _, predicted = torch.max(outputs.data, 1)
            predicted_np = outputs.cpu().numpy()
            #print(predicted_np)
            #print(np.unique(predicted_np, return_counts=True))
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        accuracy = correct / total
        print(f'Epoch {epoch+1}/{num_epochs}, Train loss: {running_loss/len(train_loader):.4f}, Val loss: {test_loss.item():.4f}, Validation Accuracy: {accuracy:.2%}')
        torch.save(model.state_dict(), f'{weights_dir}/gru_epoch_{epoch+1}.pth')
