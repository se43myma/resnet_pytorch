import torch as t
from data import ChallengeDataset
from trainer import Trainer
from matplotlib import pyplot as plt
import numpy as np
import model
import pandas as pd
from sklearn.model_selection import train_test_split


# load the data from the csv file and perform a train-test-split
df = pd.read_csv("data.csv", delimiter=';')
df_train, df_val_test = train_test_split(df, test_size=0.2)

# set up data loading for the training and validation set each using t.utils.data.DataLoader and ChallengeDataset objects
train_dl = t.utils.data.DataLoader(ChallengeDataset(df_train, mode='train'), batch_size=64, shuffle=True)
val_test_dl = t.utils.data.DataLoader(ChallengeDataset(df_val_test, mode='val'), batch_size=64, shuffle=True)

# create an instance of our ResNet model
model = model.ResNet()

# set up a suitable loss criterion 
criterion = t.nn.BCELoss()
# set up the optimizer (see t.optim)
optimizer = t.optim.Adam(model.parameters(), lr=0.001)
# create an object of type Trainer and set its early stopping criterion
trainer = Trainer(model, criterion, optimizer, train_dl, val_test_dl, cuda=True, early_stopping_patience=15)

res = trainer.fit(epochs=50)

# plot the results
plt.plot(np.arange(len(res[0])), res[0], label='train loss')
plt.plot(np.arange(len(res[1])), res[1], label='val loss')
plt.yscale('log')
plt.legend()
plt.savefig('losses.png')