import torch as t
import numpy as np
from sklearn.metrics import f1_score
from tqdm.autonotebook import tqdm


class Trainer:

    def __init__(self,
                 model,                        # Model to be trained.
                 crit,                         # Loss function
                 optim=None,                   # Optimizer
                 train_dl=None,                # Training data set
                 val_test_dl=None,             # Validation (or test) data set
                 cuda=True,                    # Whether to use the GPU
                 early_stopping_patience=-1):  # The patience for early stopping
        self._model = model
        self._crit = crit
        self._optim = optim
        self._train_dl = train_dl
        self._val_test_dl = val_test_dl
        self._cuda = cuda

        self._early_stopping_patience = early_stopping_patience

        if cuda:
            t.cuda.empty_cache()
            self.device = t.device('cuda')
            self._model = model.cuda()
            self._crit = crit.cuda()
        else:
            self.device = t.device('cpu')
            
    def save_checkpoint(self, epoch):
        t.save({'state_dict': self._model.state_dict()}, 'checkpoints/checkpoint_{:03d}.ckp'.format(epoch))
    
    def restore_checkpoint(self, epoch_n):
        ckp = t.load('checkpoints/checkpoint_{:03d}.ckp'.format(epoch_n), 'cuda' if self._cuda else None)
        self._model.load_state_dict(ckp['state_dict'])
        
    def save_onnx(self, fn):
        m = self._model.cpu()
        m.eval()
        x = t.randn(1, 3, 300, 300, requires_grad=True)
        y = self._model(x)
        t.onnx.export(m,                 # model being run
              x,                         # model input (or a tuple for multiple inputs)
              fn,                        # where to save the model (can be a file or file-like object)
              export_params=True,        # store the trained parameter weights inside the model file
              opset_version=10,          # the ONNX version to export the model to
              do_constant_folding=True,  # whether to execute constant folding for optimization
              input_names = ['input'],   # the model's input names
              output_names = ['output'], # the model's output names
              dynamic_axes={'input' : {0 : 'batch_size'},    # variable lenght axes
                            'output' : {0 : 'batch_size'}})
            
    def train_step(self, x, y):
        self._optim.zero_grad()
        # -propagate through the network
        pred = self._model(x)
        # -calculate the loss
        loss = self._crit(pred, y)
        # -compute gradient by backward propagation
        loss.backward()
        # -update weights
        self._optim.step()
        # -return the loss
        return loss.item()
        
        
    
    def val_test_step(self, x, y):
        # predict
        # propagate through the network and calculate the loss and predictions
        pred = self._model(x)
        loss = self._crit(pred, y)
        # _, pred = t.max(out.data, 1)
        # return the loss and the predictions
        return loss, pred
        
    def train_epoch(self):
        # set training mode
        print("Training")
        self._model.train()
        train_loss = []
        # iterate through the training set
        for x_train, y_train in tqdm((self._train_dl)):
            x_train = t.tensor(x_train, dtype=t.float).to(self.device)
            y_train = t.tensor(y_train, dtype=t.float).to(self.device)
        # perform a training step
            train_loss.append(self.train_step(x_train, y_train))
        # calculate the average loss for the epoch and return it
        epoch_loss = sum(train_loss)/len(self._train_dl)
        return epoch_loss
    
    def val_test(self):
        # set eval mode. Some layers have different behaviors during training and testing (for example: Dropout, BatchNorm, etc.).
        self._model.eval()
        print("Validating")
        # disable gradient computation. Since you don't need to update the weights during testing.
        with t.no_grad(): 
            val_test_loss_list = []
            val_test_pred_list = []
            val_test_correct = 0
            f1_mean = 0
            true_val_test_set = t.empty(0).to(self.device)
            pred_val_test_set = t.empty(0).to(self.device)
        # iterate through the validation set
            for x_val_test, y_val_test in tqdm((self._val_test_dl)):
        # transfer the batch to the gpu if given
                x_val_test = t.tensor(x_val_test, dtype=t.float).to(self.device)
                y_val_test = t.tensor(y_val_test, dtype=t.float).to(self.device)
        # perform a validation step
                val_test_loss, val_test_pred = self.val_test_step(x_val_test, y_val_test)
        # save the predictions and the labels for each batch
                val_test_loss_list.append(val_test_loss)
                val_test_pred_list.append(val_test_pred)
                true_val_test_set = t.cat((true_val_test_set, y_val_test), 0)
                pred_val_test_set = t.cat((pred_val_test_set, val_test_pred), 0)
                
        epoch_loss = sum(val_test_loss_list) / len(self._val_test_dl)
        f1_mean = f1_score(true_val_test_set.cpu(), pred_val_test_set.cpu() > 0.5, average='weighted')
        print("F1-score:", f1_mean)
        # return the loss and print the calculated metrics
        return epoch_loss        
    
    def fit(self, epochs=-1):
        assert self._early_stopping_patience > 0 or epochs > 0
        # create a list for the train and validation losses, and create a counter for the epoch 
        train_loss = []
        val_test_loss = []
        counter = 0
        min_val_test_loss = np.inf
        for epoch in range(epochs):
            # stop by epoch number
            print(f"#Epoch {epoch+1} of {epochs}")
            # train for a epoch and then calculate the loss and metrics on the validation set
            train_epoch_loss = self.train_epoch()
            val_test_epoch_loss = self.val_test()
            train_loss.append(train_epoch_loss)
            val_test_loss.append(val_test_epoch_loss)
            # append the losses to the respective lists
            # use the save_checkpoint function to save the model (can be restricted to epochs with improvement)
            # self.save_checkpoint(epoch)
            # check whether early stopping should be performed using the early stopping criterion and stop if so
            if val_test_epoch_loss < min_val_test_loss:
                min_val_test_loss = val_test_epoch_loss
                self.save_checkpoint(epoch)
                counter = 0
            else:
                counter += 1

            if counter >= self._early_stopping_patience:
                print("Early stopping patience reached. Stopping.")
                break
            # return the losses for both training and validation
        return train_loss, val_test_loss
                    
        
        
        
