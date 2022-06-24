import numpy as np
from sklearn.model_selection import train_test_split


def sigmod(x):
    return 1 / (1 + np.exp(-x))


def numbers_to_one_hot(numbers, n_classes):
    one_hot = np.zeros((numbers.shape[0], n_classes))
    one_hot[np.arange(numbers.shape[0]), numbers] = 1
    return one_hot


def mini_batch_generator(X, y, batch_size):
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)

    for start_idx in range(0, X.shape[0] - batch_size + 1, batch_size):
        idx = indices[start_idx:start_idx + batch_size]
        yield X[idx], y[idx]


def compute_loss_and_acc(model, X, y):
    y_proba = model.predict(X)
    return mse_loss(y, y_proba, model.number_classes), accuracy(y, y_proba.argmax(axis=-1))

class NeuralNetMLP:
    def __init__(self, n_hidden=1, lr=0.01, number_classes=10, number_features=4, random_seed=1):
        super().__init__()
        self.lr = lr
        self.random = np.random.RandomState(random_seed)
        self.number_classes = number_classes

        self.h_weights = self.random.normal(0, 0.1, (n_hidden, number_features))
        self.o_weights = self.random.normal(0, 0.1, (number_classes, n_hidden))
        self.h_bias = self.random.normal(0, 0.1, n_hidden)
        self.o_bias = self.random.normal(0, 0.1, number_classes)

    def forward(self, X):
        net_input = np.dot(X, self.h_weights.T) + self.h_bias
        h_a = sigmod(net_input)

        out_z = np.dot(h_a, self.o_weights.T) + self.o_bias
        out_a = sigmod(out_z)

        return h_a, out_a

    def predict(self, X):
        h_a, out_a = self.forward(X)
        return out_a

    def backward(self, X, a_h, a_out, y):
        y_true = numbers_to_one_hot(y, self.number_classes)

        ##Part 1: d_loss/d_OutWeights
        d_loss__d_a_out = 2.0 * (a_out - y_true) / y_true.shape[0]
        d_a_out__d_z_out = a_out * (1.0 - a_out)  # Sigmod derivative
        d_out_w = d_loss__d_a_out * d_a_out__d_z_out
        d_z_out__d_w_out = a_h
        d_loss__d_w_out = np.dot(d_out_w.T, d_z_out__d_w_out)
        d_loss__db_out = np.sum(d_out_w, axis=0)

        ##Part 2: d_loss/d_HiddenWeights
        d_z_out__a_h = self.o_weights
        d_loss__a_h = np.dot(d_out_w, d_z_out__a_h)
        d_a_h__d_z_h = a_h * (1.0 - a_h)
        d_z_h__d_w_h = X
        d_loss__d_w_h = np.dot((d_loss__a_h * d_a_h__d_z_h).T, d_z_h__d_w_h)
        d_loss__d_b_h = np.sum((d_loss__a_h * d_a_h__d_z_h), axis=0)
        return d_loss__d_w_out, d_loss__db_out, d_loss__d_w_h, d_loss__d_b_h

    def perform_epoch(self, X, y, X_val, y_val, batch_size):
        for X_batch, y_batch in mini_batch_generator(X, y, batch_size):
            h_a, out_a = self.forward(X_batch)
            d_loss__d_w_out, d_loss__db_out, d_loss__d_w_h, d_loss__d_b_h = self.backward(X_batch, h_a, out_a, y_batch)

            lr = self.lr
            self.h_weights -= lr * d_loss__d_w_h
            self.h_bias -= lr * d_loss__d_b_h
            self.o_weights -= lr * d_loss__d_w_out
            self.o_bias -= lr * d_loss__db_out

        train_loss, train_acc = compute_loss_and_acc(self, X, y)
        val_loss, val_acc = compute_loss_and_acc(self, X_val, y_val)
        return train_loss, train_acc, val_loss, val_acc

    def fit(self, X, y, X_val, y_val, batch_size=32, epochs=10):
        train_losses = []
        train_accs = []
        val_losses = []
        val_accs = []
        for epoch_num in range(epochs):
            print("Performing epoch number: ", epoch_num)
            train_loss, train_acc, val_loss, val_acc = self.perform_epoch(X, y, X_val, y_val, batch_size)
            train_losses.append(train_loss)
            train_accs.append(train_acc)
            val_losses.append(val_loss)
            val_accs.append(val_acc)
            print(f'Epoch: {epoch_num+1:03d}/{epochs:03d} '
                  f'| Train MSE: {train_loss:.2f} '
                  f'| Train Acc: {train_acc:.2f}% '
                  f'| Val MSE: {val_loss:.2f} '
                  f'| Valid Acc: {val_acc:.2f}%')
        return train_losses, train_accs, val_losses, val_accs

    def score(self, X, y):
        loss, acc = compute_loss_and_acc(self, X, y)
        return loss, acc


def mse_loss(y_true, y_proba, num_classes):
    y_true = numbers_to_one_hot(y_true.astype(np.int32), num_classes)
    return np.mean((y_true - y_proba) ** 2)


def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)


def prepare_dataset(X, y):
    X_np = X.to_numpy()
    y_np = y.to_numpy()
    X_norm = ((X_np / 255.0) - 0.5) * 2
    X_tmp, X_test, y_tmp, y_test = train_test_split(X_norm, y_np, test_size=10000, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_tmp, y_tmp, test_size=5000, random_state=42)
    return X_train, X_val, X_test, y_train, y_val, y_test
