import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm
from torchvision import datasets, transforms
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

class LeNet5(nn.Module):
    def __init__(self, num_classes, **kwargs):
        super(LeNet5, self).__init__(**kwargs)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, padding=(4,4))
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_features=480, out_features=84)
        self.fc2 = nn.Linear(in_features=84, out_features=10)


    def forward(self, X):
        h = F.max_pool2d(torch.tanh(self.conv1(X)), kernel_size =(2,2))
        h = F.max_pool2d(torch.tanh(self.conv2(h)), kernel_size =(2,2))
        h = self.flatten(torch.tanh(self.conv3(h)))
        h = torch.tanh(self.fc1(h))
        h = self.fc2(h)
        return h

lenet5 = LeNet5(10).to(device)
loss_fn = nn.CrossEntropyLoss()
learning_rate = 1e-3
batch_size = 64
epochs = 30
optimizer = torch.optim.SGD(lenet5.parameters(), lr=learning_rate,
                            momentum=0.9, nesterov=True)

#Data pipeline.

train_dataset = datasets.MNIST(
    root = 'data/MNIST',
    train=True,
    download=True,
    transform=transforms.ToTensor()

)

test_dataset = datasets.MNIST(
    root='mnist_data',
    train=False,
    download=True,
    transform=transforms.ToTensor()
)

train_dataloader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle = True,
    drop_last=True,
    num_workers=8
)

test_dataloader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    drop_last=True,
    num_workers=8
)

#Training loop.

for epoch in range(epochs):
    train_loss, tr_correct_preds = 0, 0
    val_loss, tst_correct_preds = 0, 0

    for (train_X, train_y) in train_dataloader:
        train_X, train_y = train_X.to(device), train_y.to(device)
        train_preds = lenet5(train_X)
        loss = loss_fn(train_preds, train_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        tr_correct_preds += (train_preds.argmax(dim=1) == train_y).sum().item()

    for (val_X, val_y) in test_dataloader:
        with torch.no_grad():
            val_X, val_y = val_X.to(device), val_y.to(device)
            val_preds = lenet5(val_X)
            val_loss += loss_fn(val_preds, val_y).item()
            tst_correct_preds += (val_preds.argmax(dim=1) == val_y).sum().item()

    num_train_steps = len(train_dataloader.dataset) // batch_size
    num_val_steps = len(test_dataloader.dataset) // batch_size

    print('Epoch {}: \n Train Loss:{}, Train acc: {}, Val loss: {}, Val acc:{},\n'
          'Correct Training Samples: {}, Correct Validation Samples: {} \n'
          .format(epoch,
                  train_loss/num_train_steps,
                  tr_correct_preds/ (num_train_steps * batch_size),
                  val_loss/num_val_steps,
                  tst_correct_preds/ (num_val_steps * batch_size),
                  tr_correct_preds, tst_correct_preds))






