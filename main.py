# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


# def print_hi(name):
#     # Use a breakpoint in the code line below to debug your script.
#     print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.
#
#
# # Press the green button in the gutter to run the script.
# if __name__ == '__main__':
#     print_hi('PyCharm')
#
# # See PyCharm help at https://www.jetbrains.com/help/pycharm/

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
from torch.nn import functional as F
from torchvision import datasets,transforms
from torchvision.utils import save_image

torch.set_printoptions(profile="full")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

kwargs = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {}

train_load = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True, transform=transforms.ToTensor()),
    batch_size=128, shuffle=True, **kwargs)
test_load = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.ToTensor()),
    batch_size=128, shuffle=True, **kwargs)

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.layer1 = nn.Linear(784, 400)
        self.layer21 = nn.Linear(400, 20)
        self.layer22 = nn.Linear(400, 20)
        self.layer3 = nn.Linear(20, 400)
        self.layer4 = nn.Linear(400, 784)

    def encoder(self, x):
        h1 = F.relu(self.layer1(x))
        return self.layer21(h1), self.layer22(h1)

    def decoder(self, z):
        h1 = F.relu(self.layer3(z))
        return torch.sigmoid_(self.layer4(h1))

    def repara(self, mu, log_fai2):
        fai = torch.exp_(0.5*log_fai2)
        s = torch.randn_like(mu)
        return mu+s*fai

    def forward(self, x):
        mu, log_fai2 = self.encoder(x.view(-1, 784))
        z = self.repara(mu, log_fai2)
        x1 = self.decoder(z)
        return x1, mu, log_fai2

model = VAE().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

def loss_fun(recon_x, x, mu, log_fai2):
    loss1 = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
    loss2 = -0.5 * torch.sum(1+log_fai2-mu.pow(2)-log_fai2.exp())
    return loss1+loss2

def train(epoch):
    model.train()
    train_loss = 0
    for i, (data, _) in enumerate(train_load):
        data = data.to(device)
        optimizer.zero_grad()
        recog_data, mu, log_fai2 = model(data)
        loss = loss_fun(recog_data, data, mu, log_fai2)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()

        # if i == 0:
        #     print(mu.size())
        #     print(log_fai2.size())

    print("====> Epoch:{} Average loss:{:.4f}".format(epoch, train_loss/len(train_load.dataset)))

def test(epoch):
    model.eval()
    for i, (data, _) in enumerate(test_load):
        data = data.to(device)
        recog_data, mu, log_fai2 = model(data)
        if (i == 0) and (epoch % 10 == 0):
            n = min(data.size(0), 8)
            sample_contrast = torch.cat([data[:n], recog_data.view(128, 1, 28, 28)[:n]])
            save_image(sample_contrast.cpu(), 'img/sample_contrast_' + str(epoch) + '.png', nrow=n)

if __name__ =="__main__":
    for epoch in range(1, 21):
        train(epoch)
        test(epoch)
        if (epoch % 10 == 0):
            sample = torch.randn(64, 20).to(device)
            sample = model.decoder(sample).cpu()
            save_image(sample.view(64, 1, 28, 28), 'img/7_m_sample' + str(epoch) + '.png')




























