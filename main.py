import torch
from train_test_batch import train_batch, test_batch
from torchvision.utils import save_image
from model import U_Net

epochs = 5
lr = 0.001
model = U_Net(in_channels=3, out_channels=1)
opti = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.01)
loss = torch.nn.BCEWithLogitsLoss()

if __name__ == '__main__':
    for epoch in range(epochs):
        for (imgs, masks) in train_batch:
            y_hat = model(imgs)
            l = loss(y_hat, masks)
            print(f'for epoch {epoch} loss is {l}')
            save_image(y_hat, "img.jpg")
            l.backward()
            opti.zero_grad()
            opti.step()
