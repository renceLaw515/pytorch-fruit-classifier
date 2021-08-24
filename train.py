import cv2
import numpy as np
import os
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from torchmetrics.classification.accuracy import Accuracy
import pytorch_lightning as pl
from pytorch_lightning import Trainer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class FruitDataSet():
    def __init__(self,root):
        self.img = []
        self.label = []
        transform = transforms.Compose([transforms.ToTensor(),
        transforms.Resize((224,224))])
        class_list = os.listdir(root)
        for i, each in enumerate(class_list):
            file_list = os.listdir(f'{root}/{each}')
            for file in file_list:
                img = cv2.imread(f'{root}/{each}/{file}')
                self.img.append(transform(img))
                self.label.append(i)

    def __getitem__(self,key):
        return self.img[key], self.label[key]
    
    def __len__(self):
        return len(self.img)

class FruitDataModule(pl.LightningDataModule):
    def __init__(self,batch_size):
        super().__init__()
        self.batch_size = batch_size
    
    def prepare_data(self):
        print("Preparing Dataset")
        self.train_dataset = FruitDataSet("./data/train")
        self.test_dataset = FruitDataSet("./data/test")
        
    def setup(self,stage):
        train_size = int(0.8 * len(self.train_dataset))
        val_size = len(self.train_dataset) - train_size
        self.fruit_train, self.fruit_val = random_split(self.train_dataset, [train_size, val_size])
        self.fruit_test = self.test_dataset
    def train_dataloader(self):
        fruit_train = DataLoader(self.fruit_train, batch_size=self.batch_size,num_workers=4)
        return fruit_train

    def val_dataloader(self):
        fruit_val = DataLoader(self.fruit_val, batch_size=self.batch_size, num_workers=4)
        return fruit_val

    def test_dataloader(self):
        fruit_test = DataLoader(self.fruit_test, batch_size=self.batch_size, num_workers=4)
        return fruit_test


class FruitNet(pl.LightningModule):
    def __init__(
        self, 
        conv2d_1: int = 16,
        filter1_size: int =3,
        conv2d_2: int = 16,
        filter2_size: int =5,
        conv2d_3: int = 16,
        filter3_size: int =7,
        dropout: float = 0.2,
        lin1_size: tuple = (16*5*5,128), 
        output_size: tuple = (128,6),
        lr: float = 0.001
         ):
         super().__init__()

        # this line ensures params passed to LightningModule will be saved to ckpt
        # it also allows to access params with 'self.hparams' attribute

         self.save_hyperparameters()

         self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(3, conv2d_1, filter1_size),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(3,3),
            torch.nn.Dropout(p=dropout))

         self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(conv2d_1, conv2d_2, filter2_size),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(3,3),
            torch.nn.Dropout(p=dropout))

         self.layer3 = torch.nn.Sequential(
            torch.nn.Conv2d(conv2d_2, conv2d_3, filter3_size),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(3,3),
            torch.nn.Dropout(p=dropout))
         self.fc1 = torch.nn.Sequential(
            torch.nn.Linear(in_features=lin1_size[0],out_features=lin1_size[1]),
            torch.nn.ReLU())
         self.fc2 = torch.nn.Sequential(
            torch.nn.Linear(in_features=output_size[0],out_features=output_size[1]))
        
        # loss function
         self.criterion = torch.nn.CrossEntropyLoss()
         self.train_accuracy = Accuracy()
         self.val_accuracy = Accuracy()
         self.test_accuracy = Accuracy()

    def forward(self,x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.view(-1, 16 * 5 * 5)
        out = self.fc1(out)
        out = self.fc2(out)
        return out

    def step(self, batch):
        x, y = batch
        logits = self.forward(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        return loss, preds, y

    def training_step(self, batch, batch_idx):
        loss, preds, targets = self.step(batch)

        # log train metrics
        acc = self.train_accuracy(preds, targets)
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("train/acc", acc, on_step=False, on_epoch=True, prog_bar=True)

        return {"loss": loss, "preds": preds, "targets": targets}
            
    def training_epoch_end(self, outputs):
        
        pass

    def validation_step(self, batch, batch_idx):
        loss, preds, targets = self.step(batch)

        # log val metrics
        acc = self.val_accuracy(preds, targets)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("val/acc", acc, on_step=False, on_epoch=True, prog_bar=True)

        return {"loss": loss, "preds": preds, "targets": targets}

    def validation_epoch_end(self, outputs):
        pass

    def test_step(self, batch, batch_idx: int):
        loss, preds, targets = self.step(batch)

        # log test metrics
        acc = self.test_accuracy(preds, targets)
        self.log("test/loss", loss, on_step=False, on_epoch=True)
        self.log("test/acc", acc, on_step=False, on_epoch=True)

        return {"loss": loss, "preds": preds, "targets": targets}

    def test_epoch_end(self, outputs):
        pass     

    def configure_optimizers(self):
        return torch.optim.Adam(
            params=self.parameters(), lr=self.hparams.lr
        )   

if __name__ == '__main__':
  dataset = FruitDataModule(batch_size=16)
  model = FruitNet()
  trainer = Trainer(gpus=1, max_epochs=10)
  trainer.fit(model,datamodule=dataset)
  trainer.test(model)
  torch.save(model.state_dict(), 'save.pt')