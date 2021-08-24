import streamlit as st
from PIL import Image
import torchvision.transforms as transforms
import torch
import pytorch_lightning as pl
from torchmetrics.classification.accuracy import Accuracy
from torch.utils.data import DataLoader, random_split
import os
import cv2

## Model parameters (DataSet, DataModule, NN classes) from train.py ##
class FruitDataSet():
    def __init__(self,root):
      pass

    def __getitem__(self,key):
        return self.img[key], self.label[key]
    
    def __len__(self):
        return len(self.img)

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
        acc = self.train_accuracy(preds, targets)
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("train/acc", acc, on_step=False, on_epoch=True, prog_bar=True)
        return {"loss": loss, "preds": preds, "targets": targets}
            
    def training_epoch_end(self, outputs):
        pass

    def validation_step(self, batch, batch_idx):
        loss, preds, targets = self.step(batch)

        acc = self.val_accuracy(preds, targets)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("val/acc", acc, on_step=False, on_epoch=True, prog_bar=True)

        return {"loss": loss, "preds": preds, "targets": targets}

    def validation_epoch_end(self, outputs):
        pass

    def test_step(self, batch, batch_idx: int):
        loss, preds, targets = self.step(batch)
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

## Prediction via Streamlit Framework ##
transform = transforms.Compose(
    [transforms.ToTensor(),
    transforms.Resize((224,224))])
st.title("Fruit Classifier")
st.image(Image.open('example.png'))
file_up = st.file_uploader("Upload an image to classifiy whether the fruit is rotten or fresh (apple, orange, banana)", type=["jpg","png"])
class_list = ['freshapples', 'freshbanana', 'freshoranges', 'rottenapples', 'rottenbanana', 'rottenoranges']
predict_result = ''
if file_up:
    image = Image.open(file_up)
    st.image(image, caption='Uploaded Image. Please click below button for classification', use_column_width=True)
    isClicked = st.button("Click Here for prediction")
    if isClicked:
      image = transform(image)
      image = image.view(1,3,224,224)
      model = FruitNet()
      model.load_state_dict(torch.load('save.pt'))
      model.eval()
      out = model(image)
      prob = torch.nn.functional.softmax(out, 1)[0] * 100
      predict_result = class_list[torch.argmax(prob).item()]
      print(predict_result)

if len(predict_result)>0:
    st.text(f'The predicted class is {predict_result}')