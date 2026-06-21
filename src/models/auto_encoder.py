import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import numpy as np

from src.models.base_detector import BaseDetector

from configs.settings import (
    AE_EPOCHS,
    AE_BATCH_SIZE,
    AE_LR,
    ANOMALY_PERCENTILE,
    RANDOM_STATE
)

import random

from pathlib import Path

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class AutoEncoder(nn.Module):

    def __init__(self, input_dim):

        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),

            nn.Linear(64, 32),
            nn.ReLU(),

            nn.Linear(32,8)
        )

        self.decoder = nn.Sequential(
            nn.Linear(8,32),
            nn.ReLU(),

            nn.Linear(32, 64),
            nn.ReLU(),

            nn.Linear(64, input_dim)
        )

    def forward(self, x):

        latent = self.encoder(x)

        reconstruction = self.decoder(latent)

        return reconstruction
    
class AutoEncoderDetector(BaseDetector):
    def __init__(
        self,
        input_dim,
        lr = AE_LR,
        epochs = AE_EPOCHS,
        batch_size = AE_BATCH_SIZE,
        device = None
    ):
        set_seed(seed= RANDOM_STATE)

        if not device:
            if torch.backends.mps.is_available():
                self.device = torch.device("mps")

            elif torch.cuda.is_available():
                self.device = torch.device("cuda")

            else:
                self.device = torch.device("cpu")
        
        else:
            self.device = torch.device(device)
        
        print(f"using device: {self.device}")


        self.input_dim = input_dim
        self.model = AutoEncoder(
            input_dim= input_dim
        ).to(self.device)

        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=lr
        )

        self.criterion = nn.MSELoss()
        self.threshold = None
        self.loss_history = []


    def fit(self, X):

        X_tensor = torch.FloatTensor(X).to(self.device)
        dataset = TensorDataset(X_tensor)

        generator = torch.Generator()
        generator.manual_seed(RANDOM_STATE)

        train_loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle= True,
            generator= generator
        )

        self.model.train()
        
        for epoch in range(1, self.epochs +1):
            epoch_loss = 0.0

            for (batch,) in train_loader:
                self.optimizer.zero_grad()

                reconstructed = self.model(
                    batch
                )

                loss = self.criterion(
                    reconstructed,
                    batch
                )
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()
            
            epoch_loss /= len(train_loader)
            self.loss_history.append(epoch_loss)

            if epoch % 10 == 0:
                print(
                    f"Epoch: {epoch}/{self.epochs} | "
                    f"Loss: {epoch_loss:.6f}"
                )

        self.model.eval()

        with torch.no_grad():

            reconstructed = self.model(
                X_tensor
            )
            errors = (
                (X_tensor - reconstructed) ** 2
            ).mean(dim=1)

            self.threshold = np.percentile(
                errors.cpu().numpy(),
                ANOMALY_PERCENTILE
            )

        return self
    
    def anomaly_score(self, X):
        self.model.eval()

        X_tensor = (
            torch.FloatTensor(X)
            .to(self.device)
        )

        with torch.no_grad():
            reconstructed = self.model(
                X_tensor
            )

            errors = (
                (X_tensor - reconstructed) ** 2
            ).mean(dim=1)

        return errors.cpu().numpy()

    def predict(self, X):
        scores = self.anomaly_score(X)
        return scores > self.threshold

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        torch.save(
            {
                "model_state_dict":
                    self.model.state_dict(),
                "threshold":
                    self.threshold,
                "loss_history":
                    self.loss_history,
                "input_dim":
                    self.input_dim,
                "lr":
                    self.lr,
                "epochs": 
                    self.epochs,
                "batch_size": 
                    self.batch_size,
            },
            path
        )

    def load(self, path):
        checkpoint = torch.load(
            path,
            map_location=self.device
        )

        self.model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        self.threshold = checkpoint[
            "threshold"
        ]

        self.loss_history = checkpoint.get(
            "loss_history",
            []
        )

        self.model.eval()

        return self
