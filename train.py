from dataset import PlantDiseaseDataset, SetType, CLASS_MAP
import sys
import torch.nn as nn
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from cnn import PlantDiseaseNet
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau


class PlantDiseaseTrainer:
    def __init__(self, model_name: str, epochs: int):
        self.output_filename = model_name + ".pth"
        self.epochs = epochs
        self.train_set = PlantDiseaseDataset(SetType.train, shuffle=True)
        self.train_loader = DataLoader(
            self.train_set, batch_size=128, shuffle=True, num_workers=12
        )

        self.test_set = PlantDiseaseDataset(SetType.test)
        self.test_loader = DataLoader(
            self.test_set, batch_size=128, shuffle=False, num_workers=1
        )

        self.val_set = PlantDiseaseDataset(SetType.val, shuffle=True)
        self.val_loader = DataLoader(
            self.val_set, batch_size=64, shuffle=False, num_workers=12
        )

        self.config = {
            "starting_lr": 1e-3,
            "momentum": 0.9,
            "decay": 5e-4,
            "patience": 15,
            "lr_factor": 0.3,
            "print_cadence": 100,
        }

        self.device = torch.device("cuda:0")
        self.net = PlantDiseaseNet().to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(
            self.net.parameters(),
            lr=self.config["starting_lr"],
            weight_decay=self.config["decay"],
        )
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            factor=self.config["lr_factor"],
            mode="max",
            verbose=True,
            patience=self.config["patience"],
        )

        self.writer = SummaryWriter(f"{model_name}_logs")

        for ds in [self.train_set, self.test_set, self.val_set]:
            print(f"Size of datasets: {len(ds)}")
        print("Trainer Initialized.")

    def train(self):
        log_iter = 0
        for epoch in range(self.epochs):
            running_loss = 0.0
            for i, (inputs, labels, meta) in enumerate(self.train_loader):
                self.net.train()
                self.optimizer.zero_grad()
                outputs = self.net(inputs.float().to(self.device))
                loss = self.criterion(outputs, labels.long().to(self.device))
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
                if i > 0 and i % self.config["print_cadence"] == 0:
                    mean_loss = running_loss / self.config["print_cadence"]
                    self.writer.add_scalar("Train/MRLoss", mean_loss, log_iter)
                    print(f"Epoch: {epoch}\tBatch: {i}\tLoss: {mean_loss}")
                    running_loss = 0.0
                    log_iter += 1

            train_acc = self.log_metrics(epoch, "Train")
            self.log_metrics(epoch, "Validation")
            self.scheduler.step(train_acc)
        acc = self.calculate_accuracy(self.test_loader)
        self.writer.add_text("Test/Accuracy", f"{acc}")

    def log_metrics(self, epoch, label):
        loader = self.val_loader if label == "Validation" else self.train_loader
        acc = self.calculate_accuracy(loader)
        self.writer.add_scalar(f"{label}/Accuracy", acc, epoch)
        return acc

    def calculate_accuracy(self, loader: DataLoader):
        with torch.no_grad():
            self.net.eval()
            correct = 0.0
            total = 0.0
            for inputs, labels, metadata in loader:
                outputs = self.net(inputs.float().to(self.device))
                _, preds = outputs.detach().cpu().max(1)
                total += labels.size(0)
                correct += preds.eq(labels.float()).sum().item()
        print(f"Correct:\t{correct}, Incorrect:\t{total-correct}")
        return correct / total


if __name__ == "__main__":
    trainer = PlantDiseaseTrainer(sys.argv[1], 425)
    trainer.train()
    torch.save(trainer.net.state_dict(), trainer.output_filename)
