import models.ECGFounder.finetune_model as ECGFound
import torch
from program.training.data_transformer import DataTransformer
from program.training.ecg_dataset import ECGDataset
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

class FinetuneModels:
    def __init__(self):
        self.device = torch.device('cuda:{}'.format(0) if torch.cuda.is_available() else 'cpu')
        pass

    def finetune_ECGFounder_activity(self, activities: int, linear_prob=False):
        model = ECGFound.ft_1lead_ECGFounder(self.device, "./checkpoints/1_lead_ECGFounder.pth", activities, linear_prob=linear_prob)
        training = DataTransformer("./data/PPG-Sport/converted_data", "cnn", 64, ["time", "acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z", "ecg", "ppg", "temp", "eda"])
        (X_train, y_train), (X_val, y_val), (X_test, y_test) = training.get_ecg_data()

        X_train = X_train.swapaxes(1,2)
        X_val = X_val.swapaxes(1,2)
        X_test = X_test.swapaxes(1,2)



        train_dataset = ECGDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)

        val_dataset = ECGDataset(X_val, y_val)
        val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)

        test_dataset = ECGDataset(X_test, y_test)
        test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

        self.train_model(model, train_loader, val_loader, self.device, n_epochs=10)
        all_preds, all_labels = self.evaluate(model, test_loader, self.device)
        self.plot_confusion_matrix(all_preds, all_labels)





    def train_model(self, model, train_loader, val_loader, device, n_epochs=20, lr=1e-4, patience=10, save_path="best_model.pth"):
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=patience)

        best_val_acc = 0.0

        for epoch in range(n_epochs):
            model.train()
            total_loss = 0
            for x_batch, y_batch in train_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                optimizer.zero_grad()
                outputs = model(x_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            model.eval()
            correct, total = 0, 0
            with torch.no_grad():
                for x_batch, y_batch in val_loader:
                    x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                    outputs = model(x_batch)
                    preds = outputs.argmax(dim=1)
                    correct += (preds == y_batch).sum().item()
                    total += y_batch.size(0)
            val_acc = correct / total

            scheduler.step(val_acc)

            print(f"Epoch {epoch+1}: Train Loss={total_loss/len(train_loader):.4f}, Val Acc={val_acc:.4f}")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), save_path)

        # --- load the best checkpoint before returning ---
        model.load_state_dict(torch.load(save_path, map_location=device))
        print(f"Loaded best model with Val Acc={best_val_acc:.4f}")

        return model

    def evaluate(self, model, test_loader, device):
        model.eval()
        correct, total = 0, 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for x_batch, y_batch in test_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                outputs = model(x_batch)
                preds = outputs.argmax(dim=1)

                correct += (preds == y_batch).sum().item()
                total += y_batch.size(0)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(y_batch.cpu().numpy())

        accuracy = correct / total
        print(f"Test Accuracy: {accuracy:.4f}")

        return all_preds, all_labels

    def plot_confusion_matrix(all_labels, all_preds, class_names=None):
        cm = confusion_matrix(all_labels, all_preds)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
        fig, ax = plt.subplots(figsize=(8, 8))
        disp.plot(ax=ax, cmap="Blues", xticks_rotation=45, colorbar=True)
        plt.title("Confusion Matrix")
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    finetune = FinetuneModels()
    finetune.finetune_ECGFounder_activity(6)
