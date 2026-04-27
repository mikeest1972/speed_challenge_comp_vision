import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import SpeedDataset
from model import SpeedPredictorModel
from tqdm import tqdm
import os

# params
WINDOW_SIZE = 20
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 1e-5

# dataset
print("Loading data set...")
train_dataset = SpeedDataset("../data/precomputed_flows.npy", "../data/train.txt", WINDOW_SIZE, is_train=True)
val_dataset = SpeedDataset("../data/precomputed_flows.npy", "../data/train.txt", WINDOW_SIZE, is_train=False)  
print("Done")
# data loader

print("Loading data loaders")
train_loder = DataLoader(train_dataset,batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset,batch_size=BATCH_SIZE, shuffle=False)
print("Done")
# model
print("Loading model")
CHECKPOINT = "../data/speed_predict_model_4_9.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SpeedPredictorModel()
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
model = model.to(device)
if os.path.exists(CHECKPOINT):
    try:
        model.load_state_dict(torch.load(CHECKPOINT, weights_only=True))
        print(f"Loaded weights from {CHECKPOINT}")
    except Exception as e:
        print(f"Could not load checkpoint: {e}")

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=1, factor=0.5)
best_val_loss =  float(4.9) #float('inf')

for epoch in range(EPOCHS):
    # training
    model.train()
    train_loss = 0
    for flows, lables in tqdm(train_loder, desc=f"Epoch {epoch+1}/{EPOCHS} train"):
        flows, lables = flows.to(device), lables.to(device)
        optimizer.zero_grad()
        predictions = model(flows)
        loss = criterion(predictions.squeeze(), lables.float())
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    #validation
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for flows, lables in tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} val"):
            flows, lables = flows.to(device), lables.to(device)
            predictions = model(flows)
            loss = criterion(predictions.squeeze(), lables.float())
            val_loss += loss.item()
    avg_val_loss = val_loss / len(val_loader)
    print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {train_loss/len(train_loder):.4f} | Val Loss: {avg_val_loss:.4f}")
    scheduler.step(avg_val_loss)
    print(f"  → LR: {optimizer.param_groups[0]['lr']:.2e}")
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), "../data/best_model.pth")
        print(f"  → saved best model (val loss: {best_val_loss:.4f})")

