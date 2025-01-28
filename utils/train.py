import torch
import torch.optim as optim
import torch.nn as nn
from vit_model import VisionTransformer  # Replace with your actual ViT class
from dataset_loader import FabricDataset
import torch.utils.data as data

# Set up device (GPU or CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create dataset and DataLoader
train_dataset = FabricDataset(data_dir='fabric_data/train')
train_loader = data.DataLoader(train_dataset, batch_size=32, shuffle=True)

# Initialize the Vision Transformer model
model = VisionTransformer().to(device)

# Set up optimizer and loss function
optimizer = optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()  # Change based on your model's output

# Training loop
epochs = 10
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for images in train_loader:
        images = images.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)  # Forward pass
        loss = criterion(outputs, labels)  # Compute loss (adjust labels as needed)
        loss.backward()  # Backpropagate
        optimizer.step()  # Update weights
        
        running_loss += loss.item()
    
    print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader)}")
    
    # Optionally, save the model after every epoch
    torch.save(model.state_dict(), "models/vit_fabric_model.pth")
