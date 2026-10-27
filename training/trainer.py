import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split


from training.baselinecnn import GalaxyBaseline
from datasets.galaxy10_dataset import Galaxy10Dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = GalaxyBaseline().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)

epochs = 20

dataset = Galaxy10Dataset('C:\\Users\\phelpsrw\\OneDrive - Rose-Hulman Institute of Technology\\Desktop\\DeepLearning\\DeepLearning\\data\\Galaxy10_DECals.h5')

#80/20 split
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size

#split the dataset
train_dataset, test_dataset = random_split(
    dataset,
    [train_size,test_size],
    generator=torch.Generator().manual_seed(42)
)

train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True,
    num_workers=0
)


print("Starting trianing")
for epoch in range(epochs):
    model.train()
    
    running_loss = 0.0 
    correct = 0
    total = 0
    
    for batch_idx, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device).long()
        
        optimizer.zero_grad()
        
        outputs = model(images)
        
        loss = criterion(outputs, labels)
        
        loss.backward()
        
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
        
        predictions = outputs.argmax(dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)
        
        if (batch_idx + 1) % 2 == 0:
            print(
                f"Epoch [{epoch+1}/{epochs}] "
                f"Batch [{batch_idx+1}/{len(train_loader)}] "
                f"Loss: {running_loss/total:.4f} "
                f"Accuracy: {correct/total:.2%}",
                flush=True
            )

        
    epoch_loss = running_loss / total
    epoch_accuracy = correct / total
    
    print(
        f"Epoch [{epoch + 1}/{epochs}] "
        f"Loss: {epoch_loss:.4f} "
        f"Accuracy: {epoch_accuracy:.2%}"
    )

    torch.save(model.state_dict(), "C:\\Users\\phelpsrw\\OneDrive - Rose-Hulman Institute of Technology\\Desktop\\DeepLearning\\DeepLearning\\models\\baseline.pth")
    print("Model saved successfully!")  
    
        
