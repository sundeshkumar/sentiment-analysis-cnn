import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt  
from model import TextCNN
from dataset import MRDataset

# --- Evaluation Helper Function ---
def evaluate(model, loader, criterion):
    model.eval()
    correct = 0
    total = 0
    total_loss = 0
    
    with torch.no_grad():
        for data, target in loader:
            output = model(data)
            loss = criterion(output, target.long())
            total_loss += loss.item()
            
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            
    accuracy = 100 * correct / total
    return total_loss / len(loader), accuracy

def train_static():
    # --- Configuration ---
    POS_FILE = "C:/Users/sunde/Desktop/TextCNN_Project/data/rt-polarity.pos"
    NEG_FILE = "C:/Users/sunde/Desktop/TextCNN_Project/data/rt-polarity.neg"
    W2V_PATH = "C:/Users/sunde/Desktop/TextCNN_Project/data/GoogleNews-vectors-negative300.bin"
    BATCH_SIZE = 50
    EPOCHS = 5
    LR = 0.001

    # 1. Prepare Data
    print("--- Preparing Data for CNN-Static ---")
    dataset = MRDataset(POS_FILE, NEG_FILE, W2V_PATH)
    
    train_size = int(0.9 * len(dataset))
    test_size = len(dataset) - train_size
    train_data, test_data = random_split(dataset, [train_size, test_size])
    
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)
    
    # 2. Initialize Model
    model = TextCNN(vocab_size=len(dataset.vocab), embed_dim=300, num_classes=2)
    
    # --- CNN-STATIC CONFIGURATION ---
    print("[Info] Loading pre-trained weights...")
    model.embedding.weight.data.copy_(dataset.embedding_matrix)
    model.embedding.weight.requires_grad = False 
    print("[Status] Embeddings frozen (Static Mode).")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    
    # --- LISTS TO STORE DATA FOR GRAPHS ---
    train_losses = []  
    test_accuracies = [] 

    print("\n--- Starting Training Loop ---")
    
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target.long())
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        # Calculate Average Loss for this epoch
        epoch_loss = total_loss / len(train_loader)
        
        # Evaluate on test set
        val_loss, val_acc = evaluate(model, test_loader, criterion)
        
        # --- SAVE DATA TO LISTS ---
        train_losses.append(epoch_loss)   
        test_accuracies.append(val_acc) 
        
        print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {epoch_loss:.4f} | Test Accuracy: {val_acc:.2f}%")

    print("\nTraining Complete!")
    
    # 3. Save the trained model
    SAVE_PATH = "cnn_static_mr.pth"
    torch.save(model.state_dict(), SAVE_PATH)
    print(f"Model weights saved to {SAVE_PATH}")

    # --- 4. GENERATE AND SAVE GRAPHS ---
    print("--- Generating Graphs ---")
    
    plt.figure(figsize=(12, 5)) 

    # Graph 1: Loss Curve
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss', color='red')
    plt.title('Training Loss over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # Graph 2: Accuracy Curve
    plt.subplot(1, 2, 2)
    plt.plot(test_accuracies, label='Test Accuracy', color='green')
    plt.title('Test Accuracy over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)

    # Save the graph as an image file
    plt.savefig('training_results_static.png')
    print("Graph saved as 'training_results_static.png' ✅")
    plt.show()

if __name__ == "__main__":
    train_static()