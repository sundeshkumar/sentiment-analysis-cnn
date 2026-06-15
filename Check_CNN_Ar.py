import torch
from model import TextCNN  # Importing the TextCNN class from the model file

def main():
    print("--- Checking: Starting ---")
    
    # 1. Set Hyperparameters
    VOCAB_SIZE = 5000
    EMBED_DIM = 300
    NUM_CLASSES = 2
    
    # 2. Initialize the Model
    try:
        model = TextCNN(VOCAB_SIZE, EMBED_DIM, NUM_CLASSES)
        print("[OK] Model class successfully initialized.")
    except Exception as e:
        print(f"[FAIL] Error initializing model: {e}")
        return

    # 3. Create Dummy Data (Batch Size=50, Sequence Length=60)
    dummy_input = torch.randint(0, VOCAB_SIZE, (50, 60))
    print(f"[INFO] Input shape: {dummy_input.shape}")

    # 4. Run Forward Pass
    try:
        output = model(dummy_input)
        print(f"[INFO] Output shape: {output.shape}")
        
        if output.shape == (50, 2):
            print("\nSUCCESS! Task Complete.")
            print("Output shape [50, 2] matches the expected dimensions.")
        else:
            print("\nWARNING: Incorrect output shape.")
            
    except Exception as e:
        print(f"[FAIL] Error during forward pass: {e}")

if __name__ == "__main__":
    main()