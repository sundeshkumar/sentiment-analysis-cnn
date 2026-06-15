import torch
from dataset import MRDataset, clean_str
from model import TextCNN
import sys

# Change this path to switch between static and non-static models
MODEL_PATH = "cnn_static_mr.pth"

def live_test():
    # Configuration paths
    POS_FILE = "C:/Users/sunde/Desktop/TextCNN_Project/data/rt-polarity.pos"
    NEG_FILE = "C:/Users/sunde/Desktop/TextCNN_Project/data/rt-polarity.neg"
    W2V_PATH = "C:/Users/sunde/Desktop/TextCNN_Project/data/GoogleNews-vectors-negative300.bin"
    
    print("--- Initializing Live Testing Environment ---")
    print("[Step 1/2] Loading Vocabulary and Word2Vec (This may take a moment)...")
    # We need to load the dataset again to ensure the vocabulary mapping is identical.
    try:
        d = MRDataset(POS_FILE, NEG_FILE, W2V_PATH)
        vocab = d.vocab
    except FileNotFoundError:
        print("\n[Error] Data files or Word2Vec not found. Please check paths.")
        sys.exit(1)
    
    print(f"[Step 2/2] Loading Saved Model Weights from: {MODEL_PATH}")
    try:
        model = TextCNN(vocab_size=len(vocab), embed_dim=300, num_classes=2)
        model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
        model.eval() # Set to evaluation mode
    except FileNotFoundError:
         print(f"\n[Error] Model file '{MODEL_PATH}' not found. Train the model first.")
         sys.exit(1)
    
    print("\n" + "="*50)
    print("      LIVE SENTIMENT ANALYSIS SYSTEM      ")
    print(f"      Model Loaded: {MODEL_PATH}")
    print("      (Type 'exit' to quit)              ")
    print("="*50 + "\n")

    while True:
        # Get user input
        user_input = input(">> Enter a sentence: ")
        
        if user_input.lower() in ['exit', 'quit']:
            print("Exiting...")
            break
        
        # Preprocess input
        cleaned_sent = clean_str(user_input)
        words = cleaned_sent.split()
        
        # Convert to indices based on loaded vocabulary
        indices = [vocab.get(w, 1) for w in words]
        
        # Pad to required length (60)
        max_len = 60
        if len(indices) < max_len:
            indices += [0] * (max_len - len(indices))
        else:
            indices = indices[:max_len]
            
        # Convert to tensor and add batch dimension
        tensor_input = torch.tensor([indices]) # Shape: [1, 60]
        
        # Perform prediction
        with torch.no_grad():
            output = model(tensor_input)
            prediction = torch.argmax(output, dim=1).item()
            
            # Calculate confidence score using Softmax
            probs = torch.nn.functional.softmax(output, dim=1)
            confidence = probs[0][prediction].item() * 100

        # Display result
        if prediction == 1:
            print(f"[Result] POSITIVE Sentiment (Confidence: {confidence:.2f}%)")
        else:
            print(f"[Result] NEGATIVE Sentiment (Confidence: {confidence:.2f}%)")
        print("-" * 40)

if __name__ == "__main__":
    live_test()