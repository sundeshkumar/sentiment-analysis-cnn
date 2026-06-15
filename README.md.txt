# Sentiment Analysis using TextCNN (PyTorch)

This project is a reproduction of the research paper: **"Convolutional Neural Networks for Sentence Classification"** by Yoon Kim (2014).
It implements a CNN architecture designed for natural language tasks, specifically sentiment classification.

## 👥 Contributors
* **Sundesh Kumar**
* **Ifrah Mukarram**

---

## 🏗️ Architecture Overview
The model utilizes multiple convolutional filters of different window sizes (3, 4, and 5) to capture various n-gram features from the text.
 After convolution, max-over-time pooling is applied to extract the most significant features, which are then passed through a fully connected layer with dropout for classification.
---

## 📂 Project Structure

| File Name | Description |
| :--- | :--- |
| `model.py` | The core CNN architecture (Filters: 3,4,5 | Feature Maps: 100). |
| `dataset.py` | Data processing script (Cleaning, Vocabulary, and Word2Vec integration). |
| `train_static.py` | Script to train the **CNN-Static** variant (Frozen embeddings). |
| `train_non_static.py` | Script to train the **CNN-Non-Static** variant (Fine-tuned embeddings). |
| `Check_CNN_Ar.py` | A validation script to verify model dimensions and data flow. |
| `requirements.txt` | List of necessary Python libraries. |

---

## 🛠️ Installation & Setup

1. **Install Dependencies:**
   Run the following command in your terminal:
   ```bash
   pip install torch numpy gensim nltk matplotlib
               OR
   pip install -r requirements.txt

2. Create Data Folder: Create a folder named data in your project directory to store the dataset.

   📥 Data Download Links
1. Movie Review (MR) Dataset
Download both files and save them inside the data/ folder:

Positive Reviews: https://gitlab.rlp.net/arti_ws2526/sentenceclassifiction_ifrahsundesh/-/blob/main/data/rt-polarity.pos

Negative Reviews: https://gitlab.rlp.net/arti_ws2526/sentenceclassifiction_ifrahsundesh/-/blob/main/data/rt-polarity.neg

2. Pre-trained Word2Vec (Google News)
Download Link: https://www.kaggle.com/datasets/leadbest/googlenewsvectorsnegative300

Note: Extract the file and ensure it is named GoogleNews-vectors-negative300.bin inside your data/ folder.

🚀 How to Run
Step 1: Verification
Check if the model architecture is correct:

Bash

python Check_CNN_Ar.py

Step 2: Training (Static Model)
Train the model with fixed Google News vectors:

Bash

python train_static.py
Output: Model saved as cnn_static_mr.pth.

Visualization: Graph saved as training_results_static.png.

Step 3: Training (Non-Static Model)
Train the model while fine-tuning the word vectors:

Bash

python train_non_static.py
Output: Model saved as cnn_non_static_mr.pth.

Visualization: Graph saved as training_results_non_static.png.

Zabardast! Maine aapki di hui context ko update kar diya hai. Maine ismein predict.py ka section, testing sentences, aur project structure table ko mazeed behtar banaya hai.

Aap bas neeche wala code block copy karein aur apni README.md file mein paste kar dein:

Markdown

# Sentiment Analysis using TextCNN (PyTorch)

This project is a reproduction of the research paper: **"Convolutional Neural Networks for Sentence Classification"** by Yoon Kim (2014). It implements a CNN architecture designed for natural language tasks, specifically sentiment classification.

## 👥 Contributors
* **Sundesh Kumar**
* **Ifrah Mukarram**

---

## 🏗️ Architecture Overview
The model utilizes multiple convolutional filters of different window sizes (3, 4, and 5) to capture various n-gram features from the text. After convolution, max-over-time pooling is applied to extract the most significant features, which are then passed through a fully connected layer with dropout for classification.



---

## 📂 Project Structure

| File Name | Description |
| :--- | :--- |
| `model.py` | The core CNN architecture (Filters: 3, 4, 5 | Feature Maps: 100). |
| `dataset.py` | Data processing script (Cleaning, Vocabulary, and Word2Vec integration). |
| `predict.py` | Live testing script to predict sentiment for new sentences. |
| `train_static.py` | Script to train the **CNN-Static** variant (Frozen embeddings). |
| `train_non_static.py` | Script to train the **CNN-Non-Static** variant (Fine-tuned embeddings). |
| `Check_CNN_Ar.py` | A validation script to verify model dimensions and data flow. |
| `requirements.txt` | List of necessary Python libraries. |

---

## 🛠️ Installation & Setup

1. **Install Dependencies:**
   Run the following command in your terminal:
   ```bash
   pip install torch numpy gensim nltk matplotlib
OR

Bash

pip install -r requirements.txt
Create Data Folder: Create a folder named data in your project directory to store the dataset and word vectors.

📥 Data Download Links
1. Movie Review (MR) Dataset
Download both files and save them inside the data/ folder:

Positive Reviews: rt-polarity.pos

Negative Reviews: rt-polarity.neg

2. Pre-trained Word2Vec (Google News)
Download Link: Kaggle - Google News Vectors

Note: Extract the file and ensure it is named GoogleNews-vectors-negative300.bin inside your data/ folder.

🚀 How to Run
Step 1: Verification
Check if the model architecture is correct:

Bash

python Check_CNN_Ar.py
Step 2: Training (Static Model)
Train the model with fixed Google News vectors:

Bash

python train_static.py
Output: Model saved as cnn_static_mr.pth.

Visualization: Graph saved as training_results_static.png.

Step 3: Training (Non-Static Model)
Train the model while fine-tuning the word vectors:

Bash

python train_non_static.py
Output: Model saved as cnn_non_static_mr.pth.

Visualization: Graph saved as training_results_non_static.png.

Step 4: Prediction (Live Testing)
Use the trained model to predict the sentiment of any custom sentence:

Bash

python predict.py
Example Sentences to try:

Positive: "The plot was brilliant and the acting was superb."

Negative: "The movie was a total waste of time and very boring."

📊 Results ComparisonModel 
Variant              Strategy                               Accuracy         

CNN-Static           Pre-trained vectors are frozen           ~75%           
CNN-Non-Static       Pre-trained vectors are fine-tuned       ~79%