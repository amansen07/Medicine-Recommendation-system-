# Medicine Recommendation System

**Name:** Aman sen  
**Reg. No:** 26BCY10107  
**Branch:** CSE (Cyber Security and Digital Forensics)  
**Course:** Cyber Security and Digital Forensics  
**Year:** 1st Year — VIT Bhopal  

---

## 🌟 Why I Built This
I noticed that a lot of people around me, especially in smaller towns, don't really know what medicine to take when they fall sick. They either guess, ask a random person, or just ignore it. I thought — what if there was something simple that takes your symptoms and at least points you in the right direction? So I built this. It's not perfect, but it works better than I expected.

## 🛠️ What It Does
- Asks for your name and age.
- Shows a numbered list of symptoms.
- You pick the ones you have by entering their numbers.
- The ML model predicts the most likely disease.
- Shows recommended medicines and precautions.
- Displays a confidence chart so you know how sure the AI is.
- Option to save the full result as a text file.

## 🧠 AIML Concepts Used
- Supervised Learning
- Random Forest Classifier
- Decision Tree Classifier
- Feature Engineering (binary symptom vectors)
- Model comparison and evaluation
- Prediction with probability/confidence scoring

---

## 🚀 How to Run It
Make sure Python is installed on your system first.

### Step 1 — Clone or download the project
```bash
git clone https://github.com/YOUR-USERNAME/medicine-recommender
cd medicine-recommender
```

### Step 2 — Install the required libraries
```bash
pip install -r requirements.txt
```

### Step 3 — Train the model (do this once)
```bash
python train_model.py
```
*This creates two files — `model.pkl` and `symptoms_list.pkl`. You only need to do this once.*

### Step 4 — Run the app
```bash
python app.py
```
*Just open the folder in VS Code, open a new terminal, and type `python app.py` — it'll start immediately and ask for your name.*

---

## 📂 Files in this Project
```text
medicine-recommender/
├── app.py                    # the main app you run
├── train_model.py            # trains and saves the ML model
├── disease_symptoms.csv      # dataset with diseases and symptoms
├── medicines_precautions.csv # dataset with medicines and precautions
├── requirements.txt          # libraries needed
├── model.pkl                 # saved model (created after training)
├── symptoms_list.pkl         # saved symptom list (created after training)
└── README.md
```

---

## 💻 Sample Output
```text
╔══════════════════════════════════════════════════════╗
║        💊  MEDICINE RECOMMENDATION SYSTEM  💊        ║
║             AI / ML  Based  Disease  Predictor       ║
╚══════════════════════════════════════════════════════╝

  👤 Enter your name : Ayush
  🎂 Enter your age  : 20

  Hello Ayush! 👋

  🔢 Symptom numbers: 3, 7, 15

  ✅ Symptoms selected:
     • Chills
     • High Fever
     • Headache

  🔴 Predicted Disease : Malaria

  AI Confidence Chart:
  Malaria       ██████████████████████░░░░░░░░  88.5%  ◀ TOP MATCH

  💊 Recommended Medicines:
     💊 Chloroquine
     💊 Artemisinin
     💊 Doxycycline

  🛡️ Precautions:
     1. Consult nearest hospital
     2. Avoid oily food
     3. Keep mosquitos away

  💾 Save this report to a file? (yes/no):
```

---

## 📊 ML Models Compared
| Model | Accuracy |
|-------|----------|
| Decision Tree | ~95% |
| Random Forest | ~97.56% |

*Random Forest performed better so that's the one that gets saved and used automatically.*

## ⚙️ Tech Used
- Python 3
- scikit-learn
- pandas
- numpy
- pickle

## 🔮 Future Improvements
- Web or mobile interface
- Larger dataset with more diseases
- Voice input for symptoms
- Multi-language support

## 📈 Learning Outcomes
This project helped me understand how ML models are trained, compared and deployed in a real use case. I learnt how to convert raw data into binary feature vectors, evaluate two different classifiers, and build a complete CLI application around an ML model.

---
**Author:** Aman sen -- VIT Bhopal, Cyber Security and Digital Forensics
