
# **Fake News Detection Using Passive-Aggressive Classifier with Hyperparameter Tuning**

This project implements a **Fake News Detection System** using machine learning. The system classifies news articles as "Real" or "Fake" based on their textual content. It incorporates **TF-IDF Vectorization**, a **Passive-Aggressive Classifier**, and **hyperparameter tuning** to enhance performance.

---

## **Table of Contents**
- [Introduction](#introduction)
- [Dataset](#dataset)
- [Technologies Used](#technologies-used)
- [Workflow](#workflow)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Contributions](#contributions)
- [Future Work](#future-work)
- [License](#license)

---

## **Introduction**
Fake news poses a significant challenge in today's digital era. With the rapid spread of misinformation, an efficient system to detect fake news is essential. This project uses a machine learning approach to analyze and classify news articles.

---

## **Dataset**
- **Source:** [Dataset Name/Provider](#) (Replace with actual dataset source if applicable)
- The dataset contains labeled news articles, with labels `1` for fake news and `0` for real news.
- **Preprocessing Steps:**
  - Dropped unnecessary columns like `author`, `title`, and `id`.
  - Removed null values and reset the dataset index.
  - Applied lemmatization and stopword removal to clean the text data.

---

## **Technologies Used**
- **Languages:** Python
- **Libraries:** 
  - Machine Learning: `scikit-learn`
  - Text Processing: `nltk`, `re`
  - Visualization: `matplotlib`, `seaborn`
  - Utilities: `pandas`, `numpy`, `joblib`

---

## **Workflow**

![Workflow Diagram](image/process.webp) 

The project follows these steps:
1. **Data Cleaning**: Remove unnecessary columns and handle missing values.
2. **Text Preprocessing**: Apply lemmatization and stopword removal.
3. **Dataset Splitting**: Divide data into training (80%) and testing (20%).
4. **TF-IDF Vectorization**: Convert textual data into numerical features.
5. **Model Training**: Train a Passive-Aggressive Classifier.
6. **Hyperparameter Tuning**: Optimize the classifier using GridSearchCV.
7. **Evaluation**: Measure performance with metrics like accuracy, precision, recall, and F1-score.

---

## **Installation**
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/fake-news-detection.git
   ```
2. Navigate to the project directory:
   ```bash
   cd fake-news-detection
   ```
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## **Usage**
1. Prepare the dataset by placing it in the `data/` directory.
2. Run the Jupyter notebook or the Python script to train the model:
   ```bash
   python main.py
   ```
3. Test the system using the saved model:
   ```python
   python predict.py
   ```

---

## **Results**
### **Original Model Metrics:**
- **Accuracy:** 96%
- **Precision:** 96%
- **Recall:** 95%

### **Hyperparameter Tuned Model Metrics:**
- **Accuracy:** 96%
- **Precision:** 96%
- **Recall:** 96%

![Classification](/Classification_Report.png)

Although the metrics remained the same, hyperparameter tuning validated the optimal settings of the model.

---

## **Contributions**
- Applied **hyperparameter tuning** using GridSearchCV for systematic optimization.
- Integrated a predictive system for real-time fake news classification.
- Saved the trained model and vectorizer for reuse with `joblib`.

![Hyperparameters](Hyperparameters.png)

---

## **Future Work**
- Experiment with advanced NLP models like **BERT** for improved accuracy.
- Test the model on larger and more diverse datasets.
- Extend the system to multilingual fake news detection.

---

