# Sentiment Analysis on IMDb Movie Reviews 🎬

**Author:** Berkcan Özen  

This project applies Natural Language Processing (NLP) techniques to IMDb movie reviews to predict whether a review is **Positive** or **Negative**.  

## 📌 Project Overview
The global film industry faces challenges in predicting audience reactions. This project explores sentiment analysis to support data-driven decisions for stakeholders.  

We experimented with:
- **Positive/Negative Sentiment Analysis** (Naive Bayes Classifiers)
- **Six-Emotion Classification** (Happiness, Sadness, Anger, Fear, Love, Surprise)

The Positive/Negative model achieved the highest accuracy (~94% with Multinomial Naive Bayes).  

## ⚙️ Methodology
1. **Data Collection**  
   - Web scraping IMDb reviews using Python (`requests`, `BeautifulSoup`)  
   - Automatic sentiment labeling (Positive if rating ≥ 6, Negative if < 6)

2. **Data Preprocessing**  
   - Remove HTML tags & special characters  
   - Lowercasing & tokenization  
   - Stopword removal & stemming  
   - Bag-of-Words representation with `CountVectorizer`

3. **Modeling**  
   - Gaussian NB → Accuracy ~62%  
   - Multinomial NB → Accuracy ~94%  
   - Bernoulli NB → Accuracy ~93%  

4. **Testing on New Reviews**  
   The model predicts unseen reviews with good accuracy but struggles with sarcasm and negations.  

## 📊 Findings
- Multinomial Naive Bayes performed best.  
- Sarcasm and irony remain challenges.  
- Larger and more diverse datasets can improve robustness.  

## 🛠️ Requirements
Install dependencies with:
```bash
pip install -r requirements.txt
```

## 📂 Repository Structure
```
sentiment-analysis-imdb/
│
├── notebooks/       # Jupyter/Colab notebooks with code
├── reports/         # Project report in Markdown
├── data/            # Placeholder for datasets
├── README.md        # Project description
├── requirements.txt # Python dependencies
└── .gitignore       # Ignore unnecessary files
```

## 🚀 Usage
1. Run the notebook in `notebooks/Sentiment_Analysis_IMDB.ipynb`.  
2. Upload your IMDb review dataset.  
3. Train the model and test predictions on new reviews.  

## 📖 Report
The full report is available in [`reports/Sentiment_Analysis_Report.md`](reports/Sentiment_Analysis_Report.md).  

---
