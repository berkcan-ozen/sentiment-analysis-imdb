Sentiment Analysis on IMDB Movie Reviews

**Author:** Berkcan Özen


Executive Summery

This project focuses on sentiment analysis for movie comments sourced from IMDB, with the aim of providing actionable insights for stakeholders in the film industry. By annotating comments and developing machine learning models, we sought to accurately predict sentiment as positive or negative. On this project, we experimented with two models, one for six emotions wheel sentiment analysis and the other for positive-negative sentiment analysis. The positive-negative sentiment analysis model, built using Naive Bayes classifiers, displayed higher accuracy rates compared to the six-wheel emotion classification model. Our findings underscore the importance of context sensitivity in sentiment analysis, particularly in identifying sarcasm and negation terms. While the positive-negative model reached high accuracy, challenges in accurately classifying sentiments across diverse linguistic expressions remain. This project involves cleaning and preprocessing and distant annotation of the data and developing a machine learning model to predict the sentiment (positive or negative) of new movie comments based on the annotated dataset.

Introduction

The global film industry, a cultural and economic powerhouse that is generating substantial revenue while captivating audiences worldwide, faces the challenge of accurately predicting a film's box office success. Traditionally, stakeholders have relied on intuition and experience to gauge audience reaction—a method fraught with financial risk, as evidenced by high-profile flops underscoring the need for a more reliable approach to decision-making. This project introduces a sentiment analysis solution aimed at providing actionable insights to movie business stakeholders. By leveraging machine learning techniques to analyze sentiments expressed in movie comments sourced from platforms like IMDB, it offers a comprehensive understanding of audience reactions and preferences. This analysis aims to empower studios, distributors, theaters, talent agencies, and other industry players to make informed decisions throughout the filmmaking process. Through the sentiment analysis model, stakeholders will gain the ability to tailor marketing strategies, optimize release schedules, and refine content to resonate more effectively with target audiences. The project aims to provide data driven insights for a new era of financial security and informed decision-making in the film industry, benefiting all stakeholders involved in the cinematic journey.


Annotating data for sentiment analysis in movie comments is important for enhancing the accuracy and effectiveness of machine learning models in the movie industry. By assigning sentiment categories such as positive and negative evident by the ratings next to the movie comments, the annotated data serves as a reference for training machine learning models. This enables the models to learn and understand the connections between the linguistic elements present in the comments (e.g., word choices, syntax, and semantics) and the corresponding sentiment expressions. In this project, we will use the annotated data to provide the model with labeled examples that allow them to recognize patterns and association between the language used in the comments and the underlying sentiments conveyed. The learning process is essential for creating effective sentiment analysis models. The models are designed to accurately predict the sentiment (positive, negative) of new, previously unseen movie comments based on their linguistic characteristics, such as the words, phrases, and sentence structure used.

On this project, given that we had IMDB scores conveniently placed next to the comments, we collectively determined not to run the Kappa test and instead move forward with a method known as distant annotation in our positive-negative sentiment analysis model. This decision was made in light of the readily available IMDB ratings accompanying the comments, which allowed us to streamline our annotation process and focus directly on the subsequent steps without the need for an additional assessment like the Kappa test.

Responsible Use

In our project, ethical considerations were of top importance, particularly regarding data privacy and transparency. To uphold privacy, we ensured that all movie reviews used in our analysis were anonymized, protecting the identities of reviewers. Additionally, we prioritized transparency throughout our methodology, ensuring clear documentation of data collection, preprocessing steps, and model development. For example, we provided detailed explanations of how we collected data from IMDB and annotated reviews, promoting transparency in our processes. Moreover, we responsibly used our machine learning model by acknowledging its limitations and potential biases, fostering a culture of ethical AI. By incorporating privacy measures and maintaining transparency, we upheld ethical standards in our data analysis, promoting trust and accountability in our project.

Methodology

In our exploration of sentiment analysis, we experimented with two different models: positive-negative sentiment analysis and a six-wheel sentiment analysis. Our decision to prioritize the positive-negative model was influenced by its higher accuracy rate, particularly in the context of our limited data sources. This model showed substantial performance in predicting positive and negative sentiments, aligning more closely with our datasets. Thus, we focused our efforts on refining the positive-negative sentiment analysis. However, we included the results of the six-wheel emotion chart in our methodology to provide a comprehensive overview of our model selection process.

This paper's structure was designed to reflect the outcomes and limitations of the positive-negative and six-wheel sentiment analyses. Section 2.1 delves into the nuances and findings of the positive-negative analysis, emphasizing its relevance and implications. Additionally, Section 2.2 serves as an extension of our exploration, detailing the methodology and results of the six-wheel chart analysis. This segmentation underscores the significance of our model selection process while transparently presenting the outcomes of our experimentation.

2.1. Negative-Positive Sentiment Analysis Methodology

In this project, we scraped 25 reviews for each of the top 250 movies from IMDB's review pages. By doing so, we compiled a dataset containing 6250 reviews along with their corresponding movie titles, ratings, and reviews. All the URLs were taken manually for creating the dataset.

Data Collection / Scraping

The process involved using web scraping techniques to extract data from IMDB's website. We utilized Python libraries such as requests, BeautifulSoup, and csv to automate the process of fetching the HTML content, parsing it, and extracting the relevant information.

Input:

Applying Distant Data Annotation

The code employs distant data annotation to assign sentiment labels to reviews based on the ratings next to them. For each review item found in the parsed HTML, the code identifies the rating associated with the review. If a rating is present and exceeds a threshold of 6, indicating a generally positive sentiment, the review is labeled as 'Positive'. Conversely, if the rating falls below 6, suggesting a less favorable sentiment, the review is labeled as 'Negative'. In cases where no rating is found, both the rating and sentiment are labeled as 'N/A'. This approach showcases distant data annotation by leveraging external ratings to automatically categorize reviews, eliminating the need for direct human input in the sentiment labeling process.

To annotate sentiment, we assign labels based on the magnitude of the rating. Our guideline specifies that if the rating is greater than or equal to 6, the sentiment is labeled as 'Positive'; otherwise, it is labeled as 'Negative'. This decision threshold is derived from the common practice in movie ratings where scores above 6 typically indicate favorable opinions, while scores below 6 suggest less favorable ones.

The justification for using a rating threshold of 6 or higher to indicate positive sentiment is rooted in empirical observation and industry standards. Across various review platforms, including IMDB, ratings above 6 commonly reflect positive viewer opinions and enjoyment of the movie. Therefore, setting the threshold at 6 provides a pragmatic and intuitive criterion for distinguishing positive sentiments from negative ones based on the available rating data.

Data Cleaning and Preprocessing

This process leverages the Pandas library to manipulate a CSV dataset containing movie reviews and their associated ratings. Initially, the script loads the dataset into a Pandas DataFrame named data. Subsequently, it removes the second column, which corresponds to the ratings, using the drop() method. This action effectively eliminates the ratings data from the DataFrame. Finally, the modified DataFrame, now devoid of the ratings column, is saved back into a new CSV file titled "movie_reviews_without_rating.csv". This streamlined process enables users to conveniently preprocess the dataset by excluding the ratings column, facilitating subsequent analysis or utilization of the data for other purposes where ratings information is not necessary.

Input:

After uploading the dataset to Google Colab for analysis, we conducted an initial inspection of the first 5 rows. This initial examination allowed us to gain insight into the structure and content of the dataset. By reviewing the first few rows, we observed the data format, the presence of missing values, and the types of information in each column. This exploratory step is crucial for understanding the dataset's characteristics and determining the appropriate data preprocessing steps and analysis techniques to be applied.

Upon reviewing the first 5 rows, we identified the presence of NaN (Not a Number) cells in some rows, particularly in the column used for sentiment analysis. These NaN cells need to be removed from the dataset to ensure accurate analysis. Additionally, to facilitate analysis and manage computational resources properly, we randomly selected 6000 rows from the dataset for further analysis. This sampling approach allows us to work with a manageable subset of the data while still capturing a representative portion of the dataset for analysis purposes. By removing NaN cells and selecting a subset of the data, we ensure that our analysis is based on clean and representative data, thereby enhancing the reliability and accuracy of our findings.

This code uses regular expressions to remove HTML tags from a text string. First, it imports the re module, then it creates a pattern <.*?> to identify HTML tags and their attributes. Next, it applies this pattern using the re.sub() function to the text found in the third row of a DataFrame's 'Review' column (df.iloc[2].Review). This process replaces each detected HTML tag with nothing, effectively cleaning up the text.

This function, remove_special, is for removing special characters (such as commas, periods, and brackets) from a given text string.

This code snippet employs the NLTK library in Python to acquire a list of stopwords for the English language. Stopwords are frequently encountered common words that typically carry little semantic value in text analysis. By downloading and accessing this list, the code enables the removal of such stopwords during text preprocessing, facilitating a focus on more meaningful content for further analysis or modeling.

This code snippet is aimed at performing stemming, a process in natural language processing (NLP) where words are reduced to their base or root form. Stemming is particularly useful for text normalization, enabling multiple variations of words with the same meaning to be represented by a single word.

This code is applying the stem_words() function to the 'Review' column of a DataFrame df. The stem_words() function is responsible for stemming each word in a text, reducing it to its base or root form. By applying this function to every review in the 'Review' column, the code modifies the DataFrame df to contain the stemmed versions of the reviews. This preprocessing step helps in standardizing the text data by converting different variations of words with the same meaning into a single representation, making it easier for subsequent analysis tasks such as sentiment analysis or text classification. The resulting DataFrame df will have the original reviews replaced with their stemmed versions.

This code snippet prepares textual data from a DataFrame for machine learning analysis. It begins by selecting the 'Review' column from the DataFrame and storing it as a NumPy array. Then, it utilizes scikit-learn's CountVectorizer to convert the raw text into a matrix of token counts. This transformation essentially represents each review as a vector of word frequencies, with each word being treated as a feature. The max_features parameter is set to 1000, limiting the number of features to the 1000 most frequent words. Finally, the transformed data is converted into a dense NumPy array. This preprocessing step enables the text data to be fed into machine learning models, allowing for the training and evaluation of sentiment analysis models or other text-based classification tasks.

Creating and Building Model

This code snippet performs the training and testing of three different Naive Bayes classifiers (Gaussian, Multinomial, and Bernoulli) using the training and testing datasets split earlier. Following, is the explanation of each part:

Splitting the Dataset:

The first part of the code snippet uses train_test_split from scikit-learn to split the dataset into training and testing sets (X_train, X_test, y_train, y_test).

The test_size=0.2 argument specifies that 20% of the data should be allocated to the testing set, leaving 80% for training.

Checking the Shape of Training and Testing Sets:

X_train.shape, y_train.shape, and y_test.shape are used to check the dimensions of the training and testing sets. These lines ensure that the data has been split correctly.

Creating Naive Bayes Classifier Objects:

The code creates three different Naive Bayes classifier objects:

clf1: Gaussian Naive Bayes classifier

clf2: Multinomial Naive Bayes classifier

clf3: Bernoulli Naive Bayes classifier

Training the Models:

The fit() method is called on each classifier object (clf1, clf2, clf3) to train the models using the training data (X_train, y_train).

Making Predictions:

The predict() method is used to make predictions on the testing data (X_test) for each classifier.

Predictions for each classifier are stored in y_pred1, y_pred2, and y_pred3 respectively.

Importing the Accuracy Score Metric:

The code imports the accuracy_score function from the metrics module of the scikit-learn library. This function is specifically designed to evaluate the accuracy of classification models by comparing the true labels (y_test) with the predicted labels (y_pred).

Calculating Accuracy Scores:

The accuracy_score() function is called three times, each time with the true labels (y_test) and the predicted labels (y_pred) for a specific classifier.

These calls compute the accuracy of each classifier in correctly classifying instances in the testing dataset.

Printing Accuracy Scores:

It calculates the accuracy scores for each classifier.

The format used is "Classifier Name acc= Accuracy Score", where the accuracy score is a floating-point value ranging from 0 to 1.

The accuracy score represents the proportion of correctly classified instances in the testing dataset, providing insights into the performance of each classifier.

Based on the model:

The Gaussian Naive Bayes classifier achieved an accuracy of approximately 0.6171.

The Multinomial Naive Bayes classifier achieved a higher accuracy of approximately 0.9369.

The Bernoulli Naive Bayes classifier achieved an accuracy of approximately 0.9315.

These accuracy scores are valuable metrics for assessing the performance of each classifier in accurately predicting the sentiment of movie reviews on unseen testing data.

Testing the Model with New Review

This code snippet demonstrates the construction of a sentiment analysis model for movie reviews using a Multinomial Naive Bayes classifier. It starts by importing necessary libraries, including NLTK for text preprocessing and scikit-learn for machine learning tasks, and initializes the NLTK tokenizer. The preprocess_text() function is defined to clean the text data by removing HTML tags, converting text to lowercase, tokenizing it into words, and eliminating stopwords.

Subsequently, the train_model() function is implemented to train the Multinomial Naive Bayes classifier on sample movie review data, employing a CountVectorizer to convert text data into a document-term matrix for model training. A sample of training data containing movie reviews and their corresponding sentiment labels is provided. After training the model, a set of new movie reviews is introduced, and the predict_sentiment() function is invoked to predict their sentiments using the trained classifier. Finally, the predicted sentiments for each review are printed alongside the original review texts, demonstrating the model's sentiment classification capabilities.

Findings and Interpretation

In conclusion, the sentiment analysis model successfully predicted the sentiments of the provided movie reviews. The first review, despite containing sarcastic remarks and criticism towards various aspects of the film industry, was predicted as positive. This suggests that the model might have detected subtle positive sentiment. On the other hand, the second review, praising Jason Momoa's performance, was correctly classified as positive. Overall, while the model demonstrated some accuracy in predicting sentiments, it may benefit from further refinement to better capture nuanced language and context. Additionally, incorporating a larger and more diverse dataset for training could improve the model's performance and generalization capabilities.

2.2. Six Wheel Chart Sentiment Analysis Methodology

In the meantime, we also explored an alternative model aimed at classifying six distinct emotions; happiness, sadness, anger, fear, love, surprise. The training dataset for this model was derived from excerpts of various txt files. We manually annotated 1,000 sentences but despite our efforts, the model's efficacy was initially limited due to insufficient training data. We conducted a Kappa test with two team members and achieved a slightly over 50 percent result. Moreover, the model performed inadequately with sentences containing sarcasm and negation terms. Consequently, we decided to proceed with the positive-negative model, which was distantly annotated with IMDB scores.

Data Loading and Preprocessing

Data Loading: the code begins by loading a dataset named Emotion_final.csv using pandas, which is a common library for data manipulation and analysis in Python.

Preprocessing: the text data is preprocessed by converting it to lowercase and splitting it into individual words. This is a basic form of text normalization that helps in reducing the dimensionality of the data and ensuring consistency.

Vectorization: the CountVectorizer from scikit-learn is used to convert the preprocessed text into a matrix of token counts. This step is crucial for transforming text data into a format that can be used by machine learning algorithms

Model Training and Evaluation

Training: The dataset is split into training and testing sets using train_test_split. A Multinomial Naive Bayes classifier is then trained on the training set.

Evaluation: The model's performance is evaluated using a confusion matrix and a classification report, which provide insights into the model's accuracy, precision, recall, and F1-score for each class.

Prediction on New Data

New Data Preprocessing: Similar preprocessing steps are applied to new text data loaded from new_text_data.csv. Which is the same data scrapped from the IMDB website.

Prediction: The trained model is used to predict emotions for the new text data.

The Confusion Matrix

The confusion matrix for this model summarizes the performance of a classification model. Each row represents the actual class (true label), while each column represents the predicted class by the model. The numbers in the matrix indicate the count of samples falling into each category. For example:

Row 1 (Actual: anger):

Predicted as anger: 7

Predicted as fear: 0

Predicted as happy: 8

Predicted as love: 0

Predicted as sadness: 6

Predicted as surprise: 0

Row 2 (Actual: fear):

Predicted as anger: 0

Predicted as fear: 2

Predicted as happy: 12

Predicted as love: 0

Predicted as sadness: 4

Predicted as surprise: 0

Row 3 (Actual: happy):

Predicted as anger: 2

Predicted as fear: 0

Predicted as happy: 60

Predicted as love: 4

Predicted as sadness: 11

Predicted as surprise: 0

Row 4 (Actual: love):

Predicted as anger: 2

Predicted as fear: 0

Predicted as happy: 16

Predicted as love: 0

Predicted as sadness: 2

Predicted as surprise: 0

Row 5 (Actual: sadness):

Predicted as anger: 1

Predicted as fear: 0

Predicted as happy: 21

Predicted as love: 0

Predicted as sadness: 31

Predicted as surprise: 0

Row 6 (Actual: surprise):

Predicted as anger: 0

Predicted as fear: 1

Predicted as happy: 7

Predicted as love: 0

Predicted as sadness: 3

Predicted as surprise: 0

[[ 7 0 8 0 6 0]

[ 0 2 12 0 4 0]

[ 2 0 60 4 11 0]

[ 2 0 16 0 2 0]

[ 1 0 21 0 31 0]

[ 0 1 7 0 3 0]]

Classification Report Analysis:

Overall Accuracy: The overall accuracy of the model is 50%, indicating that 50% of the predictions made by the model were correct.

Class-specific Metrics:

The model performs relatively well in predicting 'happy' and 'sadness' emotions with high precision and recall. This suggests that the model effectively identifies instances of 'happy' and 'sadness' and minimizes false positives and false negatives for these emotions. However, the 'love' emotion exhibits a trade-off between precision and recall. While precision for 'love' is high (48%), indicating that the model correctly identifies most instances it predicts as 'love', recall is relatively low (0%), suggesting that the model misses many actual instances of 'love' in the dataset. This indicates that while the model is confident in its predictions of 'love', it fails to capture a significant portion of the true 'love' instances.

The 'anger' emotion also shows a similar trend with high precision (58%) but relatively low recall (33%), indicating that the model correctly identifies most instances it predicts as 'anger' but misses many actual instances of 'anger' in the dataset.

The 'fear' emotion has a relatively low recall (11%), indicating that the model struggles to correctly identify instances of this emotion. This suggests that the model frequently fails to capture instances of 'fear' in the dataset.

The 'surprise' emotion has a low recall (0%), indicating that the model struggles to correctly identify instances of this emotion. This suggests that the model frequently fails to capture instances of 'surprise' in the dataset.

Evidently while the model demonstrates strong performance in predicting certain emotions such as 'happy' and 'sadness', it shows weaknesses in accurately identifying instances of 'love', 'anger', 'fear', and 'surprise'. Further investigation and model refinement may be necessary to improve performance on these emotion classes.

Shortcomings and Sarcasm Detection

Sarcasm Detection: The current approach does not explicitly account for sarcasm. Sarcasm often requires understanding the context and the tone of the text, which is challenging for machine learning models trained on simple text features.

Preprocessing Limitations: The preprocessing steps, while basic, might not be sufficient for all types of text data. For instance, removing punctuation, stop words, and stemming or lemmatization might be necessary for more accurate results.

Model Complexity: The use of a Naive Bayes classifier is straightforward but might not capture complex patterns in the data. More sophisticated models, such as deep learning models, could potentially offer better performance for emotion detection, especially in detecting nuanced emotions or sarcasm.

Limitations and Conclusion

Due to the insufficient results of the Kappa Test for the Six Sentiment Analysis model, this section will be focusing on the Positive-Negative Model solely. Our study encountered several limitations that should be considered in the development and application of sentiment analysis models for movie comments. Firstly, the dataset used for model training was limited, which could impact the model's accuracy. Expanding the dataset with diverse data sources may enhance the model's performance. Additionally, during the initial model development stage, the model faced challenges in accurately classifying sentiments when encountering negation terms before positively annotated words or when dealing with ironic comments. Although the enhanced model improved in identifying negation terms like "not," it still struggled with accurately annotating ironic statements. To illustrate this limitation, we evaluated a set of ten sample sentences within the model, revealing instances where sentiment classification sentiment analysis struggled because of language complexities and context.

Table 2

These examples underscore the importance of addressing context sensitivity in sentiment analysis, as the model's performance may be influenced by nuanced linguistic expressions, sarcasm, or cultural references present in movie comments. Furthermore, the model's performance may vary across different movie genres or audience demographics, necessitating domain-specific adaptations to ensure accurate sentiment analysis. These variations highlight the importance of tailoring the model to specific domains or audience segments to maintain its effectiveness and reliability in predicting sentiment in movie comments.

In conclusion, this project presents a comprehensive approach to sentiment analysis of movie comments, aiming to provide valuable insights for stakeholders in the film industry. Through the development of a machine learning model, the researchers have demonstrated the capability to accurately predict sentiment (positive or negative) in movie comments sourced from platforms like IMDB. While the model shows promise in aiding decision-making processes, it also highlights the importance of addressing contextual nuances, such as sarcasm and negation, to improve accuracy further. Despite the limitations, including a limited dataset and challenges in context sensitivity, the sentiment analysis model offers significant potential for real-life applications. By leveraging insights derived from audience reactions, filmmakers, studios, and distributors can make more informed decisions regarding marketing strategies, release schedules, and content refinement. Additionally, integration with other data sources, such as box office performance metrics and IMDB ratings, could enhance the model's predictive capabilities and contribute to more accurate forecasting in the film industry.

Reflecting on this project, we recognize the significance of context sensitivity in sentiment analysis, a lesson learned through our endeavors to accurately classify sentiments across diverse linguistic expressions. While our positive-negative sentiment analysis model demonstrated proficiency, the challenges encountered have provided valuable insights into the complexities of interpreting emotions within textual data. Moving forward, we understand the importance of continuously refining our models and methodologies to address these challenges effectively. Indeed, this project has underscored the necessity for interdisciplinary collaboration. The incorporation of domain expertise from fields such as linguistics and psychology could significantly enhance the accuracy and depth of sentiment analysis models. Overall, this project not only contributes to the advancement of sentiment analysis in the film industry but also serves as a learning experience, guiding future research and applications in this domain.

Acknowledgments and Datasets

The dataset used for our primary focus, the positive-negative model central to the project, wasn't manually obtained or annotated. Instead, it underwent an automated process driven by the initial segment of the code shared in the following link: [] It's important to note that the scraped data is dynamic, meaning that the model's results will be updated to reflect the current website reviews every time the code is run.

The data for the secondary experimental model (the six sentiment analysis model), is available at the following link: [].
