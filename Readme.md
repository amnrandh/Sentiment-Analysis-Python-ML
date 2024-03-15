# Title: Building a Sentiment Analysis Model Using Python and Machine Learning

## Dataset:

The dataset used here is the Large Movie Review Dataset (v1.0). It's meant for deciding if text is positive or negative. There are 25,000 reviews for training and 25,000 for testing. You can also use more data that isn't labeled. The dataset includes both raw text and bag-of-words formats. Remember to cite the ACL 2011 paper when using it.

https://ai.stanford.edu/~amaas/data/sentiment/

## Introduction:

Sentiment analysis, also called opinion mining, figures out if text is positive, negative, or neutral. The code creates a sentiment analysis model using Python and machine learning techniques.

## Setup and Data Loading:

The code begins by importing necessary libraries such as NumPy, pandas, Matplotlib, NLTK, and scikit-learn. It downloads NLTK resources like stopwords and WordNet and loads the dataset from a CSV file containing IMDb movie reviews.

## Text Preprocessing:

Text preprocessing is crucial for preparing the data for analysis. The code defines a function, preprocess_text, to clean and preprocess the text data. This involves removing HTML tags, special characters, accents, and common words. It also simplifies words through stemming and lemmatization. We use both lemmatization and stemming to find a balance between accuracy and speed in preparing the text for analysis.

## Splitting Data and Vectorization:

The dataset is split into two parts: training and testing. Then, the code turns the text data into numbers using TF-IDF vectorization. TF-IDF looks at how often a word appears in a document (term frequency) and how unique it is across all documents (inverse document frequency). This helps highlight important words for sentiment analysis and ignores common words. TF-IDF lets the model focus on the most meaningful words for sentiment classification.

## Model Training and Evaluation:

The code trains a logistic regression model using the training data and checks how well it works on the test data. We picked logistic regression because it's often used for tasks like sentiment analysis. We use metrics like accuracy, precision, recall, F1-score, and a confusion matrix to see how good our model is at spotting positive and negative sentiments. These metrics tell us if our model is doing a good job or not.

**Output:**

Accuracy: 0.7461
Classification Report:
              precision    recall  f1-score   support

           0       0.74      0.76      0.75      4993
           1       0.75      0.73      0.74      5007

    accuracy                           0.75     10000
   macro avg       0.75      0.75      0.75     10000
weighted avg       0.75      0.75      0.75     10000

Confusion Matrix:
[[3782 1211]
 [1328 3679]]


## Word Frequency Analysis:

The code analyzes the frequency of words in positive and negative reviews from the training data. This analysis helps identify the most common words associated with each sentiment. Understanding these words helps us understand the traits of positive and negative reviews. This information can enhance the accuracy of the sentiment analysis model.

**Output:**

```Top 30 Positive Words:
       Word  Count
0      film  35257
1      movi  30917
2       one  20879
3      like  15742
4       see  11463
5      good  10965
6      time  10663
7      make  10420
8     stori   9990
9   charact   9987
10      get   9902
11    great   9837
12    watch   9432
13     love   9312
14   realli   8467
15    would   8457
16     also   8343
17     well   8116
18     show   7977
19     even   7745
20     play   7683
21    scene   7146
22     much   7003
23    first   6913
24    think   6596
25    peopl   6536
26     best   6394
27      ...   6324
28       go   6303
29     look   6231

Top 30 Negative Words:
       Word  Count
0      movi  39734
1      film  30795
2       one  19582
3      like  19095
4      make  12285
5      even  12238
6       get  12084
7     would  11110
8     watch  11074
9      good  10918
10      bad  10336
11  charact  10265
12      see  10134
13     time   9938
14   realli   9638
15      ...   9430
16     look   9038
17    scene   8149
18    stori   7673
19      act   7638
20     much   7621
21       go   7525
22    could   7420
23    peopl   7180
24    think   7014
25    thing   6967
26      end   6543
27     show   6433
28     seem   6433
29      say   6383```


## Conclusion:

The code creates a sentiment analysis model using Python and machine learning. It's accurate about 74.61% of the time. When we check the classification report and confusion matrix, we see the model does pretty well at predicting sentiment in IMDb movie reviews.
