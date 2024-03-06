import os
import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob
from gensim import corpora
from gensim.models import LdaModel
from scipy.sparse import hstack, csr_matrix
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score


# # The first time you run it, you may need to download NLTK's stop words and WordNet data
# nltk.download('stopwords')
# nltk.download('wordnet')

# Set the root directory of the dataset
data_root = r'dataset_bbc/bbc'
# Dataset from:
# D. Greene and P. Cunningham. "Practical Solutions to the Problem of Diagonal Dominance in Kernel Document Clustering", Proc. ICML 2006.

# Initialize a list to store dictionaries before converting it to a DataFrame
data_list = []

# Set the root directory of the dataset
for category in os.listdir(data_root):
    category_path = os.path.join(data_root, category)

    # Make sure it is file path
    if os.path.isdir(category_path):
        # Iterate over all files in the directory
        for filename in os.listdir(category_path):
            file_path = os.path.join(category_path, filename)

            # Make sure it is a file
            if os.path.isfile(file_path):
                # Read the file
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
                    text = file.read()
                    # The text and category are added to the list, which is later converted to a DataFrame
                    data_list.append({'text': text, 'category': category})

# Convert the list of dictionaries to a DataFrame
data = pd.DataFrame(data_list)

# category_data = data[data['category'] == category].head(5)  # Get the first five entries of the category
# for _, row in category_data.iterrows():
#     print(f"Category: {category}")
#     print(f"Content:\n{row['text'][:200]}")  # Prints the first 200 characters of the content
#     print("\n" + "-"*80 + "\n")  # Print divider

# Create a text cleaning function
def clean_text(text):
    # Converts the text to lowercase
    text = text.lower()
    # remove symbols
    text = re.sub(r'[^\w\s]', '', text)
    # remove numbers
    text = re.sub(r'\d+', '', text)
    # remove stop words
    stop_words = set(stopwords.words('english'))
    words = text.split()
    words = [word for word in words if word not in stop_words]
    # Restoring word form
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]
    # Regroup to a new text
    text = ' '.join(words)
    return text

# Clean each text in the data DataFrame
data['text_clean'] = data['text'].apply(clean_text)

# # Use.head() to get the first few cleaned records for each category (default 5).
# for category in data['category'].unique():
#     print(f"Category: {category}")
#     # Get the top 5 cleaned text of the current category
#     category_data = data[data['category'] == category].head()
#     # Print the text and categories
#     for index, row in category_data.iterrows():
#         print(f"Content:\n{row['text_clean']}\n")
#     print("\n" + "-"*80 + "\n")

vectorizer = TfidfVectorizer(max_features=5000)  # We only focus on the top 5000 words
X_tfidf = vectorizer.fit_transform(data['text_clean'])

# Computing Sentiment Polarity
data['sentiment'] = data['text'].apply(lambda x: TextBlob(x).sentiment.polarity)
# Preparing text data
texts = data['text_clean'].apply(lambda x: x.split())

# Creating a dictionary
dictionary = corpora.Dictionary(texts)
corpus = [dictionary.doc2bow(text) for text in texts]

# Training the LDA model
lda_model = LdaModel(corpus=corpus, id2word=dictionary, num_topics=5, random_state=100, update_every=1, chunksize=100, passes=10, alpha='auto', per_word_topics=True)

# Get the document topic distribution
def get_document_topics(bow):
    topic_probs = lda_model.get_document_topics(bow)
    topic_probs = sorted(topic_probs, key=lambda x: x[1], reverse=True)
    # Only the probability of the most probable topic is returned
    return topic_probs[0][1] if topic_probs else 0

data['lda_topic_strength'] = [get_document_topics(bow) for bow in corpus]

def lexical_diversity(text):
    return len(set(text)) / len(text) if text else 0

data['lexical_diversity'] = data['text_clean'].apply(lambda x: lexical_diversity(x.split()))

# Convert other features to sparse matrix format
additional_features = csr_matrix(data[['sentiment', 'lda_topic_strength', 'lexical_diversity']].values)
# Merge TF-IDF features and other features
X_combined = hstack([X_tfidf, additional_features])
y = data['category']

# We first split the training set and a temporary test set
X_temp, X_test, y_temp, y_test = train_test_split(X_combined, y, test_size=0.2, random_state=42)

# Then the training set and development set are divided from the temporary test set
X_train, X_dev, y_train, y_dev = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42)  # 0.25 x 0.8 = 0.2

svm_model = SVC(kernel='linear', C=1.0, random_state=42)
svm_model.fit(X_train, y_train)

# Evaluation is performed on the development set
dev_predictions = svm_model.predict(X_dev)
print(classification_report(y_dev, dev_predictions))

# Evaluation on the test set
test_predictions = svm_model.predict(X_test)
print("Test set accuracy: ", accuracy_score(y_test, test_predictions))
print(classification_report(y_test, test_predictions))

#  y_test is the true labels of the test set and test_predictions is the predictions of the model
incorrect_indices = [i for i, (pred, actual) in enumerate(zip(test_predictions, y_test)) if pred != actual]

# Create a new DataFrame containing the misclassified samples
incorrect_samples = data.iloc[incorrect_indices].copy()  # incorrect_indices is the index of the previously identified misclassified samples
incorrect_samples['predicted_label'] = [test_predictions[i] for i in incorrect_indices]  # Add the predicted label column
incorrect_samples.to_csv(r'Misclassified samples/Misclassified_samples.csv', index=False)
