This project implements a text classification solution using a Support Vector Machine (SVM) to classify news articles into five categories: tech, business, sport, politics, and entertainment.

How to getting Started:
These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

Prerequisites:
The project requires the following tools:

        Python 3.x
	pandas
	scikit-learn
	NLTK
	Gensim
	TextBlob
	Installing

Ensure Python 3.x is installed on your system. Then install the required Python packages by use code: pip install pandas scikit-learn nltk gensim textblob
Additionally, you might need to download NLTK stopwords and WordNet data:
        import nltk
        nltk.download('stopwords')
        nltk.download('wordnet')

Running the Code：
1,Clone the repository or download the source code to your preferred directory.
2,Change directory to the location where you downloaded the files:
        cd path/to/your/directory
3,Run the Python script main.py:
	python main.py

Code Structure：
main.py: The main script that performs data preprocessing, feature extraction, and classification.
dataset_bbc/: Directory containing the bbc_news dataset with subdirectories for each news category.
Misclassified samples/：All misclassified samples will be placed in a.csv file in this folder for review and improvement.

Output：
The program will train an SVM classifier on the processed text data, perform predictions, and print out a classification report for both development and test sets. It will also generate a CSV file containing misclassified samples.

Authors：
Yufan Yang

