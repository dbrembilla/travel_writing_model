from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
import statistics as st
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.datasets import load_files
import nltk

from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfTransformer
import os
import time

#Credit for a significant part of the algorithm to https://stackabuse.com/text-classification-with-python-and-scikit-learn

def Training(FilePath, Method = False):
    FilePathTargets = os.listdir(FilePath) #get
    if len(FilePathTargets) <= 1:
        print("Please put text in target folders")
        exit()
    dataset=None
    try:
        dataset = load_files(FilePath) #Loads each folder inside the directory as a category e.g. if I have Travel and NotTravel folders it will use those categories
    except:
        print("Please put text in target folders")
        exit()
    Categories = dataset.target_names
    print(Categories)
    x, y = dataset.data, dataset.target
    Categories = dataset.target_names
    documents = []
    if Method is False: #uses simple preprocess from the gensim package
        from gensim.utils import simple_preprocess
        for sen in x:
            simple_preprocess(sen)
        documents=x
    elif Method is True: #uses a more elaborate preprocessing using a lemmatizer and regular expressions
        from nltk.stem import WordNetLemmatizer
        import re
        stemmer = WordNetLemmatizer()
        for sen in range(0, len(x)):
            # Remove all the special characters
            document = re.sub(r'\W', ' ', str(x[sen]))

            # remove all single characters
            document = re.sub(r'\s+[a-zA-Z]\s+', ' ', document)

            # Remove single characters from the start
            document = re.sub(r'\^[a-zA-Z]\s+', ' ', document)

            # Substituting multiple spaces with single space
            document = re.sub(r'\s+', ' ', document, flags=re.I)

            # Removing prefixed 'b'
            document = re.sub(r'^b\s+', '', document)

            # Converting to Lowercase
            document = document.lower()

            # Lemmatization
            document = document.split()

            document = [stemmer.lemmatize(word) for word in document]
            document = ' '.join(document)

            documents.append(document)
    vectorizer = CountVectorizer(min_df = 5,max_df=0.7, ngram_range=(1,3), stop_words=stopwords.words('english'))
    x = vectorizer.fit_transform(documents).toarray() #converts the processed documents in a vector
    #tfidfconverter = TfidfTransformer()
    #x = tfidfconverter.fit_transform(x).toarray() #applies tfidf
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3) #splits the corpus in test groups.

    clf = MultinomialNB() 
    clf.fit(x_train, y_train) #fits a classifier
    y_pred = clf.predict(x_test) #makes the classifier predict
    print(confusion_matrix(y_test, y_pred)) #results
    print(classification_report(y_test, y_pred, target_names=Categories)) #returns precision, recall and combined accuracy.
    print(accuracy_score(y_test, y_pred)) #returns combined accuracy.
    return accuracy_score(y_test, y_pred)


if __name__ == '__main__':
    StartTime = time.time()
    print("Warning: not tested on MacOs or Linux")
    x = input("Do you need to download stopwords? Y/N ")
    if x in "Yy":
        nltk.download('stopwords')
    FilePath = input("Please enter filepath: ")
    Met=input("Do you want to use a complete preprocess rather than a simple preprocess? Y/N ")
    if Met in "Yy":
        Met = True
    else:
        Met=False
    results = set()
    while True:
        n = input("Enter number of tests: ")

        n = int(n)
        for i in range(n):
            x = Training(FilePath, Met)
            results.add(x)

        break

    num = 0
    for i in results:
        num += i

    print("Mean of tests: ", st.mean(results))
    print("Median of tests: ", st.median(results))
    print("Standard deviation: ", st.stdev(results))
    print("--- %s seconds --- for training the module" % (time.time() - StartTime))


