import csv
import argparse
from math import log
import re

class Baiyes_classifier:

    def __init__(self):
        self.spamCount=0
        self.hamCount=0
        self.spamWords={}
        self.hamWords={}
        self.total_spam_words=0
        self.total_ham_words=0

    "This method builds the model based on the training data passed"
    def train_classifier(self, training_data):
        rows = csv.reader(open(training_data))
        for row in rows:
            listOfWords = row[0].split(" ")
            filename = '/Users/tswetha/Downloads/trec05p-1/data'+listOfWords[1]
            type=listOfWords[0]
            listOfWords=[]
            f=open(filename,"rb")
            for i,line in enumerate(f):
                listOfWords.extend(set(re.split("\s{1,}",line)))

            if type == "spam":
                "Incrementing count of spam emails"
                self.spamCount += 1
                for i in range(0,len(listOfWords)):
                    "Adding occurences of spam words in spamWords dictionary"
                    word=listOfWords[i]
                    if word in self.spamWords:
                        count = self.spamWords[word]
                        self.spamWords[word] = count+1
                        self.total_spam_words +=1.0
                    else:

                        self.spamWords[word] = 1.0
                        self.total_spam_words +=1.0

            elif type == "ham":
                "Incrementing count of ham emails"
                self.hamCount += 1
                for i in range(2, len(listOfWords), 2):
                    "Adding occurences of ham words in hamWords dictionary"
                    word = listOfWords[i]
                    if word in self.hamWords:
                        count = self.hamWords[word]
                        self.hamWords[word] = count+1.0
                        self.total_ham_words += 1.0

                    else:
                        self.hamWords[word] = 1.0
                        self.total_ham_words += 1.0

    "This method uses the model generated to classify the test data"
    def classify(self,test_data, output_file):
        total = 0.0
        correct = 0.0
        testrows = csv.reader(open(test_data))
        "This is the vocabulary of words"
        v= float(len(self.spamWords) + len(self.hamWords))
        for row in testrows:
            total = total + 1
            listOfWords = row[0].split(" ")
            filename = '/Users/tswetha/Downloads/trec05p-1/data' + listOfWords[1]
            type = listOfWords[0]
            listOfWords = []
            f = open(filename, "rb")
            for i,line in enumerate(f):
                listOfWords.extend(set(re.split("\s{1,}",line)))
            prob_spam = 0.0
            prob_ham = 0.0
            upperCase=0
            #List of common words in emails
            common=["in","is","of","are","our","the","a","an", 'i','am','hey','hello','regards','yours','morning','night'\
                ,'greetings','from','to','at','for','or','dear','respected','sir','madam','miss','faithfully','thanks','thankyou'\
                    ,'anticipation','thank','kindly','&nbsp']
            #List of common spam words
            commonspam = ['virus','malware','freeware','lottery','junk','spam','million','><']
            for word in set(listOfWords):

                if word.lower() in set(common):
                    continue
                if word.isupper():
                    upperCase+=1
                if word in self.spamWords:
                    occurences_in_spam = self.spamWords[word]

                if word in self.hamWords:
                    occurences_in_ham = self.hamWords[word]
                "Calculating the probabilty of occurence of the word given it was a spam/ham email."
                "Using Laplace smoothing here add-1 smoothing"
                sp= log((occurences_in_spam+1.0)/(self.total_spam_words+v))
                hp= log((occurences_in_ham+1.0)/(self.total_ham_words+v))
                if word.count("!") > 5 or word.count("$") > 2 or upperCase>5 or '><'in word.lower():
                    sp=1.0
                if '@' in word:
                    index = word.index('@')
                    k = sum(1 for c in word[:index+1] if c.isupper())
                    avg = float(k/ len(word[:index+1]))
                    if avg>0.3:
                        sp = 1.0

                prob_spam = prob_spam + (sp)
                prob_ham = prob_ham + (hp)

            "Adding it to the probabilty of an email being spam/ham"
            prob_ham=prob_ham+log(float(self.hamCount)/(self.hamCount+self.spamCount))
            prob_spam=prob_spam+log(float(self.spamCount)/(self.spamCount+self.hamCount))

            "Classifying the test data"
            if prob_spam > prob_ham:
                spam=True
            else:
                spam=False
            with open(output_file, 'a') as csvfile:
                output_writer=csv.writer(csvfile, delimiter=' ')
                if spam==True:
                    classified_type="spam"
                else:
                    classified_type="ham"
                output_writer.writerow([type,classified_type])
            if spam and type== "spam":
                correct = correct + 1
            elif spam==False and type == "ham":
                correct = correct + 1
        return (correct / total)* 100


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f1', help='training file in csv format', required=True)
    parser.add_argument('-f2', help='test file in csv format', required=True)
    parser.add_argument('-o', help='output labels for the test dataset', required=False)
    args = vars(parser.parse_args())

    training_data=args['f1']
    test_data = args['f2']
    output_file = args['o']
    classifier = Baiyes_classifier()
    classifier.train_classifier(training_data)

    accuracy = classifier.classify(test_data, output_file)
    print('Accuracy:',accuracy)