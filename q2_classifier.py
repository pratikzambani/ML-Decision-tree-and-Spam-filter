import csv
import argparse
from math import log
class Baiyes_classifier:

    def __init__(self):
        self.spamCount=0
        self.hamCount=0
        self.spamWords={}
        self.hamWords={}
        self.total_spam_words=0
        self.total_ham_words=0

    def train_classifier(self, training_data):
        rows = csv.reader(open(training_data))
        for row in rows:
            listOfWords = row[0].split(" ")
            if listOfWords[1] == "spam":
                self.spamCount += 1 # emails
                for i in range(2,len(listOfWords),2):
                    word=listOfWords[i]
                    if word in self.spamWords:
                        count = self.spamWords[word]
                        self.spamWords[word] = count+int(listOfWords[i+1])
                        self.total_spam_words +=int(listOfWords[i+1])
                    else:
                        self.spamWords[word] = int(listOfWords[i+1])
                        self.total_spam_words +=int(listOfWords[i+1])

            elif listOfWords[1] == "ham":
                self.hamCount += 1
                for i in range(2, len(listOfWords), 2):
                    word = listOfWords[i]
                    if word in self.hamWords:
                        count = self.hamWords[word]
                        self.hamWords[word] = count+int(listOfWords[i+1])
                        self.total_ham_words += int(listOfWords[i + 1])

                    else:
                        self.hamWords[word] = int(listOfWords[i+1])
                        self.total_ham_words += int(listOfWords[i + 1])


    def classify(self,test_data, output_file):
        total = 0.0
        correct = 0.0
        testrows = csv.reader(open(test_data))
        v= float(len(self.spamWords) + len(self.hamWords))
        for row in testrows:
            total = total + 1
            listOfWords = row[0].split(" ")
            words = listOfWords[2:]
            prob_spam = 0.0
            prob_ham = 0.0

            for word in set(words):
                if word in self.spamWords:
                    occurences_in_spam = self.spamWords[word]

                if word in self.hamWords:
                    occurences_in_ham = self.hamWords[word]

                sp= log((occurences_in_spam+1.0)/(self.total_spam_words+v))
                hp= log((occurences_in_ham+1.0)/(self.total_ham_words+v))
                prob_spam = prob_spam + (sp)
                prob_ham = prob_ham + (hp)

            prob_ham=prob_ham+log(float(self.hamCount)/(self.hamCount+self.spamCount))
            prob_spam=prob_spam+log(float(self.spamCount)/(self.spamCount+self.hamCount))

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
                output_writer.writerow([listOfWords[0],classified_type])
            if spam and listOfWords[1] == "spam":
                correct = correct + 1
            elif spam==False and listOfWords[1] == "ham":
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