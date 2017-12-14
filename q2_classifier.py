import csv
if __name__ == '__main__':
    rows = csv.reader(open("train"), delimiter="\t")
    count=0
    spamCount=0
    hamCount=0
    spamWords = {}
    hamWords = {}
    for row in rows:
        flag=0
        flag1 = 0
        listOfWords = row[0].split(" ")
        if listOfWords[1] == "spam":
            spamCount+=1
            for word in listOfWords[2:]:
                if word in spamWords:
                    count= spamWords[word]
                    if flag==0:
                        count=count+1
                        spamWords[word] = count
                else:
                    spamWords[word]=1.0
                    flag=1
        if listOfWords[1] == "ham":
            hamCount+=1
            for word in listOfWords[2:]:
                if word in hamWords:
                    count = hamWords[word]
                    if flag1==0:
                        count = count+1
                        hamWords[word] = count
                else:
                    hamWords[word]=1.0
                    flag1=1
    v= spamCount+hamCount
    correct=0.0
    total=0.0
    testrows = csv.reader(open("test"), delimiter="\t")
    for row in testrows:
        total=total+1
        listOfWords = row[0].split(" ")
        words = listOfWords[2:]
        prob_spam=1.0
        prob_ham=1.0
        for word in words:
            spam_word_count=0
            ham_word_count=0
            if word in spamWords:
                spam_word_count=spamWords[word]
            if word in hamWords:
                ham_word_count = hamWords[word]
            sp = (spam_word_count+1)/(spamCount+1+v)
            hp = (ham_word_count+1)/(hamCount+1+v)
            prob_spam = prob_spam*sp
            prob_ham = prob_ham*hp
        if prob_spam>prob_ham and listOfWords[1]=="spam":
            correct=correct+1
        elif prob_ham>prob_spam and listOfWords[1]=="ham":
            correct=correct+1
    Accuracy = correct/total * 100
    print(Accuracy)