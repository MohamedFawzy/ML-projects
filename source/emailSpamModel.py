import os
from collections import Counter
def makeDictonary(train_dir):
    emails = [os.path.join(train_dir, f) for f in os.listdir(train_dir)]
    all_words = []
    for mail in emails:
        with open(mail) as m:
            for i,line in enumerate(m):
                if i == 2:
                    words = line.split()
                    all_words += words

    dictionary = Counter(all_words)
    # remove non words such as punctation etc
    list_to_remove = dictionary.keys()
    for item in list_to_remove:
        if item.isalpha() == False:
            del dictionary[item]
        elif len(item) == 1:
            del dictionary[item]

    dictionary  = dictionary.most_common(3000)
    return dictionary

cwd = os.getcwd()
train_dir = cwd+"/source/train-mails"
print(makeDictonary(train_dir))