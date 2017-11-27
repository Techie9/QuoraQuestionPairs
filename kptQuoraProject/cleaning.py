
# @Author  : Pallavi Chandanshive
# @email   : pvc8661@g.rit.edu

import re
from nltk import SnowballStemmer
from nltk.corpus import stopwords
import pandas as pd


globals
train_1 = []
train_2 = []
labels = []
test_1 = []
test_2 = []

def clean_text( text):

        """
        Clean text
         :param text: the string of text
        :return: text string after cleaning

        """
        #   convert the text to lower case
        text = str(text).lower()

        # unit
        text = re.sub(r"(\d)kgs+", lambda m: m.group(1) + ' kg ', text)        # e.g. 4kgs => 4 kg
        text = re.sub(r"(\d)kg+",lambda m: m.group(1) + ' kg ', text)          # e.g. 4kg =>  4 kg
        text = re.sub(r"(\d)\$+",lambda m: m.group(1) + ' dollar ', text)      # e.g. 1000$ =>1000 dollar
        text = re.sub(r"\$(\d+)",lambda m: m.group(1) + ' dollar ', text)      # e.g $500  => 500 dollar
        text = re.sub(r"rs(\d+)",lambda m: m.group(1) + ' rupee ', text)       # e.g rs500 => 500 rupee
        text = re.sub(r"(\d)kv+",lambda m: m.group(1) + ' kv ', text)          # e.g 70kv => 70 kv
        text = re.sub(r"(\d)k+", lambda m: m.group(1) + '000 ', text)          # e.g 70k  => 7000
        text = re.sub(r"(\d)(\s)rs", lambda m: m.group(1) + ' rupee ', text)   # e.g 100 rs => 100 rupee
        text = re.sub(r"(\d)(\s)lacs", lambda m: m.group(1) + ' lakh ', text)   # e.g 10 lacs => 10 lakh
        text = re.sub(r"(\d)rs" , lambda m:m.group(1) + ' rupee ', text)       # e.g  500rs => 500 rupee

        # acronym
        text = re.sub(r"can\'t", "can not", text)
        text = re.sub(r"cannot", "can not ", text)
        text = re.sub(r"what\'s", "what is", text)
        text = re.sub(r"What\'s", "what is", text)
        text = re.sub(r"\'ve ", " have ", text)
        text = re.sub(r"n\'t", " not ", text)
        text = re.sub(r"i\'m", "i am ", text)
        text = re.sub(r"\'re", " are ", text)
        text = re.sub(r"\'d", " would ", text)
        text = re.sub(r"\'ll", " will ", text)
        text = re.sub(r"c\+\+", "cplusplus", text)
        text = re.sub(r"c \+\+", "cplusplus", text)
        text = re.sub(r"c \+ \+", "cplusplus", text)
        text = re.sub(r"c#", "csharp", text)
        text = re.sub(r"f#", "fsharp", text)
        text = re.sub(r"g#", "gsharp", text)
        text = re.sub(r" e mail ", " email ", text)
        text = re.sub(r" e \- mail ", " email ", text)
        text = re.sub(r" e\-mail ", " email ", text)
        text = re.sub(r",000", '000', text)
        text = re.sub(r"\'s", " ", text)
        text =re.sub(r"inr", "indian rupee",text)

        # spelling correction
        text = re.sub(r"ph\.d", "phd", text)
        text = re.sub(r"PhD", "phd", text)
        text = re.sub(r"pokemons", "pokemon", text)
        text = re.sub(r"pokémon", "pokemon", text)
        text = re.sub(r"pokemon go ", "pokemon-go ", text)
        text = re.sub(r" e g ", " eg ", text)
        text = re.sub(r" b g ", " bg ", text)
        text = re.sub(r" 9 11 ", " 911 ", text)
        text = re.sub(r" j k ", " jk ", text)
        text = re.sub(r" fb ", " facebook ", text)
        text = re.sub(r"facebooks", " facebook ", text)
        text = re.sub(r"facebooking", " facebook ", text)
        text = re.sub(r"insidefacebook", "inside facebook", text)
        text = re.sub(r"donald trump", "trump", text)
        text = re.sub(r"the big bang", "big-bang", text)
        text = re.sub(r"the european union", "eu", text)
        text = re.sub(r" usa ", " america ", text)
        text = re.sub(r" us ", " america ", text)
        text = re.sub(r" u s ", " america ", text)
        text = re.sub(r" U\.S\. ", " america ", text)
        text = re.sub(r" US ", " america ", text)
        text = re.sub(r" American ", " america ", text)
        text = re.sub(r" America ", " america ", text)
        text = re.sub(r" quaro ", " quora ", text)
        text = re.sub(r" mbp ", " macbook-pro ", text)
        text = re.sub(r" mac ", " macbook ", text)
        text = re.sub(r"macbook pro", "macbook-pro", text)
        text = re.sub(r"macbook-pros", "macbook-pro", text)
        text = re.sub(r" 1 ", " one ", text)
        text = re.sub(r" 2 ", " two ", text)
        text = re.sub(r" 3 ", " three ", text)
        text = re.sub(r" 4 ", " four ", text)
        text = re.sub(r" 5 ", " five ", text)
        text = re.sub(r" 6 ", " six ", text)
        text = re.sub(r" 7 ", " seven ", text)
        text = re.sub(r" 8 ", " eight ", text)
        text = re.sub(r" 9 ", " nine ", text)
        text = re.sub(r"googling", " google ", text)
        text = re.sub(r"googled", " google ", text) 
        text = re.sub(r"googleable", " google ", text)
        text = re.sub(r"googles", " google ", text)
        text = re.sub(r"the european union", " eu ", text)
        text = re.sub(r"dollars", " dollar ", text)

        # punctuation
        text = re.sub(r"\+", " + ", text)
        text = re.sub(r"'", " ", text)
        text = re.sub(r"-", " - ", text)
        text = re.sub(r"/", " / ", text)
        text = re.sub(r"\\", " \ ", text)
        text = re.sub(r"=", " = ", text)
        text = re.sub(r"\^", " ^ ", text)
        text = re.sub(r":", " : ", text)
        text = re.sub(r"\.", " . ", text)
        text = re.sub(r",", " , ", text)
        text = re.sub(r"\?", " ? ", text)
        text = re.sub(r"!", " ! ", text)
        text = re.sub(r"\"", " \" ", text)
        text = re.sub(r"&", " & ", text)
        text = re.sub(r"\|", " | ", text)
        text = re.sub(r";", " ; ", text)
        text = re.sub(r"\(", " ( ", text)
        text = re.sub(r"\)", " ( ", text)

        # symbol replacement
        text = re.sub(r"&", " and ", text)
        text = re.sub(r"\|", " or ", text)
        text = re.sub(r"=", " equal ", text)
        text = re.sub(r"\+", " plus ", text)
        text = re.sub(r"₹", " rs ", text)  # 测试！
        text = re.sub(r"\$", " dollar ", text)

        # remove extra space
        text = ' '.join(text.split())
        text = text.split()

        # calling the stopword method to remove stop words
        removed_stopword_text=removestopwords(text)
        # calling the stemmer method
        stemmedText = stem(removed_stopword_text)
        str_text = " ".join(stemmedText)

        return str_text


def removestopwords(text,remove_stopwords=True):

    # remove stopwords for the text
    if remove_stopwords:
        stops = set(stopwords.words("english"))

        # Adding interrogative words
        stops.remove('why');
        stops.remove('who');
        stops.remove('how');
        stops.remove('when');
        stops.remove('where');
        stops.remove('what');
        stops.remove('which');
        for w in text:
            if w in stops:
                text.remove(w)
    # Return the text after removing the stop words
    return text


def stem(text, stem_words=True):
    # shorten words to their stems
    if stem_words:
        stemmer = SnowballStemmer('english')
        stemmedtext = [ ]

        for word in text:
            stemmed_words = stemmer.stem(word)
            stemmedtext.append(stemmed_words)

    # Return a list of stemmed words
    return stemmedtext


def main():
    #   filename
    trainingfile = "train.csv"
    testingfile  = "test.csv"

    # reading the CSV file
    df_training = pd.read_csv(trainingfile, encoding='ISO-8859-1', usecols=["question1", "question2", "is_duplicate"])
    df_testing = pd.read_csv(testingfile, encoding='ISO-8859-1', usecols=["question1", "question2"])

    # preprocessing the train data
    for i in df_training.index:
        train_1.append(clean_text(df_training['question1'][i]))
        train_2.append(clean_text(df_training['question2'][i]))
        labels.append(df_training['is_duplicate'][i])

    print("Checking whether all the train data values have properly been processed ")
    print(len(train_1))
    print(len(train_2))

    print("Writing the train dataframe to csv")
    traindata = {'question1': train_1, 'question2': train_2, 'labels': labels }
    dataframe_train=pd.DataFrame(data=traindata)
    dataframe_train.to_string
    dataframe_train.to_csv('PreproceesedDataTrainstr.csv')

    # preprocessing the test data

    for i in df_testing.index:
        test_1.append(clean_text(df_testing['question1'][i]))
        test_2.append(clean_text(df_testing['question2'][i]))
        print(test_1[i])

    print("Checking whether all the test data values have properly been processed ")
    print(len(test_1))
    print(len(test_2))

    # print("Writing the test dataframe to csv")
    testdata = {'question1': test_1, 'question2': test_2}
    dataframe_test=pd.DataFrame(data=testdata)
    dataframe_test.to_csv('PreprocessedDataTeststr.csv')



if __name__ == '__main__':
    main();


