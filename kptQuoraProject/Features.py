from collections import Counter
import pandas as pd
from nltk.corpus import  stopwords
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import  log_loss, accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import svm


class Features(object):

    global weights

    def word_match(self,row):

        q1word = {}
        q2word = {}
        stops = set(stopwords.words("english"))

        for word in str(row['question1']).lower().split():
            if word not in stops:
                q1word[word] = 1

        for word in str(row['question2']).lower().split():
            if word not in stops:
                q2word[word] = 1

        if len(q1word) == 0 or len(q2word) == 0:
            #  the questions has only stop words
            return 0

        print('q1words', q1word)
        print('q2words', q2word)

        shared_word_q1 =[w for w in q1word.keys() if w in q2word]
        shared_word_q2 =[w for w in q2word.keys() if w in q1word]

        # print('',len(shared_word_q1))
        # print(len(shared_word_q2))
        # print(len(q1words))
        # print(len(q2words))

        responsefunction =(len(shared_word_q1) + len(shared_word_q2))/(len(q1word) + len(q2word))

        print('word_match responsefunction : ' , responsefunction)

        return responsefunction

    def get_weight(self,count , eps=10000, min_count=2):
        if count < min_count:
            return 0
        else:
            return 1 / (count + eps)

    def findweights(self,words):
        eps = 5000
        counts = Counter(words)
        weights = {word: self.get_weight(count) for word, count in counts.items()}

        return weights


    def tdIDF(self,row):

        # df_train = pd.read_csv('PreprocessedDataTrainstr.csv', encoding='ISO-8859-1', usecols=["question1", "question2"])
        # train_qs = pd.Series(df_train['question1'] + df_train['question2']).astype(str)
        # words = (" ".join(train_qs)).split()
        # eps = 5000
        # counts = Counter(words)
        # weights = {word: get_weight(count) for word, count in counts.items()}

        q1word = {}
        q2word = {}
        stops = set(stopwords.words("english"))

        for word in str(row['question1']).lower().split():
            if word not in stops:
                q1word[word] = 1

        for word in str(row['question2']).lower().split():
            if word not in stops:
                q2word[word] = 1

        if len(q1word) ==0 or len(q2word)==0:
            #  the questions has only stop words
            return 0


            #print(q1word)
            #print(q2word)

        sharedweights_q1 =[self.weights.get(word1) for word1 in q1word.keys() if word1 in q2word ]
        sharedweights_q2 =[self.weights.get(word2) for word2 in q2word.keys() if word2 in q1word ]

        # removing none value from the shared weights

        sharedweights_q1=[0 if i is None else i for i in sharedweights_q1]
        sharedweights_q2=[0 if i is None else i for i in sharedweights_q2]


        sharedweights = sharedweights_q1 + sharedweights_q2
        totalweights = [self.weights.get(word1,0) for word1 in q1word] + [self.weights.get(word2,0) for word2 in q2word]

        print('sharedweights_q1 ',sharedweights_q1)
        print('sharedweights_q2 ' ,sharedweights_q2)
        print('totalweights ' ,totalweights)


        num=np.sum(sharedweights)
        deno=np.sum(totalweights)
        responsefuntcion=num/deno
        # where_are_NaNs = np.isnan(responsefuntcion)
        # responsefuntcion[where_are_NaNs] = 0
        # print('where_are_NaNs',where_are_NaNs )

        # responsefuntcion.fillna(responsefuntcion.mean())
        print("num ", num)
        print("deno ", deno)
        print('tdIDF responsefuntcion : ', responsefuntcion)


        return responsefuntcion


    def input(self,text):
            pass

def main():

    Ft=Features()
    # initializing the dataframe for feature of train data and test data
    x_train = pd.DataFrame()
    x_test = pd.DataFrame()

    #  creating word match for train data
    df_train = pd.read_csv('PreproceesedDataTrainstr.csv', encoding='ISO-8859-1', usecols=["question1", "question2","labels"])
    x_train_word_match = df_train.apply(Ft.word_match, axis=1)
    print('x_train_word_match : ',x_train_word_match)

    # creating the TDIDF for train data
    train_qs = pd.Series(df_train['question1'] + df_train['question2']).astype(str)
    words = (" ".join(train_qs)).split()
    Ft.weights = Ft.findweights(words)
    x_train_tdidf = df_train.apply(Ft.tdIDF, axis=1)
    x_train_tdidf.fillna(x_train_tdidf.mean())
    print('x_train_tdidf ', x_train_tdidf)

    # creating word match for test data
    df_test=pd.read_csv('TestData100.csv', encoding='ISO-8859-1', usecols=["question1", "question2"])
    x_test_word_match = df_test.apply(Ft.word_match, axis=1)
    print('x_test_word_match : ',x_test_word_match)


    # create the TDIDF for test data
    Ft.weights.clear()
    test_qs = pd.Series(df_test['question1'] + df_test['question2']).astype(str)
    wordstest =(" ".join(test_qs)).split()
    Ft.weights = Ft.findweights(wordstest)
    print('weight test ', len(Ft.weights))
    x_test_tdidf =df_test.apply(Ft.tdIDF,axis=1)

    print('x_test_tdidf :', x_test_tdidf)

    # train data
    x_train['x_train_word_match']= x_train_word_match
    x_train['x_train_tdidf'] = x_train_tdidf
    x_train.fillna(x_train.mean())

    # print("Writing the trainfeatureset dataframe to csv")
    dataframe_train = pd.DataFrame(data=x_train)
    dataframe_train.to_csv('TrainFeatureSet.csv')


    # test data
    x_test['x_test_word_match'] = x_test_word_match
    x_test['x_test_tdidf'] = x_test_tdidf
    x_test.fillna(x_test.mean())

    print(" Writing the testfeatureset dataframe to csv")
    dataframe_test = pd.DataFrame(data=x_test)
    dataframe_test.to_csv('TestFeatureSet.csv')


    # True labels
    y_train = df_train['labels'].values

    print("Writing the true label dataframe to csv")
    dataframe_truelabels = pd.DataFrame(data=y_train)
    dataframe_truelabels.to_csv('truelabels.csv')


    X_train, X_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.4, random_state=42)

    model = RandomForestClassifier(50, n_jobs=8)
    model.fit(X_train, y_train)
    predictions_proba = model.predict_proba(X_test)
    predictions = model.predict(X_test)

    log_loss_score = log_loss(y_test, predictions_proba)
    acc = accuracy_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)

    print("RandonForest Classifier ")
    print('Log loss: %.5f' % log_loss_score)
    print('Acc: %.5f' % acc)
    print('F1: %.5f' % f1)


    modelXGB = XGBClassifier(n_estimators=500)
    modelXGB.fit(X_train,y_train)
    predictions_probaXGB = modelXGB.predict_proba(X_test)
    predictionsXGB=modelXGB.predict(X_test)

    log_loss_score_XGB = log_loss(y_test, predictions_probaXGB)
    acc_XGB = accuracy_score(y_test, predictionsXGB)
    f1_XGB = f1_score(y_test, predictionsXGB)

    print("XGBoost Classifier ")
    print('Log loss: %.5f' % log_loss_score_XGB)
    print('Acc: %.5f' % acc_XGB)
    print('F1: %.5f' % f1_XGB)

    clf = GaussianNB()
    clf.fit(X_train,y_train)
    predictions_probaNB = clf.predict_proba(X_test)
    predictionsNB = clf.predict(X_test)

    log_loss_score_Naiye_Bayes= log_loss(y_test, predictions_probaNB)
    acc_Naiye_Bayes = accuracy_score(y_test, predictionsNB)
    f1_Naiye_Bayes = f1_score(y_test, predictionsNB)

    print("SVM Classifier ")
    print('Log loss: %.5f' % log_loss_score_Naiye_Bayes)
    print('Acc: %.5f' % acc_Naiye_Bayes)
    print('F1: %.5f' % f1_Naiye_Bayes)


    modelSVM =svm.SVC(decision_function_shape='ovo')
    modelSVM.fit(X_train, y_train)
    predictionsSVM = modelSVM.predict(X_test)

    acc_SVM = accuracy_score(y_test, predictionsSVM)
    f1_SVM= f1_score(y_test, predictionsSVM)

    print("SVM  Classifier ")
    print('Acc: %.5f' % acc_SVM)
    print('F1: %.5f' % f1_SVM)





if __name__ == '__main__':
    main()



