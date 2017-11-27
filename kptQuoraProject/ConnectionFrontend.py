import pandas as pd
from kptQuoraProject.cleaning import clean_text
from xgboost import XGBClassifier as xgb
from sklearn.metrics import  log_loss, accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from  kptQuoraProject.Features import Features


class FrontEnd(object):

    def __init__(self):
        pass

    def preprocess(self):

        trainingfeatures = "/Users/pallavi/Documents/MS /RITCOURSES /FAll2017/KPT/PROJECT /CheckPoint3/TrainFeatureSet.csv"
        testingfeatures = "/Users/pallavi/Documents/MS /RITCOURSES /FAll2017/KPT/PROJECT /CheckPoint3/TestFeatureSet.csv"
        truelabels = "/Users/pallavi/Documents/MS /RITCOURSES /FAll2017/KPT/PROJECT /CheckPoint3/truelabels.csv"


        df_trainingfeatures = pd.read_csv(trainingfeatures, encoding='ISO-8859-1',
                                  usecols=["word_match", "tdidf"])
        df_testfeatures = pd.read_csv(testingfeatures, encoding='ISO-8859-1',
                                          usecols=["x_test_word_match", "x_test_tdidf"])
        df_truelabels = pd.read_csv(truelabels, encoding='ISO-8859-1',
                                          usecols=["labels"])
        return df_trainingfeatures,df_testfeatures,df_truelabels

    def trainforAllModel(self,x_train,y_train):
        X_train, X_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.4, random_state=4242)


        model = RandomForestClassifier(50, n_jobs=8)
        model.fit(X_train, y_train.values.ravel())
        predictions_proba = model.predict_proba(X_test)
        predictions = model.predict(X_test)

        log_loss_score = log_loss(y_test, predictions_proba)
        acc = accuracy_score(y_test, predictions)
        f1 = f1_score(y_test, predictions)

        print("RandonForest Classifier ")
        print('Log loss: %.5f' % log_loss_score)
        print('Acc: %.5f' % acc)
        print('F1: %.5f' % f1)

        modelXGB = xgb(n_estimators=500)
        modelXGB.fit(X_train, y_train.values.ravel())
        predictions_probaXGB = modelXGB.predict_proba(X_test)
        predictionsXGB = modelXGB.predict(X_test)

        log_loss_score_XGB = log_loss(y_test, predictions_probaXGB)
        acc_XGB = accuracy_score(y_test, predictionsXGB)
        f1_XGB = f1_score(y_test, predictionsXGB)

        print("XGBoost Classifier ")
        print('Log loss: %.5f' % log_loss_score_XGB)
        print('Acc: %.5f' % acc_XGB)
        print('F1: %.5f' % f1_XGB)

        clf = GaussianNB()
        clf.fit(X_train, y_train.values.ravel())
        predictions_probaNB = clf.predict_proba(X_test)
        predictionsNB = clf.predict(X_test)

        log_loss_score_Naiye_Bayes = log_loss(y_test, predictions_probaNB)
        acc_Naiye_Bayes = accuracy_score(y_test, predictionsNB)
        f1_Naiye_Bayes = f1_score(y_test, predictionsNB)

        print("Naiye_Bayes Classifier ")
        print('Log loss: %.5f' % log_loss_score_Naiye_Bayes)
        print('Acc: %.5f' % acc_Naiye_Bayes)
        print('F1: %.5f' % f1_Naiye_Bayes)


    def trainXgboost(self,x_train,y_train,user_test_data):
        X_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.2, random_state=4242)

        modelXGB = xgb(max_depth=4,n_estimators=500,learning_rate=0.05)
        modelXGB.fit(X_train, y_train.values.ravel())

        # predictions_probaXGB = modelXGB.predict_proba(x_valid)
        # predictionsXGB = modelXGB.predict(x_valid)
        # predictions = [round(value) for value in predictionsXGB]
        #
        #
        # # predictionsXGB=modelXGB.predict(text)
        # log_loss_score_XGB = log_loss(y_valid, predictions_probaXGB)
        # acc_XGB = accuracy_score(y_valid, predictions)
        # f1_XGB = f1_score(y_valid, predictions)
        #
        # print("XGBoost Classifier ")
        # print('Log loss: %.5f' % log_loss_score_XGB)
        # print('Acc: %.5f' % (acc_XGB * 100.0))
        # print('F1: %.5f' % f1_XGB)

        predictions_test = modelXGB.predict(user_test_data)
        predictions_test_binary = [round(value) for value in predictions_test]

        print('score for test data : ', predictions_test_binary)

        return predictions_test_binary

    def test(self,userinput):
        FT=Features()
        x_test = pd.DataFrame()
        user_test_data=pd.DataFrame()
        test_features = "/Users/pallavi/Documents/MS /RITCOURSES /FAll2017/KPT/PROJECT /CheckPoint3/testcase3.csv"
        df_test_features = pd.read_csv(test_features, encoding='ISO-8859-1',usecols=["question"])
        copyOF_df_test_features=df_test_features
        df_test_features=df_test_features['question'].apply(clean_text)
        x_test['question1'] = df_test_features
        x_test['question2'] = userinput


        # creating word match for input data
        x_test['word_match']=x_test.apply(FT.word_match,axis=1)

        # creating the TDIDF for input  data
        train_qs = pd.Series(x_test['question1'] + x_test['question2']).astype(str)
        words = (" ".join(train_qs)).split()
        FT.weights = FT.findweights(words)
        x_tdidf = x_test.apply(FT.tdIDF, axis=1)
        x_tdidf.fillna(x_tdidf.mean())
        x_test['tdidf'] = x_tdidf

        user_test_data['word_match']=x_test['word_match']
        user_test_data['tdidf']=x_test['tdidf']

        return user_test_data,copyOF_df_test_features

    def userinput(self,text):
        userText=clean_text(text)

        return userText

    def response(self,prediction,x_test):
        indices = [i for i, x in enumerate(prediction) if x == 1]
        responseList = []
        print(indices)
        if not indices:
            print("Indices None")
            responseList.append("No duplicate question present")
        else:
            for i in indices:
                if i <= len(x_test):
                    responseList.append(x_test.get_value(i,'question'))


        for p in responseList:
            print("Duplicate Question set :", p)

        return responseList


    def run(self,request):
        FE = FrontEnd()
        trainingfeatures,testfeatures,truelabels = FE.preprocess()
        trainingfeatures.fillna(trainingfeatures.mean(), inplace=True)
        userText = FE.userinput(request);
        user_test_data,x_test = FE.test(userText)
        print(user_test_data)
        prediction =FE.trainXgboost(trainingfeatures, truelabels,user_test_data)
        responseList=FE.response(prediction,x_test)

        return responseList




