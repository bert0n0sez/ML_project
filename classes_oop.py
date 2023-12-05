import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import tree
from sklearn import svm
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier


class Primary_Data_Processing(object):
    def __init__(self):
        self.df = pd.read_csv('microarray_BRCA_public.csv')
        self.y = self.df.type 
        self.X = self.df.drop(self.df.columns[0], axis=1)
        le = preprocessing.LabelEncoder()
        le.fit(self.y)
        self.y = le.transform(self.y)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, train_size=0.9, test_size=0.1)
        dict = { 1 : 'basal' , 3 : 'luminal_A', 4 : 'luminal_B', 0 : 'HER', 2 : 'cell_line', 5 : 'normal'}

    def pca(self):
        df = pd.read_csv('microarray_BRCA_public.csv')
        pca = PCA(n_components=120)
        X = df.drop(df.columns[0], axis=1)
        principalComponents = pca.fit_transform(X)
        principalDf = pd.DataFrame(data = principalComponents)
        finalDF = pd.concat([principalDf, df.type], axis = 1)
        self.X_train = finalDF.drop(finalDF.columns[0], axis=1)
        return self.X_train

    
    def None_Processing(self):
        return self.X_train


class Machine_Learning(Primary_Data_Processing):

    def random_forest(self, X_train):
        self.clf = RandomForestClassifier(n_estimators=10)
        self.clf.fit(X_train, self.y_train)
        prediction = self.clf.predict(self.X_test)
        return prediction
    
    def gradient_boosting(self, X_train):
        self.clf = GradientBoostingClassifier(n_estimators=10, learning_rate=1.0, max_depth=1, random_state=0).fit(X_train, self.y_train)
        prediction = self.clf.predict(self.X_test)
        return prediction

    def svm(self, X_train):
        clf = svm.SVC(decision_function_shape='ovo')
        clf.fit(X_train, self.y_train)
        prediction = clf.predict(self.X_test)
        return prediction

    def lin_svm(self, X_train):
        self.clf = svm.LinearSVC(max_iter=10000, dual=False).fit(X_train, self.y_train)
        self.clf.fit(X_train, self.y_train)
        prediction = self.clf.predict(self.X_test)
        return prediction
        
    def decision_tree(self, X_train):
        self.clf = tree.DecisionTreeClassifier()
        self.clf = self.clf.fit(X_train, self.y_train)
        prediction = self.clf.predict(self.X_test)
        return prediction
    

class Scores(Primary_Data_Processing):

    def accuracy_score(self, prediction):
        scores = accuracy_score(self.y_test, prediction)
        return scores

    
    def f1_macro(self, prediction):
        scores = f1_score(self.y_test, prediction, average='macro')
        return scores
    

    def f1_micro(self, prediction):
        scores = f1_score(self.y_test, prediction, average='micro')
        return scores
    
    def f1_weighted(self, prediction):
        scores = f1_score(self.y_test, prediction, average='weighted')
        return scores
    

    
def main():
    primary = Primary_Data_Processing()
    machine = Machine_Learning()
    score = Scores()
    data = primary.None_Processing() 
    type = machine.random_forest(data)
    final_score = score.accuracy_score(type)
    print('Наконец-то у меня спустя 5 часов дебага получился ответ:', final_score)

if __name__ == '__main__':
    main()

    


    

    






    





