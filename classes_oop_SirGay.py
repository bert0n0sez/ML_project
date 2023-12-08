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
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


class Primary_Data_Processing(object):
    def pca(self):
        df = pd.read_csv('microarray_BRCA_public.csv')
        pca = PCA(n_components=120)
        X = df.drop(df.columns[0], axis=1)
        principalComponents = pca.fit_transform(X)
        principalDf = pd.DataFrame(data = principalComponents)
        y = df.type
        le = preprocessing.LabelEncoder()
        le.fit(y)
        y = le.transform(y)
        X_train, X_test, y_train, y_test = train_test_split(principalDf, y, train_size=0.9, test_size=0.1)
        return X_train, X_test, y_train, y_test

    def None_Processing(self):
        df = pd.read_csv('microarray_BRCA_public.csv')
        y = df.type 
        X = df.drop(df.columns[0], axis=1)
        le = preprocessing.LabelEncoder()
        le.fit(y)
        y = le.transform(y)
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.9, test_size=0.1)
        return X_train, X_test, y_train, y_test


class Machine_Learning(Primary_Data_Processing):
    
    def gradient_boosting(self, X_train, X_test, y_train):
        clf = GradientBoostingClassifier(n_estimators=10, learning_rate=1.0, max_depth=1, random_state=0).fit(X_train, y_train)
        prediction = clf.predict(X_test)
        return prediction

    def lin_svc(self, X_train, X_test, y_train):
        clf = svm.LinearSVC(max_iter=10000, dual=False).fit(X_train, y_train)
        clf.fit(X_train, y_train)
        prediction = clf.predict(X_test)
        return prediction
    
    def log_reg(self, X_train, X_test, y_train):
        clf = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000)
        clf = clf.fit(X_train, y_train)
        prediction = clf.predict(X_test)
        return prediction

class Scores(Primary_Data_Processing):
    def accuracy_score(self, prediction, y_test):
        scores = accuracy_score(y_test, prediction)
        return scores

    
    def f1_macro(self, prediction, y_test):
        scores = f1_score(y_test, prediction, average='macro')
        return scores
    

    def f1_micro(self, prediction, y_test):
        scores = f1_score(y_test, prediction, average='micro')
        return scores
    
    def f1_weighted(self, prediction, y_test):
        scores = f1_score(y_test, prediction, average='weighted')
        return scores


def main():
    primary = Primary_Data_Processing()
    machine = Machine_Learning()
    score = Scores()
    X_train, X_test, y_train, y_test = primary.pca()
    prediction = machine.lin_svc(X_train, X_test, y_train)
    final_score = score.f1_weighted(prediction, y_test)
    print('Наконец-то у меня спустя 5 часов дебага получился ответ:', final_score)

if __name__ == '__main__':
    main()

    

    






    





