from classes_oop import Primary_Data_Processing, Machine_Learning, Scores


def main():
    primary = Primary_Data_Processing()
    machine = Machine_Learning()
    score = Scores()
    X_train, X_test, y_train, y_test = primary.None_Processing()
    prediction = machine.lin_svm(X_train, X_test, y_train)
    final_score = score.accuracy_score(prediction, y_test)
    print('Наконец-то у меня спустя 5 часов дебага получился ответ:', final_score)

if __name__ == '__main__':
    main()
