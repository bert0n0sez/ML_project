from classes_oop_SirGay import Primary_Data_Processing, Machine_Learning, Scores
def main():
    primary = Primary_Data_Processing()
    machine = Machine_Learning()
    score = Scores()
    X_train, X_test, y_train, y_test = primary.pca()
    prediction = machine.lin_svc(X_train, X_test, y_train)
    final_score = score.f1_weighted(prediction, y_test)
    print(final_score)

if __name__ == '__main__':
    main()