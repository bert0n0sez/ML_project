from classes_oop import Primary_Data_Processing, Machine_Learning, Scores


def main():
    primary = Primary_Data_Processing()
    machine = Machine_Learning()
    score = Scores()
    data = primary.None_Processing() 
    type = machine.lin_svm(data)
    final_score = score.f1_macro(type)
    print('Наконец-то у меня спустя 5 часов дебага получился ответ:', final_score)

if __name__ == '__main__':
    main()