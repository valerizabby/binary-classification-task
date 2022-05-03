# Задача бинарной классификации для предсказания сердечного приступа

как вставлять картинки в гитхаб: https://ru.numberempire.com/latexequationeditor.php

## Постановка задачи
Сравнить методы классификации на примере открытого датасета ![Personal Key Indicators of Heart Disease](https://www.kaggle.com/datasets/kamilpytlak/personal-key-indicators-of-heart-disease)

___

## Анализ данных
Датасет собран на основе опросов людей в США об их здоровье, бинарным признаком (вектором ответов) является стобец HeartDisease, который описывает, бывали ли у человека болезни сердца. Изначально датасет имел более 300 признаков, но был сжат до примерно 20 самых существенных, таких как курение, сахарный диабет, генетика etc. Автор предлагает его использовать в целях построения учебных моделей, так как датасет уже подчищен, единственный недостаток - несбалансированность. Так, для вектора ответов имеет место следующее распределение: 

![HeartDisease](https://github.com/valerizabby/binary-classification-task/blob/main/pictures/%20HeartDisease.png)

Но это легко поправить с помощью UnderSampling. Метод больший класс ужимает до размеров меньшего. 

![Undersampling](https://github.com/valerizabby/binary-classification-task/blob/main/pictures/Undersampling.png)

Недостаток метода очевиден: в выкинутых значениях может содеражться критически важная информация для модели. Зато скорость обучения возрастет, что в учебных целях приоритенее. Делим данные 7 к 3 на обучение и тест и начинаем обучение. 

## Методы классификации 

![Тут можно потрогать основы](https://tproger.ru/translations/scikit-learn-in-python/)

- **Метод опорных векторов (Support Vector Machines)**

https://habr.com/ru/company/ods/blog/484148/

https://github.com/esokolov/ml-course-hse/blob/master/2021-fall/lecture-notes/lecture06-linclass.pdf

SVM  - алгоритм из семейства линейных классификаторов, использующийся для задач классификации и регрессии. Цель SVM - найти уравнение разделяющей гиперплоскости, которая разделила бы классы оптимальным образом. 
Для понимания работы алгоритма, рассмотрим модель задачи бинарной классификации. Предположим, выборка данных линейно разделима. Это значит, что существует такое положение гиперплоскости, что любых объектов выборки положительны. Тогда нам интересно максимизировать ширину "полосы" между классами:

![SVM](https://github.com/valerizabby/binary-classification-task/blob/main/pictures/svm.png)

- **Метод k-ближайших соседей (K-Nearest Neighbors)**
- **Классификатор дерева решений (Decision Tree Classifier)**

## Обучение с помощью каждого метода, результаты
- Support Vector Machines
``` 
svclassifier = SVC(kernel = 'sigmoid')
svclassifier.fit(X_train, y_train)
```
- K-Nearest Neighbors
Модель:
```
KNN = KNeighborsClassifier(n_neighbors = 15)
KNN.fit(X_train, y_train)
```
Количество соседей выбрано экспериментальным путем, запустили модель с количеством соседей от 1 до 40 и посмотрим, где минимальная ошибка:
- Decision Tree Classifier
```
classifier_RF = RandomForestClassifier(n_estimators = 80, random_state = 0)
classifier_RF.fit(X_train, y_train)
```

## Сравнительная сводка 

## Список литературы 
