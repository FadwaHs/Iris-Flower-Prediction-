from django.shortcuts import render
from django.http import JsonResponse
import pandas as pd
from .models import PredResults
# Create your views here.

#from django.http import HttpResponse


#def index(request):

    #message = "Salut tout le monde !"

    #return HttpResponse(message)

def predict(request):

    return render(request , 'predict.html')

def predict_chances(request):

    if request.POST.get('action') == 'post':

        # Receive data from client
        sepal_length = float(request.POST.get('sepal_length'))
        sepal_width = float(request.POST.get('sepal_width'))
        petal_length = float(request.POST.get('petal_length'))
        petal_width = float(request.POST.get('petal_width'))

        # from pandas import read_csv
        # from sklearn.model_selection import train_test_split
        # from sklearn.svm import SVC
        # import pandas as pd
        # from sklearn.metrics import accuracy_score

        # iris = pd.read_csv("/home/fadwa/Downloads/iris.csv")
        # df=iris
        # X = df.drop('variety', axis=1)
        # y = df['variety']

        # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=99)
        # model = SVC(gamma='auto')
        # model.fit(X_train, y_train)
        # predictions = model.predict(X_test)

        # score = accuracy_score(y_test, predictions)
        
        # Unpickle model
        # model= pd.read_pickle(r'/home/fadwa/Downloads/new_model.pickle')
        # # Make prediction
        # result = model.predict([[sepal_length, sepal_width, petal_length, petal_width]])
        # classification = result[0]

        import numpy as np
        from sklearn.datasets import load_iris
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import OneHotEncoder
        from keras.models import Sequential
        from keras.layers import Dense
        from keras.optimizers import Adam

        iris_data = load_iris() # load the iris dataset
        x = iris_data.data
        y = iris_data.target # Convert data to a single column
        z = iris_data.target_names

        train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.20)

        model = Sequential()
        model.add(Dense(100, input_shape=(4,), activation='relu'))
        model.add(Dense(100, activation='relu'))
        model.add(Dense(100, activation='relu', name='fc1'))
        model.add(Dense(3, activation='softmax', name='output'))
        model.compile(optimizer= 'Adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        model.fit(train_x, train_y,verbose=2, batch_size=30, epochs=350)

        results = model.evaluate(test_x, test_y)
        score = results[1]
        scorepr = score *100
        scorepr = format(scorepr, ".2f")
        result = model.predict([[sepal_length, sepal_width, petal_length, petal_width]])
        predicted = np.argmax(result,axis=1)
        classification = z[predicted[0]]


        PredResults.objects.create(sepal_length=sepal_length, sepal_width=sepal_width, petal_length=petal_length,
                                   petal_width=petal_width, classification=classification,score=scorepr)

        
        return JsonResponse({'result': classification, 'sepal_length': sepal_length,
                             'sepal_width': sepal_width, 'petal_length': petal_length, 'petal_width': petal_width,'score':score},
                            safe=False)


def view_results(request):
    # Submit prediction and show all
    data = {"dataset": PredResults.objects.all()}
    return render(request, "results.html", data)

