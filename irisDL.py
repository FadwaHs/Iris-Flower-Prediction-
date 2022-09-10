import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
#from sklearn.preprocessing import OneHotEncoder

from keras.models import Sequential
from keras.layers import Dense
#from keras.optimizers import Adam

iris_data = load_iris() # load the iris dataset
x = iris_data.data
y = iris_data.target # Convert data to a single column
z = iris_data.target_names


# Split the data for training and testing
train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.20)

# Build the model

model = Sequential()
model.add(Dense(100, input_shape=(4,), activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu', name='fc1'))
model.add(Dense(3, activation='softmax', name='output'))


model.compile(optimizer= 'Adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(train_x, train_y,verbose=2, batch_size=20, epochs=100)

results = model.evaluate(test_x, test_y)

print('Final test set loss: {:4f}'.format(results[0]))
print('Final test set accuracy: {:4f}'.format(results[1]))



# Take input from user
sepal_length = float(input("Enter sepal_length: "))
sepal_width = float(input("Enter sepa_width: "))
petal_length = float(input("Enter petal_length: "))
petal_width = float(input("Enter petal_width: "))

result = model.predict([[sepal_length,sepal_width,petal_length,petal_width]])  # input must be 2D array


predicted = np.argmax(result,axis=1)

classification = z[predicted[0]]

for i in predicted :
    print("{}".format(z[i]) )























           
              
              
              
              
              
              
              
              
              