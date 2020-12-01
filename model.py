import pandas as pd
import numpy as np
import pickle

dataset=pd.read_csv("hiring.csv")
dataset.head()

dataset.isnull().sum()
dataset['experience']=dataset.experience.replace(to_replace=np.nan, value=0)
dataset['test_score']=dataset.test_score.replace(to_replace=np.nan, value=dataset['test_score'].mean())

dataset.info()
# lets convert string into int:
    
def string_int(string_data):
    word_dict={'two':2, 'five':5, 'seven':7, 'three':3, 'ten':10, 'eleven':11, 0:0}
    return word_dict[string_data]

dataset['experience']=dataset['experience'].apply(lambda x: string_int(x))

dataset.info()

X=dataset.drop(['salary'], axis=1)
Y=dataset['salary']

# since we do have very less dataset, so use entire data to build the model, just doing for prectice:
    
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()

regressor.fit(X,Y)

#lets save the model into dist:
pickle.dump(regressor, open('model.pkl', 'wb'))    

# now lets load the model:

model=pickle.load(open('model.pkl','rb'))

print(model.predict([[2,9,6]]))






