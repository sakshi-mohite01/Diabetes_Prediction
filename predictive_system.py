import numpy as np
import pickle

# laoding the save model using pickle.load
load_model = pickle.load(open('trained_model.sav','rb'))

input_data = (1,103,30,38,83,43.3,0.183,33)
arrayed_data = np.asarray(input_data)

reshaped_data = arrayed_data.reshape(1,-1)

#standardisation
# std_dat = scaler.transform(reshaped_data)
# print(std_dat)

#predicting the y value
prediction = load_model.predict(reshaped_data)
print(prediction)

if prediction[0] == 0:
    print('The person is not diabetic')
else:
    print('The person is diabetc')