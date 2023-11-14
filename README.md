# Effect of Wave Energy Convertors on Power Output

## Introduction

Wave energy is an innovative and up and coming energy source, that can directly target global warming as a renewable resource. Such energy is harvested in wave farms, and is a complicated process due to the complex hydrodynamic interactions betewen wave energy converters (known as WECs). In this blog, we attempt to predict the approximate total power output given the configuration of an array of 49 WECs using different models.

The email exchange that reiterates and confirms this problem statement can be found below.

![Head](/assets/email.png)

## Dataset

We reference the "Large-scale Wave Energy Farm" dataset from the UCI Machine Learning Repository – there are multiple sub-datasets on different WEC configurations, but we namely look at the ones with 49 WECs (_WEC_PERTH_49.csv_ and _WEC_SYDNEY_49.csv_). We perform training on our models with _WEC_PERTH_49.csv_, and use _WEC_SYDNEY_49.csv_ as further cross validation.

## Approach

### Load in Data

We import the above CSV datasets of wave farm configurations and power output into Google Drive, and then mount the data like so into our Jupyter Notebook. We must manually grant access to our Google account.

```python
import pandas as pd
from google.colab import drive

drive.mount('/content/gdrive', force_remount=True)
fulldf = pd.read_csv("gdrive/My Drive/WEC_Perth_49.csv")
verifydf = pd.read_csv("gdrive/My Drive/WEC_Sydney_49.csv")
```

### Library Imports

```python
import statistics
import random
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
```

### Data Cleaning

We mainly only care about the total power output as a result of the coordinates and the positions of WECs indexed from 1 through 49 – we thus drop the rest of the columns.

```python
#can drop the power1 - power49 columns and qW (won't be given as input)

num_positions = 49
columns_to_delete = ['Power'+str(i+1) for i in range(num_positions)] + ['qW']
fulldf.drop(columns_to_delete, axis=1, inplace=True)

fulldf
```

We end up with the following data, where the columns (X_i, Y_i) represent the position of WEC i, where 1 <= i <= 49, and and are left with the arrangement and the Total_Power output.

![Head](/assets/data0.png)

#### Data Investigation

We inspect what different arrangements of WECs could look like if mapped onto the coordinate plane, and choose two to display below.

![Head](/assets/coordinates.png)
![Head](/assets/coordinates1.png)

### Feature Engineering

For a given pandas row, which stores the entire arrangement and power output, we look at the average (X, Y) position, standard deviation for each X and Y coordinate, the extremes for the coordinates (min and max), and the overall hull area (based on the coordinates' geometry), creating them as additional column featuresin the dataframe.

```python
num_position_columns = num_positions * 2

mean_Xs, mean_Ys = [], []
min_Xs, min_Ys = [], []
max_Xs, max_Ys = [], []
std_Xs, std_Ys = [], []
areas = []

#convex hull area
def get_convex_hull(X_points, Y_points):
  points = np.column_stack((X_points, Y_points))
  hull = ConvexHull(points)
  area = hull.volume
  return area

#get 2 columns at a time (49 pairs)
def process_positions(row_idx):
  row = fulldf.iloc[[row_idx]]
  position_columns = row.iloc[:, : num_position_columns].values.flatten().tolist()

  all_Xs, all_Ys = [], []

  #iteratively take two columns
  for j in range(0, num_position_columns - 1, 2):
    i = j//2 + 1
    X_i, Y_i = position_columns[j], position_columns[j+1]
    all_Xs.append(X_i)
    all_Ys.append(Y_i)

  area = get_convex_hull(all_Xs, all_Ys)

  mean_X = sum(all_Xs) / num_positions
  mean_Y = sum(all_Ys) / num_positions
  std_X, std_Y = statistics.pstdev(all_Xs), statistics.pstdev(all_Ys)

  areas.append(area)
  mean_Xs.append(mean_X)
  mean_Ys.append(mean_Y)
  std_Xs.append(std_X)
  std_Ys.append(std_Y)
  min_Xs.append(min(all_Xs))
  min_Ys.append(min(all_Ys))
  max_Xs.append(max(all_Xs))
  max_Ys.append(max(all_Ys))


for i in range(len(fulldf.index)):
  process_positions(i)

fulldf['mean_X'], fulldf['mean_Y'] = mean_Xs, mean_Ys
fulldf['std_Xs'], fulldf['std_Ys'] = std_Xs, std_Ys
fulldf['min_X'], fulldf['min_Y'] = min_Xs, min_Ys
fulldf['max_X'], fulldf['max_Y'] = max_Xs, max_Ys
fulldf['area'] = areas
```

Our new data model would look like the below table.

```python
fulldf
```

### Data Train / Test Split

We split our original dataframes into train and test data in preparation for training on our models.

```python
#traindf, test_df. can shuffle first. measure difference from actual prediction
X = fulldf.drop(['Total_Power'], axis=1)
y = fulldf['Total_Power']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.33, random_state= 101)
```

### Model Engineering

#### Linear Regression

We first try a linear regression model that represents each WEC position as a variable, fitting to a final power output.

```python
# Instantiating LinearRegression() Model
lr = LinearRegression()

# Training/Fitting the Model
lr.fit(X_train, y_train)

# Making Predictions
lr.predict(X_test)
pred = lr.predict(X_test)

ave_y = sum(y) / len(y)

# Evaluating Model's Performance
rmse = np.sqrt(mean_squared_error(y_test, pred))
diff = max(y) - min(y) #determine range of values

print('R2:', r2_score(pred, y_test))
print('Average Power Output:', ave_y)
print('Root Mean Squared Error (RMSE):', rmse)
print('Normalized RMSE:', rmse / diff)

#relative RMSE of 0.05 means that, on average, the model's predictions are off by 5% of the average value of Y.
```

```
R2: 0.8831360957109651
Average Power Output: 3938246.455667426
Root Mean Squared Error (RMSE): 39582.626795134114
Normalized RMSE: 0.05018623435417899
```

Generally, the performance is quite good already, and we verify this with the scatterplot below – the predicted points are generally clustered around their actual value.

```python
plt.figure(figsize=(11,11))
plt.scatter(y_test, pred, c='crimson')
plt.yscale('log')
plt.xscale('log')

p1 = max(max(pred), max(y_test))
p2 = min(min(pred), min(y_test))
plt.plot([p1, p2], [p1, p2], 'b-')
plt.xlabel('True Values', fontsize=15)
plt.ylabel('Predictions', fontsize=15)
plt.title('Actual vs. Predicted Power Values')
plt.axis('equal')
plt.show()
```

![Head](/assets/reet0.png)

#### Random Forest

The next natural progression is to use an ensemble of decision trees, known as a Random Forest.

```python
rf_model = RandomForestRegressor(n_estimators=10, max_features="auto", random_state=42)
rf_model.fit(X_train, y_train)

# Make predictions
y_pred = rf_model.predict(X_test)
```

The predicting accuracy actually improves from the previous decision tree, as expected.

```python
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
diff = max(y) - min(y) #determine range of values

print('R2:', r2_score(y_pred, y_test))
print('Average Power Output:', ave_y)
print('Root Mean Squared Error (RMSE):', rmse)
print('Normalized RMSE:', rmse / diff)
```

```
R2: 0.9401886715594915
Average Power Output: 3938246.455667426
Root Mean Squared Error (RMSE): 28625.40062891788
Normalized RMSE: 0.03629372734357631
```

The spread of data around the 1:1 trendline appears to be more tight below.

![Head](/assets/ret.png)

```python
plt.figure(figsize=(11,11))
plt.scatter(y_test, y_pred, c='crimson')
plt.yscale('log')
plt.xscale('log')

p1 = max(max(y_pred), max(y_test))
p2 = min(min(y_pred), min(y_test))
plt.plot([p1, p2], [p1, p2], 'b-')
plt.xlabel('True Values', fontsize=15)
plt.ylabel('Predictions', fontsize=15)
plt.title('Actual vs. Predicted Power Values')
plt.axis('equal')
plt.show()
```

### Neural Net

We attempt to use a neural net as an additional model.

```python
# use minMax scaler
min_max_scaler = StandardScaler()
X_train = min_max_scaler.fit_transform(X_train)
X_test = min_max_scaler.transform(X_test)

# Define the model
model = Sequential([
    Conv1D(64, kernel_size=3, activation='relu', input_shape=(107, 1)),
    MaxPooling1D(pool_size=2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),  # Added dropout layer for regularization
    Dense(64, activation='relu'),
    Dense(1)  # Output layer with 1 neuron for regression
])

# Compile the model with a lower learning rate
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

# Compile the model
model.compile(optimizer=optimizer, loss='mean_squared_error')

# Print a summary of the model architecture
model.summary()

model.fit(X_train, y_train, batch_size = 10, epochs = 100)

y_pred = model.predict(X_test)
```

```python3
y_pred = model.predict(X_test)

plt.plot(y_test, color = 'red', label = 'Real data')
plt.plot(y_pred, color = 'blue', label = 'Predicted data')
plt.title('Prediction')
plt.legend()
plt.show()# decision tree
```

As seen below, there were far too many variables and features for the neural net to account for, leading to an overfitting that completely mispredicts the data.

![Head](/assets/nnfail.png)

### Cross Validation Using Winning Model: Random Forest

All our previous training was done on WEC_Perth_49, and so we test our results and best model on an unseen dataset: WEC_Sydney_49. We choose random arrangements and test their expected and actual value.

#### Data Processing

```python
#actual prediction – feed in to winning rf_model (or other models) and see predicted result. Crosscheck on WEC_Sydney_49 spreadsheet (used Perth as training).
#These power outputs have not been seen before, good way to cross validate accuracy.

#preprocess features
mean_Xs, mean_Ys = [], []
min_Xs, min_Ys = [], []
max_Xs, max_Ys = [], []
std_Xs, std_Ys = [], []
areas = []

for i in range(len(verifydf.index)):
  process_positions(i)

verifydf['mean_X'], verifydf['mean_Y'] = mean_Xs, mean_Ys
verifydf['std_Xs'], verifydf['std_Ys'] = std_Xs, std_Ys
verifydf['min_X'], verifydf['min_Y'] = min_Xs, min_Ys
verifydf['max_X'], verifydf['max_Y'] = max_Xs, max_Ys
verifydf['area'] = areas
```

#### Random Trials

```python
def predict(row_idx, errors, should_print):
  columns_to_delete = ['Power'+str(i+1) for i in range(num_positions)] + ['qW']
  actual_power = verifydf.iloc[[row_idx]]['Total_Power'].values[0]
  x_input = verifydf.iloc[[row_idx]].drop(columns=['Total_Power']+columns_to_delete)
  y_pred = rf_model.predict(x_input) # Predictions
  percentage_err = abs(y_pred[0]-actual_power) / actual_power

  if should_print:
    print("Input Coordinates and Features: \n \n", x_input)
    print("Prediction Power Output: ", y_pred[0])
    print("Actual Power Output: ", actual_power)
    print("Percentage Error: ", percentage_err)

  errors.append(percentage_err)
```

```python
#Random trials

n = len(verifydf.index)
random_row_idxes = [random.randint(0, n-1) for _ in range(500)]

err = []

for idx in random_row_idxes:
  predict(idx, err, True)
  print('-----------------')
  print('\n \n')
```

```
Input Coordinates and Features:

        X1   Y1    X2    Y2     X3   Y3    X4     Y4     X5   Y5  ...     Y49  \
9396  0.0  1.0  98.0  50.0  148.0  0.0  48.0  150.0  401.0  0.0  ...  1000.0

          mean_X      mean_Y      std_Xs      std_Ys  min_X  min_Y   max_X  \
9396  497.959184  492.857143  321.660104  344.193823    0.0    0.0  1000.0

       max_Y      area
9396  1000.0  956250.0

[1 rows x 107 columns]
Prediction Power Output:  3937234.3680000007
Actual Power Output:  4102475.53
Percentage Error:  0.04027840283059509
-----------------



Input Coordinates and Features:

         X1   Y1   X2    Y2   X3     Y3   X4     Y4     X5   Y5  ...     Y49  \
15643  1.0  1.0  1.0  51.0  1.0  101.0  1.0  151.0  601.0  1.0  ...  1000.0

           mean_X      mean_Y     std_Xs      std_Ys  min_X  min_Y   max_X  \
15643  478.571429  466.326531  293.29236  318.700767    0.0    0.0  1000.0

        max_Y      area
15643  1000.0  868750.0

[1 rows x 107 columns]
Prediction Power Output:  3676853.3090000004
Actual Power Output:  4077527.3
Percentage Error:  0.09826396281883863
-----------------



Input Coordinates and Features:

          X1   Y1      X2     Y2      X3     Y3     X4     Y4     X5   Y5  ...  \
8482  198.0  0.0  197.46  75.37  193.18  150.0  85.07  198.0  398.0  0.0  ...

         Y49      mean_X      mean_Y      std_Xs      std_Ys  min_X  min_Y  \
8482  1010.0  507.142857  482.653061  315.258236  325.850695    0.0    0.0

       max_X   max_Y      area
8482  1000.0  1000.0  936250.0

[1 rows x 107 columns]
Prediction Power Output:  3925104.22
Actual Power Output:  3997357.5
Percentage Error:  0.01807526096927778
-----------------



Input Coordinates and Features:

         X1   Y1   X2    Y2   X3     Y3   X4     Y4     X5   Y5  ...     Y49  \
14258  1.0  1.0  1.0  51.0  1.0  101.0  1.0  151.0  398.0  0.0  ...  1010.0

           mean_X      mean_Y      std_Xs      std_Ys  min_X  min_Y   max_X  \
14258  515.306122  496.938776  329.223589  334.177052    0.0    0.0  1000.0

        max_Y      area
14258  1000.0  910000.0

[1 rows x 107 columns]
Prediction Power Output:  3927853.9989999994
Actual Power Output:  4022594.49
Percentage Error:  0.023552085907620493
-----------------



Input Coordinates and Features:

         X1   Y1   X2    Y2   X3     Y3   X4     Y4     X5   Y5  ...     Y49  \
16279  1.0  1.0  1.0  51.0  1.0  101.0  1.0  151.0  448.0  0.0  ...  1000.0

           mean_X      mean_Y      std_Xs      std_Ys  min_X  min_Y   max_X  \
16279  474.489796  431.632653  288.225119  309.161783    0.0    0.0  1000.0

       max_Y      area
16279  950.0  822500.0

[1 rows x 107 columns]
Prediction Power Output:  3843551.357
Actual Power Output:  4110764.37
Percentage Error:  0.06500324245050325
-----------------



Input Coordinates and Features:

       X1     Y1    X2     Y2     X3     Y3     X4     Y4     X5   Y5  ...  \
940  1.0  198.0  50.0  198.0  100.0  198.0  150.0  198.0  201.0  1.0  ...

        Y49      mean_X      mean_Y      std_Xs      std_Ys  min_X  min_Y  \
940  1010.0  505.102041  492.857143  343.933563  333.809184    0.0    0.0

      max_X   max_Y      area
940  1000.0  1000.0  903750.0

[1 rows x 107 columns]
Prediction Power Output:  3901976.062
Actual Power Output:  4079797.83
Percentage Error:  0.043585926413417436
-----------------



Input Coordinates and Features:

           X1   Y1     X2     Y2     X3      Y3     X4     Y4     X5   Y5  ...  \
12618  398.0  0.0  397.6  72.29  393.0  147.07  349.0  198.0  798.0  0.0  ...

          Y49      mean_X      mean_Y      std_Xs      std_Ys  min_X  min_Y  \
12618  1000.0  463.265306  530.612245  285.495543  321.637444    0.0    0.0

        max_X   max_Y      area
12618  1000.0  1000.0  883750.0

[1 rows x 107 columns]
Prediction Power Output:  3858956.4329999997
Actual Power Output:  3958446.42
Percentage Error:  0.025133594456989063
-----------------



Input Coordinates and Features:

          X1   Y1      X2     Y2      X3     Y3    X4     Y4     X5   Y5  ...  \
9845  198.0  0.0  197.07  77.37  193.01  150.0  87.3  198.0  398.0  0.0  ...

         Y49      mean_X      mean_Y      std_Xs      std_Ys  min_X  min_Y  \
9845  1010.0  482.653061  518.367347  352.042295  317.627345    0.0    0.0

       max_X   max_Y      area
9845  1000.0  1000.0  972500.0

[1 rows x 107 columns]
Prediction Power Output:  3830421.1618
Actual Power Output:  3884207.33
Percentage Error:  0.013847398871985597
-----------------



Input Coordinates and Features:

          X1   Y1      X2     Y2     X3      Y3     X4     Y4     X5   Y5  ...  \
7247  198.0  0.0  197.46  75.65  193.6  144.02  149.0  198.0  598.0  0.0  ...

         Y49      mean_X      mean_Y      std_Xs      std_Ys  min_X  min_Y  \
7247  1010.0  480.612245  495.918367  310.663596  338.790877    0.0    0.0

       max_X   max_Y      area
7247  1000.0  1000.0  957500.0

[1 rows x 107 columns]
Prediction Power Output:  3955328.536000001
Actual Power Output:  3937192.59
Percentage Error:  0.004606314165597098
-----------------



Input Coordinates and Features:

           X1   Y1      X2     Y2      X3      Y3     X4     Y4     X5   Y5  \
12042  198.0  0.0  196.79  73.16  192.57  152.66  85.43  198.0  398.0  0.0

       ...     Y49      mean_X      mean_Y      std_Xs      std_Ys  min_X  \
12042  ...  1000.0  512.244898  491.836735  316.957895  302.598768    0.0

       min_Y   max_X   max_Y      area
12042    0.0  1000.0  1000.0  932500.0

[1 rows x 107 columns]
Prediction Power Output:  3929632.4310000003
Actual Power Output:  4036680.54
Percentage Error:  0.02651884585347933
-----------------
```

#### Conclusions

In this project, we are able to capably predict the overall power output of a certain coordinate arrangement of 49 wave energy converters to a great degree of accuracy, using a random forest model. We also tested linear regression and neural net models, which were not nearly as successful. The greatest potential error achieved from this model is generally <= 10%, which is quite reliable for estimating the overall power generated from a specific set of WECs.

### Reflection

As usual, ChatGPT was extremely helpful in terms of helping give suggestions for implementing approaches and high level ideas. For example, it suggested the statistic of Normalized RMSE after performing regression, which measures the relative error – this was found to be a more accurate statistic of how inaccurate a prediction was, due to the inherent large power output values.

![Head](/assets/chat1.png)

Moreover, ChatGPT had great ideas when it came to feature engineering brainstorming. When I wanted to derive a feature related to the geometry of the scatterplot, it suggested the convex hull area (a library implementation that existed) that improved the overall accuracy of my models.

![Head](/assets/chat15.png)

![Head](/assets/chat2.png)

However, ChatGPT wasn't very helpful in terms of engineering the neural net model. Although it knew that the model was overfitting and not predicting correctly, the neural net structure it suggested did not make any overall changes – however, this may be due to the structure of the data itself.

Overall, it still relied on me for driving the direction of the project. If the overall project was represented as a graph with nodes and edges, all the edges and main conclusions and output (nodes) were pushed onwards by me, while there may be some sub-edges from the nodes that don't lead to any other node that were created by ChatGPT.
