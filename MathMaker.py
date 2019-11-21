import pandas as pd

import statsmodels.api as sm


df = pd.read_csv('C:/Python/abalone_data.csv')


# x is the collection of explanatory variables...Note: capital letters and spaces matter
X = df[['Lengthmm', 'Rings']]

# y is the dependent variable
y = df['Shucked_weight_grams']

#this line prints the first few rows of data so you can see what it looks like and make sure  you have the right data
print(df.head())

#the line below adds a constant to the explanatory variables so it will run your regression using a constant (and not force best fit line through the origin)
X = sm.add_constant(X)

# creates this set of results called "est" from the OLS regression from the sm package.  (dep varibale, explanatory variables)
est = sm.OLS(y, X).fit()
print(est.summary())


