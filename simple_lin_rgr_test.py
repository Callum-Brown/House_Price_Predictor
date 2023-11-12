import numpy as np
import csv
import pandas as pd
from sklearn import linear_model

df = pd.read_csv("realtor-data.zip.csv")
df = df[df['house_size'].notna()]
df = df[df['bed'].notna()]
df = df[df['price'].notna()]
df = df[df['bath'].notna()]
df = df[df['city'].notna()]

#.loc[:, lambda df: ['house_price']]
X = df[['house_size', 'bed', 'bath',]]
Y = df['price']

reg = linear_model.LinearRegression()
reg.fit(X,Y)

predictedprice = reg.predict([[2300, 3, 2]])

print(predictedprice)
print(reg.coef_)







# x1 = df.at[1, 'house_size']
# x2 = df.at[2, 'house_size']
# x3 = df.at[3, 'house_size']
# x4 = df.at[4, 'house_size']

# x_list = [x1,x2,x3,x4]

# y1 = df.at[1, 'price']
# y2 = df.at[2, 'price']
# y3 = df.at[3, 'price']
# y4 = df.at[4, 'price']
# print(y1)
# print(y2)
# print(y3)
# print(y4)

# y_list = [y1,y2,y3,y4]

# mean_y = (float(y1) + float(y2) + float(y3) + float(y4)) / 4
# mean_x = (float(x1) + float(x2) + float(x3) + float(x4)) / 4

# print(mean_y)

# def var_calc_sxx(lis, mean):
#     print(type(mean))
#     print(type(lis[0]))
#     S = 0
#     for i in lis:
#         print(i)
#         S = S + (float(i) - float(mean))**2
#     print(S)
#     return S

# def var_calc_sxy(lis1, lis2, mean1, mean2):
#     S = 0
#     for i,l in zip(lis1,lis2):
#         S = S + (float(i) - float(mean1)) * (float(l) - float(mean2)) 
#     print(S) 
#     return S      


# S_xx = var_calc_sxx(x_list, mean_x)
# S_xy = var_calc_sxy(x_list, y_list, mean_x, mean_y)

# print(S_xx)
# print(S_xy)
# B_theta = S_xy // S_xx
# print(B_theta)

