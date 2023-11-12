import numpy as np
import csv
import pandas as pd
import torch 


#t1 = torch.tensor([[5.,6.],
 #                 [7,10],
  #                [6,3]])
#print(t1)
#t1.shape

# Create tensors.
#x = torch.tensor(2.)
#w = torch.tensor(4., requires_grad=True)
#b = torch.tensor(5., requires_grad=True)
#x, w, b

# Arithmetic operations
#y = w * x + b
#y

# Compute derivatives
#y.backward()

# Display gradients
#print('dy/dx:', x.grad)
#print('dy/dw:', w.grad)
#print('dy/db:', b.grad)


input_list = []
targets_list = []

df = pd.read_csv("realtor-data.zip.csv")
df = df[df['house_size'].notna()]
df = df[df['bed'].notna()]
df = df[df['price'].notna()]
df = df[df['bath'].notna()]
df = df[df['city'].notna()]

X = df[['house_size', 'bed', 'bath',]]
Y = df['price']


"""with open("realtor-data.zip.csv") as csv_file:
csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            if row[0] != 'status':
                 input_list.append([float(row[1]), float(row[2]), float(row[7])])
                 targets_list.append(float(row[9])) """


# Input (bed, bath, squareft)

inputs = np.array([input_list], dtype='float32')
print(type(inputs[0][0]))

targets = np.array([targets_list], dtype='float32')

"""
inputs = np.array([[73, 67, 43],
                   [91, 88, 64],
                   [87, 134, 58],
                   [102, 43, 37],
                   [69, 96, 70]], dtype='float32')
"""
# Targets (apples, oranges)

""" targets = np.array([[56, 70],
                    [81, 101],
                    [119, 133],
                    [22, 37],
                    [103, 119]], dtype='float32')"""

inputs = torch.from_numpy(inputs)
targets = torch.from_numpy(targets)

 
w = torch.randn(1, 3, requires_grad=True)
b = torch.randn(1, requires_grad=True)
print(w)
print(b)

def model(x):
    return x @ w.t() + b

# # Generate predictions
# preds = model(inputs)
# print(preds)
# print("here")
# print(targets)

# MSE loss
def mse(t1, t2):
    diff = t1 - t2
    return torch.sum(diff * diff) / diff.numel()

# # Compute loss
# loss = mse(preds, targets)
# print(loss)

# # Compute gradients
# loss.backward()

# Gradients for weights
"""
print(w)
print(w.grad)
print(b.grad)
"""
"""
with torch.no_grad():
    w -= w.grad * 1e-5
    b -= b.grad * 1e-5
    w.grad.zero_()
    b.grad.zero_()
    """
"""
print(w)
print(b)
"""
""" 
# Let's verify that the loss is actually lower
preds = model(inputs)
loss = mse(preds, targets)
print(loss)
"""
print("strart")
"""
""" 
for i in range(5):
    preds = model(inputs)
    loss = mse(preds, targets)
    loss.backward()
    with torch.no_grad():
        w -= w.grad * 1e-5
        b -= b.grad * 1e-5
        w.grad.zero_()
        b.grad.zero_()
    
print(loss)
print(preds)
print(targets)

    

#calculate mean x, mean y

#calculate s_xx and s_xy
# find beta
# test linear model against prediction data
#