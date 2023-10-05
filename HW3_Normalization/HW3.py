import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D

dataframe=pd.read_excel('worldwide.xlsx')
# print(dataframe)
X0=dataframe.values[:,2]
X1=dataframe.values[:,4]
Y=dataframe.values[:,0]

# print(X1)
# print(Y)
# Tạo một đối tượng subplot 3D
fig = plt.figure() 
ax0 = fig.add_subplot(121, projection='3d')

# Vẽ biểu đồ 3D
ax0.scatter(X0, X1, Y, c='r', marker='o')

# Đặt nhãn cho các trục
ax0.set_xlabel('Cost Index')
ax0.set_ylabel('Purchasing Power Index')
ax0.set_zlabel('Rank')
plt.title('Raw Data')

def Normalization(X):
    return (X-X.min())/(X.max()-X.min())
# Normalization
X0_min=X0.min()
X0_max=X0.max()
X0_normalization=Normalization(X0)
X1_min=X1.min()
X1_max=X1.max()
X1_normalization=Normalization(X1)
X=[X0_normalization,X1_normalization]

ax1 = fig.add_subplot(122, projection='3d')

# Vẽ biểu đồ 3D
ax1.scatter(X0_normalization, X1_normalization, Y, c='r', marker='o')

# Đặt nhãn cho các trục
ax1.set_xlabel('Cost Index')
ax1.set_ylabel('Purchasing Power Index')
ax1.set_zlabel('Rank')
plt.title('Normalization Data')


def Predict(X,weight,bias):
    return X[0]*weight[0]+X[1]*weight[1]+bias
def Cost_function(X,Y,weight,bias):
    n=len(Y)
    sum_error=0
    for i in range(n):
        sum_error=sum_error+(Y[i]-(X[0][i]*weight[0]+X[1][i]*weight[1]+bias))**2
    return sum_error/n
def Update_weight_bias(X,Y,weight,bias,learning_rate_w0,learning_rate_w1,learning_rate_b):
    n=len(Y)
    gradient_weight=[0.0,0.0]
    gradient_bias=0.0
    for i in range(n):
        gradient_weight[0]+=(-2*X[0][i]*(Y[i]-weight[0]*X[0][i]-weight[1]*X[1][i]-bias))
        gradient_weight[1]+=(-2*X[1][i]*(Y[i]-weight[0]*X[0][i]-weight[1]*X[1][i]-bias))
        gradient_bias+=(-2*(Y[i]-weight[0]*X[0][i]-weight[1]*X[1][i]-bias))
    new_weight=[weight[0]-(gradient_weight[0]/n)*learning_rate_w0,weight[1]-(gradient_weight[1]/n)*learning_rate_w1]
    new_bias=bias-(gradient_bias/n)*learning_rate_b
    return new_weight,new_bias
def Train(X,Y,weight,bias,learning_rate_w0,learning_rate_w1,learning_rate_b,iter):
    Cost_his=[]
    for i in range(iter):
        weight,bias=Update_weight_bias(X,Y,weight,bias,learning_rate_w0,learning_rate_w1,learning_rate_b)
        Cost_his.append(Cost_function(X,Y,weight,bias))
    return weight,bias,Cost_his

weight,bias,cost=Train(X,Y,weight=[0.03,0.01],bias=0.0014,learning_rate_w0=0.001,learning_rate_w1=0.001,learning_rate_b=0.001,iter=1000)
# print(weight,bias)
x0_plot=np.linspace(np.min(X0_normalization),np.max(X0_normalization),100)
x1_plot=np.linspace(np.min(X1_normalization),np.max(X1_normalization),100)
x0_plot, x1_plot = np.meshgrid(x0_plot, x1_plot)
y_final=weight[0]*x0_plot+weight[1]*x1_plot+bias

fig_result = plt.figure() 
ax_result = fig_result.add_subplot(111, projection='3d')
# Vẽ mặt phẳng
surf = ax_result.plot_surface(x0_plot, x1_plot, y_final, cmap='viridis')
ax_result.scatter(X0_normalization, X1_normalization, Y, c='r', marker='o')
# Đặt tên cho các trục

ax_result.set_xlabel('Cost Index')
ax_result.set_ylabel('Purchasing Power Index')
ax_result.set_zlabel('Rank')

plt.title('Result')
plt.show()