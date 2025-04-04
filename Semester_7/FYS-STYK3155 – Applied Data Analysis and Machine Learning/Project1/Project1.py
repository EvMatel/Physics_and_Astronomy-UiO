import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import *
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn import linear_model
from imageio import imread


def R2(y_data, y_model):
   return 1 - (np.sum((y_data - y_model) ** 2) / np.sum((y_data - np.mean(y_data)) ** 2))


def MSE(y_data, y_model):
   n = np.size(y_model)
   return np.sum((y_data - y_model) ** 2) / n


def FrankeFunction(x, y):
   term1 = 0.75 * np.exp(-(0.25 * (9 * x - 2) ** 2) - 0.25 * ((9 * y - 2) ** 2))
   term2 = 0.75 * np.exp(-((9 * x + 1) ** 2) / 49.0 - 0.1 * (9 * y + 1))
   term3 = 0.5 * np.exp(-(9 * x - 7) ** 2 / 4.0 - 0.25 * ((9 * y - 3) ** 2))
   term4 = -0.2 * np.exp(-(9 * x - 4) ** 2 - (9 * y - 7) ** 2)
   return term1 + term2 + term3 + term4


# taken from week 37
def Bias_Variance(z_test, z_pred):
   bias = np.mean((z_test - np.mean(z_pred, keepdims=True)) ** 2)
   variance = np.mean(np.var(z_pred, keepdims=True))

   return bias, variance


# taken from lecture notes
def create_X(x, y, n):
   if len(x.shape) > 1:
       x = np.ravel(x)
       y = np.ravel(y)

   N = len(x)
   l = int((n + 1) * (n + 2) / 2)  # Number of elements in beta
   X = np.ones((N, l))

   for i in range(1, n + 1):
       q = int((i) * (i + 1) / 2)
       for k in range(i + 1):
           X[:, q + k] = (x ** (i - k)) * (y ** k)

   return X



def OLS(N, n, data=None, bootstrap_iter=None):

   if data:

       terrain = read_data()
       z_data = terrain[:N, :N]

       # scaling the z-data
       scaler = MinMaxScaler()
       scaler.fit(z_data)
       z_data = scaler.transform(z_data)

       # both x and y are innately scaled
       x = np.sort(np.linspace(0, 1, np.shape(z_data)[0]))
       y = np.sort(np.linspace(0, 1, np.shape(z_data)[1]))
       x_mesh, y_mesh = np.meshgrid(x, y)
       X_data = create_X(x_mesh, y_mesh, n)


       z_data = z_data.reshape(-1)

       X_train, X_test, z_train, z_test = train_test_split(X_data, z_data, test_size=0.2)

   else:
       # Make data set
       x = np.sort(np.random.uniform(0, 1, N))
       y = np.sort(np.random.uniform(0, 1, N))

       X = create_X(x, y, n)

       # we have to explain our choice to not scale the data

       z = FrankeFunction(x, y)

       X_train, X_test, z_train, z_test = train_test_split(X, z, test_size=0.2)


   if bootstrap_iter:

       # taken from week 37
       def bootstrap(data, z):
           datapoints = len(data)
           rand_idx = np.random.randint(0, datapoints, datapoints)
           new_data = data[rand_idx]
           new_z = z[rand_idx]
           return new_data, new_z

       z_pred = np.empty((z_test.shape[0], bootstrap_iter))
       for i in range(bootstrap_iter):
           new_X_train, new_z_train = bootstrap(X_train, z_train)
           OLS_beta = np.linalg.pinv(new_X_train.T @ new_X_train) @ new_X_train.T @ new_z_train
           z_pred[:, i] = X_test @ OLS_beta

       return z_pred, z_test  # returns a 2D array of predictions for every bootstrap iteration

   else:

       OLS_beta = np.linalg.pinv(X_train.T @ X_train) @ X_train.T @ z_train
       OLS_ztilde = X_train @ OLS_beta
       OLS_zpredict = X_test @ OLS_beta

       return z_train, OLS_ztilde, z_test, OLS_zpredict, OLS_beta


def OLS_analysis(N, n, data=None):
   MSE_train_list = []
   R2_train_list = []
   MSE_test_list = []
   R2_test_list = []
   OLS_beta_list = []
   for n in range(n):
       if data:
           z_train, OLS_ztilde, z_test, OLS_zpredict, OLS_beta = OLS(N, n, data=1)
       else:
           z_train, OLS_ztilde, z_test, OLS_zpredict, OLS_beta = OLS(N, n)
       MSE_train_list.append(MSE(z_train, OLS_ztilde))
       R2_train_list.append(R2(z_train, OLS_ztilde))
       MSE_test_list.append(MSE(z_test, OLS_zpredict))
       R2_test_list.append(R2(z_test, OLS_zpredict))
       OLS_beta_list.append(OLS_beta)

   fig, axs = plt.subplots(1, 2, figsize=(10, 4))

   if data:
       fig.suptitle('OLS Performance on Topographic Data', fontsize=16)
   else:
       fig.suptitle('OLS Performance on Synthetic Data', fontsize=16)

   axs[0].plot(np.arange(1, n+2), MSE_train_list, label='MSE')
   axs[0].plot(np.arange(1, n+2), R2_train_list, label='R2')
   axs[0].set_xlabel('nth order polynomial')
   axs[0].set_title('Training Data')
   axs[0].legend()

   axs[1].plot(np.arange(1, n + 2), MSE_test_list, label='MSE')
   axs[1].plot(np.arange(1, n + 2), R2_test_list, label='R2')
   axs[1].set_xlabel('nth order polynomial')
   axs[1].set_title('Test Data')
   axs[1].legend()

   plt.tight_layout()
   plt.show()

   return OLS_zpredict


def Ridge(N, n, data=None):
    if data:
        terrain = read_data()
        z_data = terrain[:N, :N]

        # scaling the z-data
        scaler = MinMaxScaler()
        scaler.fit(z_data)
        z_data = scaler.transform(z_data)

        x = np.linspace(0, 1, np.shape(z_data)[0])
        y = np.linspace(0, 1, np.shape(z_data)[1])
        x_mesh, y_mesh = np.meshgrid(x, y)
        X_data = create_X(x_mesh, y_mesh, n)
        z_data = z_data.flatten()

        X_train, X_test, z_train, z_test = train_test_split(X_data, z_data, test_size=0.2)

    else:
        x = np.sort(np.random.uniform(0, 1, N))
        y = np.sort(np.random.uniform(0, 1, N))

        X = create_X(x, y, n)

        z = FrankeFunction(x, y)

        X_train, X_test, z_train, z_test = train_test_split(X, z, test_size=0.2)
    c, v = np.shape(X_train.T @ X_train)
    I = np.eye(c, v)
    nlambdas = 5
    Ridge_MSEPredict = np.zeros(nlambdas)
    Ridge_MSETrain = np.zeros(nlambdas)
    Ridge_R2Predict = np.zeros(nlambdas)
    Ridge_R2train = np.zeros(nlambdas)
    lambdas = [0.001, 0.01, 0.1, 1, 10]

    for i in range(nlambdas):
        lmb = lambdas[i]
        Ridge_beta = np.linalg.pinv(X_train.T @ X_train + lmb * I) @ X_train.T @ z_train
        Ridge_ztilde = X_train @ Ridge_beta
        Ridge_zpredict = X_test @ Ridge_beta
        Ridge_MSETrain[i] = MSE(z_train, Ridge_ztilde)
        Ridge_MSEPredict[i] = MSE(z_test, Ridge_zpredict)
        Ridge_R2train[i] = R2(z_train, Ridge_ztilde)
        Ridge_R2Predict[i] = R2(z_test, Ridge_zpredict)

    return Ridge_MSETrain, Ridge_MSEPredict, Ridge_R2train, Ridge_R2Predict


def Ridge_analysis(N, n, data=None):
    Ridge_MSETrain_list = []
    Ridge_MSEPredict_list = []
    Ridge_R2Train_list = []
    Ridge_R2Predict_list = []

    for n in range(n):
        if data:
            MSETrain, MSEPredict, R2Train, R2Predict = Ridge(N, n, 1)
        else:
            MSETrain, MSEPredict, R2Train, R2Predict = Ridge(N, n)
        Ridge_MSETrain_list.append(MSETrain)
        Ridge_MSEPredict_list.append(MSEPredict)
        Ridge_R2Train_list.append(R2Train)
        Ridge_R2Predict_list.append(R2Predict)

    fig, axs = plt.subplots(1, 2, figsize=(10, 4))

    if data:
        fig.suptitle('Ridge Performance on Topographic Data', fontsize=16)
    else:
        fig.suptitle('Ridge Performance on Synthetic Data', fontsize=16)

    lambdas = np.asarray([0.001, 0.01, 0.1, 1, 10])

    for i in range(len(lambdas)):
        axs[0].plot(np.arange(1, n + 2), np.asarray(Ridge_MSETrain_list)[:, i], label=f'MSE_Lam={lambdas[i]}')
        axs[0].plot(np.arange(1, n + 2), np.asarray(Ridge_R2Train_list)[:, i], label=f'R2_Lam={lambdas[i]}')
    axs[0].set_xlabel('nth order polynomial')
    axs[0].set_title('Training Data')
    axs[0].legend(loc='right')

    for i in range(len(lambdas)):
        axs[1].plot(np.arange(1, n + 2), np.asarray(Ridge_MSEPredict_list)[:, i], label=f'MSE_Lam={lambdas[i]}')
        axs[1].plot(np.arange(1, n + 2), np.asarray(Ridge_R2Predict_list)[:, i], label=f'R2_Lam={lambdas[i]}')
    axs[1].set_xlabel('nth order polynomial')
    axs[1].set_title('Test Data')
    axs[1].legend(loc='right')

    plt.tight_layout()
    plt.show()


def Lasso(N, n, data=None):
    if data:
        terrain = read_data()
        z_data = terrain[:N, :N]

        # scaling the z-data
        scaler = MinMaxScaler()
        scaler.fit(z_data)
        z_data = scaler.transform(z_data)

        x = np.linspace(0, 1, np.shape(z_data)[0])
        y = np.linspace(0, 1, np.shape(z_data)[1])
        x_mesh, y_mesh = np.meshgrid(x, y)
        X_data = create_X(x_mesh, y_mesh, n)
        z_data = z_data.flatten()

        X_train, X_test, z_train, z_test = train_test_split(X_data, z_data, test_size=0.2)

    else:
        x = np.sort(np.random.uniform(0, 1, N))
        y = np.sort(np.random.uniform(0, 1, N))

        X = create_X(x, y, n)

        z = FrankeFunction(x, y)

        X_train, X_test, z_train, z_test = train_test_split(X, z, test_size=0.2)

    lambdas = [0.001, 0.01, 0.1, 1, 10]
    nlambdas = len(lambdas)
    Lasso_MSEPredict = np.zeros(nlambdas)
    Lasso_MSEtrain = np.zeros(nlambdas)
    Lasso_R2Predict = np.zeros(nlambdas)
    Lasso_R2train = np.zeros(nlambdas)

    for i in range(nlambdas):
        lmb = lambdas[i]
        RegLasso = linear_model.Lasso(lmb)
        RegLasso.fit(X_train, z_train)
        ytildeLasso = RegLasso.predict(X_train)
        ypredictLasso = RegLasso.predict(X_test)
        Lasso_MSEtrain[i] = MSE(z_train, ytildeLasso)
        Lasso_MSEPredict[i] = MSE(z_test, ypredictLasso)
        Lasso_R2train[i] = R2(z_train, ytildeLasso)
        Lasso_R2Predict[i] = R2(z_test, ypredictLasso)

    return Lasso_MSEtrain, Lasso_MSEPredict, Lasso_R2train, Lasso_R2Predict


def Lasso_analysis(N, n, data=None):
    Lasso_MSETrain_list = []
    Lasso_MSEPredict_list = []
    Lasso_R2Train_list = []
    Lasso_R2Predict_list = []

    for deg in range(n):
        if data:
            MSETrain, MSEPredict, R2Train, R2Predict = Lasso(N, deg, data=1)

        else:
            MSETrain, MSEPredict, R2Train, R2Predict = Lasso(N, deg)

        Lasso_MSETrain_list.append(MSETrain)
        Lasso_MSEPredict_list.append(MSEPredict)
        Lasso_R2Train_list.append(R2Train)
        Lasso_R2Predict_list.append(R2Predict)


    fig, axs = plt.subplots(1, 2, figsize=(10, 4))

    if data:
        fig.suptitle('Lasso Performance on Topographic Data', fontsize=16)
    else:
        fig.suptitle('Lasso Performance on Synthetic Data', fontsize=16)

    lambdas = np.asarray([0.001, 0.01, 0.1, 1, 10])

    for i in range(len(lambdas)):
        axs[0].plot(np.arange(1, n + 1), np.asarray(Lasso_MSETrain_list)[:, i], label=f'MSE_Lam={lambdas[i]}')
        axs[0].plot(np.arange(1, n + 1), np.asarray(Lasso_R2Train_list)[:, i], label=f'R2_Lam={lambdas[i]}')
    axs[0].set_xlabel('nth order polynomial')
    axs[0].set_title('Training Data')
    axs[0].legend(loc='right')

    for i in range(len(lambdas)):
        axs[1].plot(np.arange(1, n + 1), np.asarray(Lasso_MSEPredict_list)[:, i], label=f'MSE_Lam={lambdas[i]}')
        axs[1].plot(np.arange(1, n + 1), np.asarray(Lasso_R2Predict_list)[:, i], label=f'R2_Lam={lambdas[i]}')
    axs[1].set_xlabel('nth order polynomial')
    axs[1].set_title('Test Data')
    axs[1].legend(loc='right')

    plt.tight_layout()
    plt.show()


def Bias_variance_figure(N, n, data=None):
   error = np.zeros(n)
   bias = np.zeros(n)
   variance = np.zeros(n)

   for degree in range(n):
       if data:
           z_pred, z_test = OLS(N, degree, bootstrap_iter=N, data=1)
       else:
           z_pred, z_test = OLS(N, degree, bootstrap_iter=N)

       error[degree] = np.mean(((z_test.reshape((N//5, 1)) - z_pred)**2).mean(axis=1, keepdims=True))  # np.mean((z_test - z_pred.mean(axis=1)) ** 2)
       bias[degree] = np.mean((z_test - np.mean(z_pred, axis=1, keepdims=True)) ** 2)
       variance[degree] = np.mean(np.var(z_pred, axis=1, keepdims=True))

   polydegree = np.arange(1, n + 1)
   plt.plot(polydegree, error, label='Error')
   plt.plot(polydegree, bias, label='bias')
   plt.plot(polydegree, variance, label='Variance')
   plt.ylabel('n-order polynomial')
   if data:
       plt.title('Bias-Variance Tradeoff for Topographic Data')
   else:
       plt.title('Bias-Variance Tradeoff for Synthetic Data')

   plt.legend()
   plt.show()

   return error


def k_folds_cross_val(X, z, k):
   N = len(X[:, 0])
   # shuffling data and z
   rand_idx = np.random.randint(0, N, N)
   data = X[rand_idx]
   z = z[rand_idx]

   # finding size of subgroups
   subset_size = N // k
   new_datasets = []
   new_z = []

   # splitting into subgroups
   for i in range(k):
       new_datasets.append(data[(i * subset_size):(i + 1) * subset_size][:])
       new_z.append(z[(i * subset_size):(i + 1) * subset_size])

   MSE_list = []
   for i in range(k):
       X_train = np.asarray(list(new_datasets[:i][:]) + list(new_datasets[i + 1:][:]))
       # X_train = np.reshape(X_train, -1)
       X_train = X_train.reshape(-1, X_train.shape[-1])
       X_test = new_datasets[i]
       z_train = np.asarray(list(new_z[:i]) + list(new_z[i + 1:]))
       z_train = np.reshape(z_train, -1)
       z_test = new_z[i]

       OLS_beta = np.linalg.pinv(X_train.T @ X_train) @ X_train.T @ z_train
       OLS_zpredict = X_test @ OLS_beta

       MSE_list.append(MSE(z_test, OLS_zpredict))

   # print(MSE_list)
   Final_MSE = np.mean(MSE_list)
   # print(Final_MSE)

   return Final_MSE


def k_folds_analysis(N, n, mse=None, skl=None, data=None):

   if data:
       terrain = read_data()
       z_data = terrain[:N, :N]

       # scaling the z-data
       scaler = MinMaxScaler()
       scaler.fit(z_data)
       z_data = scaler.transform(z_data)

       # both x and y are innately scaled
       x = np.sort(np.linspace(0, 1, np.shape(z_data)[0]))
       y = np.sort(np.linspace(0, 1, np.shape(z_data)[1]))
       x_mesh, y_mesh = np.meshgrid(x, y)
       X = create_X(x_mesh, y_mesh, n)

       z = z_data.reshape(-1)
   else:
       x = np.sort(np.random.uniform(0, 1, N))
       y = np.sort(np.random.uniform(0, 1, N))

       X = create_X(x, y, n)
       z = FrankeFunction(x, y)

   # finding the different MSE's of k using own code and scikit's function
   OLS = LinearRegression()
   SK_MSE_vals = []
   k_fold_MSE_list = []
   k_values = np.arange(5, 11)
   for k in k_values:
       k_fold_MSE_list.append(k_folds_cross_val(X, z, k))
       if skl:
           kf = KFold(n_splits=k)
           SK_MSE_vals.append(np.mean(cross_val_score(OLS, X=X, y=z, cv=kf)))

   plt.plot(k_values, k_fold_MSE_list, label='K-fold MSE')
   plt.xlabel('k-folds')
   if data:
       plt.title('K-folds Cross-Val MSE for Topographic Data')
   else:
       plt.title('K-folds Cross-Val MSE for Synthetic Data')

   if skl:
       plt.plot(k_values, SK_MSE_vals, label='cross_val_score')

   if mse.any():
       plt.plot(k_values, mse[4:], label='Bootstrapping MSE')

   plt.legend()
   plt.show()


def read_data(plot=None):

   terrain2 = imread('SRTM_data_Norway_2.tif')

   if plot == '2D':
       plt.figure()
       plt.title('Terrain over Norway 1')
       plt.imshow(terrain2, cmap='gray')
       plt.xlabel('X')
       plt.ylabel('Y')
       plt.show()

   elif plot == '3D':
       x = np.linspace(0, 1, terrain2.shape[1])
       y = np.linspace(0, 1, terrain2.shape[0])
       X, Y = np.meshgrid(x, y)
       Z = terrain2

       fig = plt.figure()
       ax = fig.add_subplot(111, projection='3d')
       ax.plot_surface(X, Y, Z, cmap='viridis')
       ax.set_xlabel('X Label')
       ax.set_ylabel('Y Label')
       ax.set_zlabel('Z Label')
       ax.set_title('3D Surface Plot')
       plt.show()

   return np.asarray(terrain2)



if __name__ == "__main__":
   np.random.seed(2022)

   #read_data(plot='3D')


   #OLS_analysis(N=500, n=15, data=1)
   #Ridge_analysis(N=500, n=15, data=1)
   #Lasso_analysis(N=500, n=15, data=1)
   #mse = Bias_variance_figure(N=500, n=10)
   #k_folds_analysis(N=500, n=10, mse=mse)
   #read_data(plot=1)