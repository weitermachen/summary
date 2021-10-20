# # _*_ coding:utf-8 _*_
#
# import sys
# from numpy import *
# from svm import *
# from os import listdir
# import pandas as pd
#
# class PlattSMO:
#     def __init__(self,dataMat,classlabels,C,toler,maxIter,**kernelargs):
#         self.x = array(dataMat)
#         self.label = array(classlabels).transpose()
#         self.C = C
#         self.toler = toler
#         self.maxIter = maxIter
#         self.m = shape(dataMat)[0]
#         self.n = shape(dataMat)[1]
#         self.alpha = array(zeros(self.m),dtype='float64')
#         self.b = 0.0
#         self.eCache = array(zeros((self.m,2)))
#         self.K = zeros((self.m,self.m),dtype='float64')
#         self.kwargs = kernelargs
#         self.SV = ()
#         self.SVIndex = None
#         for i in range(self.m):
#             for j in range(self.m):
#                 self.K[i,j] = self.kernelTrans(self.x[i,:],self.x[j,:])
#     def calcEK(self,k):
#         fxk = dot(self.alpha*self.label,self.K[:,k])+self.b
#         Ek = fxk - float(self.label[k])
#         return Ek
#     def updateEK(self,k):
#         Ek = self.calcEK(k)
#
#         self.eCache[k] = [1 ,Ek]
#     def selectJ(self,i,Ei):
#         maxE = 0.0
#         selectJ = 0
#         Ej = 0.0
#         validECacheList = nonzero(self.eCache[:,0])[0]
#         if len(validECacheList) > 1:
#             for k in validECacheList:
#                 if k == i:continue
#                 Ek = self.calcEK(k)
#                 deltaE = abs(Ei-Ek)
#                 if deltaE > maxE:
#                     selectJ = k
#                     maxE = deltaE
#                     Ej = Ek
#             return selectJ,Ej
#         else:
#             selectJ = selectJrand(i,self.m)
#             Ej = self.calcEK(selectJ)
#             return selectJ,Ej
#
#     def innerL(self,i):
#         Ei = self.calcEK(i)
#         if (self.label[i] * Ei < -self.toler and self.alpha[i] < self.C) or \
#                 (self.label[i] * Ei > self.toler and self.alpha[i] > 0):
#             self.updateEK(i)
#             j,Ej = self.selectJ(i,Ei)
#             alphaIOld = self.alpha[i].copy()
#             alphaJOld = self.alpha[j].copy()
#             if self.label[i] != self.label[j]:
#                 L = max(0,self.alpha[j]-self.alpha[i])
#                 H = min(self.C,self.C + self.alpha[j]-self.alpha[i])
#             else:
#                 L = max(0,self.alpha[j]+self.alpha[i] - self.C)
#                 H = min(self.C,self.alpha[i]+self.alpha[j])
#             if L == H:
#                 return 0
#             eta = 2*self.K[i,j] - self.K[i,i] - self.K[j,j]
#             if eta >= 0:
#                 return 0
#             self.alpha[j] -= self.label[j]*(Ei-Ej)/eta
#             self.alpha[j] = clipAlpha(self.alpha[j],H,L)
#             self.updateEK(j)
#             if abs(alphaJOld-self.alpha[j]) < 0.00001:
#                 return 0
#             self.alpha[i] +=  self.label[i]*self.label[j]*(alphaJOld-self.alpha[j])
#             self.updateEK(i)
#             b1 = self.b - Ei - self.label[i] * self.K[i, i] * (self.alpha[i] - alphaIOld) - \
#                  self.label[j] * self.K[i, j] * (self.alpha[j] - alphaJOld)
#             b2 = self.b - Ej - self.label[i] * self.K[i, j] * (self.alpha[i] - alphaIOld) - \
#                  self.label[j] * self.K[j, j] * (self.alpha[j] - alphaJOld)
#             if 0<self.alpha[i] and self.alpha[i] < self.C:
#                 self.b = b1
#             elif 0 < self.alpha[j] and self.alpha[j] < self.C:
#                 self.b = b2
#             else:
#                 self.b = (b1 + b2) /2.0
#             return 1
#         else:
#             return 0
#
#     def smoP(self):
#         iter = 0
#         entrySet = True
#         alphaPairChanged = 0
#         while iter < self.maxIter and ((alphaPairChanged > 0) or (entrySet)):
#             alphaPairChanged = 0
#             if entrySet:
#                 for i in range(self.m):
#                     alphaPairChanged+=self.innerL(i)
#                 iter += 1
#             else:
#                 nonBounds = nonzero((self.alpha > 0)*(self.alpha < self.C))[0]
#                 for i in nonBounds:
#                     alphaPairChanged+=self.innerL(i)
#                 iter+=1
#             if entrySet:
#                 entrySet = False
#             elif alphaPairChanged == 0:
#                 entrySet = True
#         self.SVIndex = nonzero(self.alpha)[0]
#         self.SV = self.x[self.SVIndex]
#         self.SVAlpha = self.alpha[self.SVIndex]
#         self.SVLabel = self.label[self.SVIndex]
#         self.x = None
#         self.K = None
#         self.label = None
#         self.alpha = None
#         self.eCache = None
#     def K(self,i,j):
#       return self.x[i,:]*self.x[j,:].T
#
#     def kernelTrans(self,x,z):
#         if array(x).ndim != 1 or array(x).ndim != 1:
#             raise Exception("input vector is not 1 dim")
#         if self.kwargs['name'] == 'linear':
#             return sum(x*z)
#         elif self.kwargs['name'] == 'rbf':
#             theta = self.kwargs['theta']
#             return exp(sum((x-z)*(x-z))/(-1*theta**2))
#
#     def calcw(self):
#         for i in range(self.m):
#             self.w += dot(self.alpha[i]*self.label[i],self.x[i,:])
#
#     def predict(self,testData):
#         test = array(testData)
#         #return (test * self.w + self.b).getA()
#         result = []
#         m = shape(test)[0]
#         for i in range(m):
#             tmp = self.b
#             for j in range(len(self.SVIndex)):
#                 tmp += self.SVAlpha[j] * self.SVLabel[j] * self.kernelTrans(self.SV[j],test[i,:])
#             while tmp == 0:
#                 tmp = random.uniform(-1,1)
#             if tmp > 0:
#                 tmp = 1
#             else:
#                 tmp = -1
#             result.append(tmp)
#         return result
# def plotBestfit(data,label,w,b):
#     import matplotlib.pyplot as plt
#     n = shape(data)[0]
#     fig = plt.figure()
#     ax = fig.add_subplot(111)
#     x1 = []
#     x2 = []
#     y1 = []
#     y2 = []
#     for i in range(n):
#         if int(label[i]) == 1:
#             x1.append(data[i][0])
#             y1.append(data[i][1])
#         else:
#             x2.append(data[i][0])
#             y2.append(data[i][1])
#     ax.scatter(x1,y1,s=10,c='red',marker='s')
#     ax.scatter(x2,y2, s=10, c='green', marker='s')
#     x = arange(-2,10,0.1)
#     y = ((-b-w[0]*x)/w[1])
#     plt.plot(x,y)
#     plt.xlabel('X')
#     plt.ylabel('y')
#     plt.show()
#
# '''
# def loadImage(dir,maps = None):
#     dirList = listdir(dir)
#     data = []
#     label = []
#     for file in dirList:
#         label.append(file.split('_')[0])
#         lines = open(dir +'/'+file).readlines()
#         row = len(lines)
#         col = len(lines[0].strip())
#         line = []
#         for i in range(row):
#             for j in range(col):
#                 line.append(float(lines[i][j]))
#         data.append(line)
#         if maps != None:
#             label[-1] = float(maps[label[-1]])
#         else:
#             label[-1] = float(label[-1])
#     return array(data),array(label)
# '''
#
# def loadDataSet(filename): #读取需要处理的数据
#     data = []
#     label = []
#     train_data = pd.read_excel(filename,skiprows = 1,usecols = 'B:SK')
#     train_label = pd.read_excel(filename,skiprows = 1,usecols = 'SL')
#     data.append(train_data.values)
#     label.append(train_label.values)
#     return data, label  # 返回数据特征和数据类别
#
# def main():
#     '''
#     data,label = loadDataSet('testSetRBF.txt')
#     smo = PlattSMO(data,label,200,0.0001,10000,name = 'rbf',theta = 1.3)
#     smo.smoP()
#     smo.calcw()
#     print smo.predict(data)
#     '''
#     maps = {'1':1.0,'9':-1.0}
#     # data,label = loadImage("digits/trainingDigits",maps)
#     data,label = loadDataSet(r'C:\Users\sugar\Desktop\data\有用数据\Book1去除0_train.xlsx')
#     smo = PlattSMO(data, label, 200, 0.0001, 10000, name='rbf', theta=20)
#     smo.smoP()
#     print(len(smo.SVIndex))
#     # test,testLabel = loadImage("digits/testDigits",maps)
#     test,testLabel = loadDataSet(r'C:\Users\sugar\Desktop\data\有用数据\Book1去除0_test.xlsx')
#     testResult = smo.predict(test)
#     m = shape(test)[0]
#     count  = 0.0
#     for i in range(m):
#         if testLabel[i] != testResult[i]:
#             count += 1
#     print("classfied error rate is:",count / m)
#     #smo.kernelTrans(data,smo.SV[0])
#
# if __name__ == "__main__":
#     sys.exit(main())



# _*_ coding:utf-8 _*_
# _*_ coding:utf-8 _*
from numpy import *
import pandas as pd
import matplotlib.pyplot as plt
import  numpy  as np

# def loadDataSet(filename):  # 读取数据
#     dataMat = []
#     labelMat = []
#     fr = open(filename)
#     for line in fr.readlines():
#         lineArr = line.strip().split()
#         dataMat.append([float(lineArr[0]), float(lineArr[1])])
#         labelMat.append(float(lineArr[2]))
#     return dataMat, labelMat  # 返回数据特征和数据类别

def loadDataSet(filename): #读取需要处理的数据
    # dataMat = []
    # labelMat = []
    train_data = pd.read_excel(filename,skiprows = 1,usecols = 'B:BI',sheetname = 'Sheet2')
    train_label = pd.read_excel(filename,skiprows = 1,usecols = 'B',sheetname = 'Sheet3')
    # dataMat.append(train_data.values)
    # labelMat.append(train_label.values)
    dataMat = train_data.values
    labelMat = train_label.values.flatten()
    return dataMat, labelMat  # 返回数据特征和数据类别


def selectJrand(i, m):  # 在0-m中随机选择一个不是i的整数
    j = i
    while (j == i):
        j = int(random.uniform(0, m))
    return j


def clipAlpha(aj, H, L):  # 保证a在L和H范围内（L <= a <= H）
    if aj > H:
        aj = H
    if L > aj:
        aj = L
    return aj


def kernelTrans(X, A, kTup):  # 核函数，输入参数,X:支持向量的特征树；A：某一行特征数据；kTup：('lin',k1)核函数的类型和参数
    m, n = shape(X)
    K = mat(zeros((m, 1)))
    if kTup[0] == 'lin':  # 线性函数
        K = X * A.T
    elif kTup[0] == 'rbf':  # 径向基函数(radial bias function)
        for j in range(m):
            deltaRow = X[j, :] - A
            K[j] = deltaRow * deltaRow.T
        K = exp(K / (-1 * kTup[1] ** 2))  # 返回生成的结果
    else:
        raise NameError('Houston We Have a Problem -- That Kernel is not recognized')
    return K


# 定义类，方便存储数据
class optStruct:
    def __init__(self, dataMatIn, classLabels, C, toler, kTup):  # 存储各类参数
        self.X = dataMatIn  # 数据特征
        self.labelMat = classLabels  # 数据类别
        self.C = C  # 软间隔参数C，参数越大，非线性拟合能力越强
        self.tol = toler  # 停止阀值
        self.m = shape(dataMatIn)[0]  # 数据行数
        self.alphas = mat(zeros((self.m, 1)))
        self.b = 0  # 初始设为0
        self.eCache = mat(zeros((self.m, 2)))  # 缓存
        self.K = mat(zeros((self.m, self.m)))  # 核函数的计算结果
        for i in range(self.m):
            self.K[:, i] = kernelTrans(self.X, self.X[i, :], kTup)


def calcEk(oS, k):  # 计算Ek（参考《统计学习方法》p127公式7.105）
    fXk0 = multiply(oS.alphas, oS.labelMat).T * oS.K[:, k] + oS.b
    fXk = float(multiply(oS.alphas, oS.labelMat).T * oS.K[:, k] + oS.b)
    # fXk = fXk0.astype(np.float)
    Ek = fXk - float(oS.labelMat[k])
    # print(oS.labelMat[k-1].astype(np.float))
    # Ek = fXk - oS.labelMat[k].astype(np.float)
    print(Ek)
    return Ek


# 随机选取aj，并返回其E值
def selectJ(i, oS, Ei):
    maxK = -1
    maxDeltaE = 0
    Ej = 0
    oS.eCache[i] = [1, Ei]
    validEcacheList = nonzero(oS.eCache[:, 0].A)[0]  # 返回矩阵中的非零位置的行数
    if (len(validEcacheList)) > 1:
        for k in validEcacheList:
            if k == i:
                continue
            Ek = calcEk(oS, k)
            deltaE = abs(Ei - Ek)
            if (deltaE > maxDeltaE):  # 返回步长最大的aj
                maxK = k
                maxDeltaE = deltaE
                Ej = Ek
        return maxK, Ej
    else:
        j = selectJrand(i, oS.m)
        Ej = calcEk(oS, j)
    return j, Ej


def updateEk(oS, k):  # 更新os数据
    Ek = calcEk(oS, k)
    oS.eCache[k] = [1, Ek]


# 首先检验ai是否满足KKT条件，如果不满足，随机选择aj进行优化，更新ai,aj,b值
def innerL(i, oS):  # 输入参数i和所有参数数据
    Ei = calcEk(oS, i)  # 计算E值（预测值与真实值之间的差距）
    if ((oS.labelMat[i] * Ei < -oS.tol) and (oS.alphas[i] < oS.C)) or (
            (oS.labelMat[i] * Ei > oS.tol) and (oS.alphas[i] > 0)):

    # if ((oS.labelMat[i] * Ei < -oS.tol).all() and (oS.alphas[i] < oS.C).all()) or (
    #             (oS.labelMat[i] * Ei > oS.tol).all() and (oS.alphas[i] > 0).all()):
        # 检验这行数据是否符合KKT条件 参考《统计学习方法》p128公式7.111-113
        j, Ej = selectJ(i, oS, Ei)  # 随机选取aj，并返回其E值
        alphaIold = oS.alphas[i].copy()
        alphaJold = oS.alphas[j].copy()
        if (oS.labelMat[i] != oS.labelMat[j]):  # 以下代码的公式参考《统计学习方法》p126
            L = max(0, oS.alphas[j] - oS.alphas[i])
            H = min(oS.C, oS.C + oS.alphas[j] - oS.alphas[i])
        else:
            L = max(0, oS.alphas[j] + oS.alphas[i] - oS.C)
            H = min(oS.C, oS.alphas[j] + oS.alphas[i])
        if L == H:
            print("L==H")
            return 0
        eta = 2.0 * oS.K[i, j] - oS.K[i, i] - oS.K[j, j]  # 参考《统计学习方法》p127公式7.107
        if eta >= 0:
            print("eta>=0")
            return 0
        oS.alphas[j] -= oS.labelMat[j] * (Ei - Ej) / eta  # 参考《统计学习方法》p127公式7.106
        oS.alphas[j] = clipAlpha(oS.alphas[j], H, L)  # 参考《统计学习方法》p127公式7.108
        updateEk(oS, j)
        if (abs(oS.alphas[j] - alphaJold) < oS.tol):  # alpha变化大小阀值（自己设定）
            print("j not moving enough")
            return 0
        oS.alphas[i] += oS.labelMat[j] * oS.labelMat[i] * (alphaJold - oS.alphas[j])  # 参考《统计学习方法》p127公式7.109
        updateEk(oS, i)  # 更新数据
        # 以下求解b的过程，参考《统计学习方法》p129公式7.114-7.116
        b1 = oS.b - Ei - oS.labelMat[i] * (oS.alphas[i] - alphaIold) * oS.K[i, i] - oS.labelMat[j] * (
                    oS.alphas[j] - alphaJold) * oS.K[i, j]
        b2 = oS.b - Ej - oS.labelMat[i] * (oS.alphas[i] - alphaIold) * oS.K[i, j] - oS.labelMat[j] * (
                    oS.alphas[j] - alphaJold) * oS.K[j, j]
        if (0 < oS.alphas[i] < oS.C):
            oS.b = b1
        elif (0 < oS.alphas[j] < oS.C):
            oS.b = b2
        else:
            oS.b = (b1 + b2) / 2.0
        return 1
    else:
        return 0


# SMO函数，用于快速求解出alpha
def smoP(dataMatIn, classLabels, C, toler, maxIter, kTup=('lin', 0)):
    # 输入参数：数据特征，数据类别，参数C，阀值toler，最大迭代次数，核函数（默认线性核）
    oS = optStruct(mat(dataMatIn), mat(classLabels).transpose(), C, toler, kTup)
    iter = 0
    entireSet = True
    alphaPairsChanged = 0
    while (iter < maxIter) and ((alphaPairsChanged > 0) or (entireSet)):
        alphaPairsChanged = 0
        if entireSet:
            for i in range(oS.m):  # 遍历所有数据
                alphaPairsChanged += innerL(i, oS)
                print("fullSet, iter: %d i:%d, pairs changed %d" % (iter, i, alphaPairsChanged))
                #  显示第多少次迭代，那行特征数据使alpha发生了改变，这次改变了多少次alpha
            iter += 1
        else:
            nonBoundIs = nonzero((oS.alphas.A > 0) * (oS.alphas.A < C))[0]
            for i in nonBoundIs:  # 遍历非边界的数据
                alphaPairsChanged += innerL(i, oS)
                print("non-bound, iter: %d i:%d, pairs changed %d" % (iter, i, alphaPairsChanged))
            iter += 1
        if entireSet:
            entireSet = False
        elif (alphaPairsChanged == 0):
            entireSet = True
        print("iteration number: %d" % iter)
    return oS.b, oS.alphas


def testRbf(data_train, data_test):
    dataArr, labelArr = loadDataSet(data_train)  # 读取训练数据
    b, alphas = smoP(dataArr, labelArr, 200, 0.0001, 10000, ('rbf', 1.3))  # 通过SMO算法得到b和alpha

    # w = calcWs(dataArr, labelArr, alphas)
    # showClassifer(dataArr, labelArr, alphas, w, b)

    datMat = mat(dataArr)
    labelMat = mat(labelArr).transpose()
    svInd = nonzero(alphas)[0]  # 选取不为0数据的行数（也就是支持向量）
    sVs = datMat[svInd]  # 支持向量的特征数据
    labelSV = labelMat[svInd]  # 支持向量的类别（1或-1）
    print("there are %d Support Vectors" % shape(sVs)[0])  # 打印出共有多少的支持向量
    m, n = shape(datMat)  # 训练数据的行列数
    errorCount = 0
    for i in range(m):
        kernelEval = kernelTrans(sVs, datMat[i, :], ('rbf', 1.3))  # 将支持向量转化为核函数
        predict = kernelEval.T * multiply(labelSV, alphas[svInd]) + b
        # 这一行的预测结果（代码来源于《统计学习方法》p133里面最后用于预测的公式）注意最后确定的分离平面只有那些支持向量决定。
        # if sign(predict) == sign(labelArr[i]):  # sign函数 -1 if x < 0, 0 if x==0, 1 if x > 0
        #     errorCount += 1
        if sign(predict) != sign(labelArr[i]):  # sign函数 -1 if x < 0, 0 if x==0, 1 if x > 0
            errorCount += 1
    print("the training error rate is: %f" % (float(errorCount) / m))  # 打印出错误率
    dataArr_test, labelArr_test = loadDataSet(data_test)  # 读取测试数据
    errorCount_test = 0
    datMat_test = mat(dataArr_test)
    labelMat = mat(labelArr_test).transpose()
    m, n = shape(datMat_test)
    for i in range(m):  # 在测试数据上检验错误率
        kernelEval = kernelTrans(sVs, datMat_test[i, :], ('rbf', 1.3))
        predict = kernelEval.T * multiply(labelSV, alphas[svInd]) + b
        if sign(predict) != sign(labelArr_test[i]):
            errorCount_test += 1
    print("the test error rate is: %f" % (float(errorCount_test) / m))



# def calcWs(dataMat, labelMat, alphas):
#     alphas, dataMat, labelMat = np.array(alphas), np.array(dataMat), np.array(labelMat)
#     w = np.dot((np.tile(labelMat.reshape(1, -1).T, (1, 2)) * dataMat).T, alphas)
#     return w.tolist()
#
#
# def showClassifer(dataMat,labelMat,alphas, w, b):
#     data_plus = []
#     data_minus = []
#     for i in range(len(dataMat)):
#         if labelMat[i] > 0:
#             data_plus.append(dataMat[i])
#         else:
#             data_minus.append(dataMat[i])
#     data_plus_np = array(data_plus)
#     data_minus_np = array(data_minus)
#     plt.scatter(transpose(data_plus_np)[0], transpose(data_plus_np)[1], s=30, alpha=0.7)
#     plt.scatter(transpose(data_minus_np)[0], transpose(data_minus_np)[1], s=30, alpha=0.7)
#     x1 = max(dataMat)[0]
#     x2 = min(dataMat)[0]
#     a1, a2 = w
#     b = float(b)
#     a1 = float(a1[0])
#     a2 = float(a2[0])
#     y1, y2 = (-b- a1*x1)/a2, (-b - a1*x2)/a2
#     plt.plot([x1, x2], [y1, y2])
#     for i, alpha in enumerate(alphas):
#         if 0.6>abs(alpha) > 0:
#             x, y = dataMat[i]
#             plt.scatter([x], [y], s=150, c='none', alpha=0.7, linewidth=1.5, edgecolor='red')
#         if 50==abs(alpha) :
#             x, y = dataMat[i]
#             plt.scatter([x], [y], s=150, c='none', alpha=0.7, linewidth=1.5, edgecolor='yellow')
#     plt.show()




# '''
# 函数名称：cal_W
# 函数功能：根据alpha和y来计算W
# 输入参数：dS         dataStruct类的数据
# 返回参数：W          超平名的法向量W
# 作者：Leo Ma
# 时间：2019.05.20
# '''
#
#
# def cal_W(dS):
#     W = np.dot(dS.dataMat.T, np.multiply(dS.labelMat, dS.alphas))
#     return W
#
#
# '''
# 函数名称：showClassifer
# 函数功能：画出原始数据点、超平面，并标出支持向量
# 输入参数：dS         dataStruct类的数据
#           W          超平名的法向量W
# 返回参数：None
# 作者：机器学习实践SVM chapter 6
# 修改：Leo Ma
# 时间：2019.05.20
# '''
#
#
# def showClassifer(dS, w):
#     # 绘制样本点
#     dataMat = dS.dataMat.tolist()
#     data_plus = []  # 正样本
#     data_minus = []  # 负样本
#     for i in range(len(dataMat)):
#         if dS.labelMat[i, 0] > 0:
#             data_plus.append(dataMat[i])
#         else:
#             data_minus.append(dataMat[i])
#     data_plus_np = np.array(data_plus)  # 转换为numpy矩阵
#     data_minus_np = np.array(data_minus)  # 转换为numpy矩阵
#     plt.scatter(np.transpose(data_plus_np)[0], np.transpose(data_plus_np)[1], s=30, alpha=0.7, c='r')  # 正样本散点图
#     plt.scatter(np.transpose(data_minus_np)[0], np.transpose(data_minus_np)[1], s=30, alpha=0.7, c='g')  # 负样本散点图
#     # 绘制直线
#     x1 = max(dataMat)[0]
#     x2 = min(dataMat)[0]
#     a1, a2 = w
#     b = float(dS.b)
#     a1 = float(a1[0])
#     a2 = float(a2[0])
#     y1, y2 = (-b - a1 * x1) / a2, (-b - a1 * x2) / a2
#     plt.plot([x1, x2], [y1, y2])
#     # 找出支持向量点
#     for i, alpha in enumerate(dS.alphas):
#         if abs(alpha) > 0.000000001:
#             x, y = dataMat[i]
#             plt.scatter([x], [y], s=150, c='none', alpha=0.7, linewidth=1.5, edgecolor='red')
#     plt.xlabel("happy 520 day, 2018.06.13")
#     plt.savefig("svm.png")
#     plt.show()

# 主程序
def main():
    filename_traindata = r'C:\Users\sugar\Desktop\Book1去除0.xlsx'
    filename_testdata = r'C:\Users\sugar\Desktop\Book1去除0.xlsx'
    testRbf(filename_traindata, filename_testdata)


if __name__ == '__main__':
    main()


'''
#Implement svm algorithm only using basic python
#Author:Leo Ma
#For csmath2019 assignment5,ZheJiang University
#Date:2019.05.20
'''

# import numpy as np
# import random
# import matplotlib.pyplot as plt
# import pandas as pd
#
# '''
# 类名称：dataStruct
# 功能：用于存储一些需要保存或者初始化的数据
# 作者：Leo Ma
# 时间：2019.05.20
# '''
#
#
# class dataStruct:
#     def __init__(self, dataMatIn, labelMatIn, C, toler, eps):
#         self.dataMat = dataMatIn  # 样本数据
#         self.labelMat = labelMatIn  # 样本标签
#         self.C = C  # 参数C
#         self.toler = toler  # 容错率
#         self.eps = eps  # 乘子更新最小比率
#         self.m = np.shape(dataMatIn)[0]  # 样本数
#         self.alphas = np.mat(np.zeros((self.m, 1)))  # 拉格朗日乘子alphas，shape(m,1),初始化全为0
#         self.b = 0  # 参数b，初始化为0
#         self.eCache = np.mat(np.zeros((self.m, 2)))  # 误差缓存，
#
#
# '''
# 函数名称：loadData
# 函数功能：读取文本文件中的数据，以样本数据和标签的形式返回
# 输入参数：filename       文本文件名
# 返回参数：dataMat        样本数据
#          labelMat       样本标签
# 作者：Leo Ma
# 时间：2019.05.20
# '''
#
# def loadDataSet(filename): #读取需要处理的数据
#     # dataMat = []
#     # labelMat = []
#     train_data = pd.read_excel(filename,skiprows = 1,usecols = 'B:SK')
#     train_label = pd.read_excel(filename,skiprows = 1,usecols = 'SL')
#     # dataMat.append(train_data.values)
#     # labelMat.append(train_label.values)
#     dataMat = train_data.values
#     labelMat = train_label.values
#     return dataMat, labelMat  # 返回数据特征和数据类别
#
# # def loadData(filename):
# #     dataMat = []
# #     labelMat = []
# #     fr = open(filename)
# #     for line in fr.readlines():  # 逐行读取
# #         lineArr = line.strip().split('\t')  # 滤除行首行尾空格，以\t作为分隔符，对这行进行分解
# #         num = np.shape(lineArr)[0]
# #         dataMat.append(list(map(float, lineArr[0:num - 1])))  # 这一行的除最后一个被添加为数据
# #         labelMat.append(float(lineArr[num - 1]))  # 这一行的最后一个数据被添加为标签
# #     dataMat = np.mat(dataMat)
# #     labelMat = np.mat(labelMat).T
# #     return dataMat, labelMat


# '''
# 函数名称：takeStep
# 函数功能：给定alpha1和alpha2，执行alpha1和alpha2的更新,执行b的更新
# 输入参数：i1            alpha1的标号
#           i2            alpha2的标号
#           dataMat       样本数据
#           labelMat      样本标签
# 返回参数：如果i1==i2 or L==H or eta<=0 or alpha更新前后相差太小，返回0
#          正常执行，返回1
# 作者：Leo Ma
# 时间：2019.05.20
# '''
#
#
# def takeStep(i1, i2, dS):
#     # 如果选择了两个相同的乘子，不满足线性等式约束条件，因此不做更新
#     if (i1 == i2):
#         print("i1 == i2")
#         return 0
#     # 从数据结构中取得需要用到的数据
#     alpha1 = dS.alphas[i1, 0]
#     alpha2 = dS.alphas[i2, 0]
#     y1 = dS.labelMat[i1]
#     y2 = dS.labelMat[i2]
#
#     # 如果E1以前被计算过，就直接从数据结构的cache中读取它，这样节省计算量,#如果没有历史记录，就计算E1
#     if (dS.eCache[i1, 0] == 1):
#         E1 = dS.eCache[i1, 1]
#     else:
#         u1 = (np.multiply(dS.alphas, dS.labelMat)).T * np.dot(dS.dataMat, dS.dataMat[i1, :].T) + dS.b  # 计算SVM的输出值u1
#         E1 = float(u1 - y1)  # 误差E1
#         # dS.eCache[i1] = [1,E1] #存到cache中
#
#     # 如果E2以前被计算过，就直接从数据结构的cache中读取它，这样节省计算量,#如果没有历史记录，就计算E2
#     if (dS.eCache[i2, 0] == 1):
#         E2 = dS.eCache[i2, 1]
#     else:
#         u2 = (np.multiply(dS.alphas, dS.labelMat)).T * np.dot(dS.dataMat, dS.dataMat[i2, :].T) + dS.b  # 计算SVM的输出值u2
#         E2 = float(u2 - y2)  # 误差E2
#         # dS.eCache[i2] = [1,E2] #存到cache中
#
#     s = y1 * y2
#
#     # 计算alpha2的上界H和下界L
#     if (s == 1):  # 如果y1==y2
#         L = max(0, alpha1 + alpha2 - dS.C)
#         H = min(dS.C, alpha1 + alpha2)
#     elif (s == -1):  # 如果y1!=y2
#         L = max(0, alpha2 - alpha1)
#         H = min(dS.C, dS.C + alpha2 - alpha1)
#     if (L == H):
#         print("L==H")
#         return 0
#
#     # 计算学习率eta
#     k11 = np.dot(dS.dataMat[i1, ::], dS.dataMat[i1, :].T)
#     k12 = np.dot(dS.dataMat[i1, ::], dS.dataMat[i2, :].T)
#     k22 = np.dot(dS.dataMat[i2, ::], dS.dataMat[i2, :].T)
#     eta = k11 - 2 * k12 + k22
#
#     if (eta > 0):  # 正常情况下eta是大于0的，此时计算新的alpha2,新的alpha2标记为a2
#         a2 = alpha2 + y2 * (E1 - E2) / eta  # 这个公式的推导，曾经花费了我很多精力，现在写出来却是如此简洁，数学真是个好东西
#         # 对a2进行上下界裁剪
#         if (a2 < L):
#             a2 = L
#         elif (a2 > H):
#             a2 = H
#     else:  # 非正常情况下，也有可能出现eta《=0的情况
#         print("eta<=0")
#         return 0
#         '''
#         Lobj =
#         Hobj =
#         if(Lobj < Hobj-eps):
#             a2 = L
#         elif(Lobj > Hobj+eps):
#             a2 = H
#         else:
#             a2 = alpha2
#         '''
#
#     # 如果更新量太小，就不值浪费算力继续算a1和b，不值得对这三者进行更新
#     if (abs(a2 - alpha2) < dS.eps * (a2 + alpha2 + dS.eps)):
#         print("so small update on alpha2!")
#         return 0
#
#     # 计算新的alpha1，标记为a1
#     a1 = alpha1 + s * (alpha2 - a2)
#
#     # 计算b1和b2,并且更新b
#     b1 = -E1 + y1 * (alpha1 - a1) * np.dot(dS.dataMat[i1, :], dS.dataMat[i1, :].T) + y2 * (alpha2 - a2) * np.dot(
#         dS.dataMat[i1, :], dS.dataMat[i2, :].T) + dS.b
#     b2 = -E2 + y1 * (alpha1 - a1) * np.dot(dS.dataMat[i1, :], dS.dataMat[i2, :].T) + y2 * (alpha2 - a2) * np.dot(
#         dS.dataMat[i2, :], dS.dataMat[i2, :].T) + dS.b
#     if (a1 > 0 and a1 < dS.C):
#         dS.b = b1
#     elif (a2 > 0 and a2 < dS.C):
#         dS.b = b2
#     else:
#         dS.b = (b1 + b2) / 2
#
#     # 用a1和a2更新alpha1和alpha2
#     dS.alphas[i1] = a1
#     dS.alphas[i2] = a2
#
#     # 由于本次alpha1、alpha2和b的更新，需要重新计算Ecache，注意Ecache只存储那些非零的alpha对应的误差
#     validAlphasList = np.nonzero(dS.alphas.A)[0]  # 所有的非零的alpha标号列表
#     dS.eCache = np.mat(np.zeros((dS.m, 2)))  # 要把Ecache先清空
#     for k in validAlphasList:  # 遍历所有的非零alpha
#         uk = (np.multiply(dS.alphas, dS.labelMat).T).dot(np.dot(dS.dataMat, dS.dataMat[k, :].T)) + dS.b
#         yk = dS.labelMat[k, 0]
#         Ek = float(uk - yk)
#         dS.eCache[k] = [1, Ek]
#     print("updated")
#     return 1
#
#
# '''
# 函数名称：examineExample
# 函数功能：给定alpha2，如果alpha2不满足KKT条件，则再找一个alpha1,对这两个乘子进行一次takeStep
# 输入参数：i2            alpha的标号
#           dataMat       样本数据
#           labelMat      样本标签
# 返回参数：如果成功对一对乘子alpha1和alpha2执行了一次takeStep，返回1;否则，返回0
# 作者：Leo Ma
# 时间：2019.05.20
# '''
#
#
# def examineExample(i2, dS):
#     # 从数据结构中取得需要用到的数据
#     y2 = dS.labelMat[i2, 0]
#     alpha2 = dS.alphas[i2, 0]
#
#     # 如果E2以前被计算过，就直接从数据结构的cache中读取它，这样节省计算量,#如果没有历史记录，就计算E2
#     if (dS.eCache[i2, 0] == 1):
#         E2 = dS.eCache[i2, 1]
#     else:
#         u2 = (np.multiply(dS.alphas, dS.labelMat)).T * np.dot(dS.dataMat, dS.dataMat[i2, :].T) + dS.b  # 计算SVM的输出值u2
#         E2 = float(u2 - y2)  # 误差E2
#         # dS.eCache[i2] = [1,E2]
#
#     r2 = E2 * y2
#     # 如果当前的alpha2在一定容忍误差内不满足KKT条件，则需要对其进行更新
#     if ((r2 < -dS.toler and alpha2 < dS.C) or (r2 > dS.toler and alpha2 > 0)):
#         '''
#         #随机选择的方法确定另一个乘子alpha1，多执行几次可可以收敛到很好的结果，就是效率比较低
#         i1 = random.randint(0, dS.m-1)
#         if(takeStep(i1,i2,dS)):
#             return 1
#         '''
#         # 启发式的方法确定另一个乘子alpha1
#         nonZeroAlphasList = np.nonzero(dS.alphas.A)[0].tolist()  # 找到所有的非0的alpha
#         nonCAlphasList = np.nonzero((dS.alphas - dS.C).A)[0].tolist()  # 找到所有的非C的alpha
#         nonBoundAlphasList = list(set(nonZeroAlphasList) & set(nonCAlphasList))  # 所有非边界（既不=0,也不=C）的alpha
#
#         # 如果非边界的alpha数量至少两个，则在所有的非边界alpha上找到能够使\E1-E2\最大的那个E1,对这一对乘子进行更新
#         if (len(nonBoundAlphasList) > 1):
#             maxE = 0
#             maxEindex = 0
#             for k in nonBoundAlphasList:
#                 if (abs(dS.eCache[k, 1] - E2) > maxE):
#                     maxE = abs(dS.eCache[k, 1] - E2)
#                     maxEindex = k
#             i1 = maxEindex
#             if (takeStep(i1, i2, dS)):
#                 return 1
#
#             # 如果上面找到的那个i1没能使alpha和b得到有效更新，则从随机开始处遍历整个非边界alpha作为i1,逐个对每一对乘子尝试进行更新
#             randomStart = random.randint(0, len(nonBoundAlphasList) - 1)
#             for i1 in range(randomStart, len(nonBoundAlphasList)):
#                 if (i1 == i2): continue
#                 if (takeStep(i1, i2, dS)):
#                     return 1
#             for i1 in range(0, randomStart):
#                 if (i1 == i2): continue
#                 if (takeStep(i1, i2, dS)):
#                     return 1
#
#         # 如果上面的更新仍然没有return 1跳出去或者非边界alpha数量少于两个，这种情况只好从随机开始的位置开始遍历整个可能的i1,对每一对尝试更新
#         randomStart = random.randint(0, dS.m - 1)
#         for i1 in range(randomStart, dS.m):
#             if (i1 == i2): continue
#             if (takeStep(i1, i2, dS)):
#                 return 1
#         for i1 in range(0, randomStart):
#             if (i1 == i2): continue
#             if (takeStep(i1, i2, dS)):
#                 return 1
#         '''
#         i1 = random.randint(0,dS.m-1)
#         if(takeStep(i1,i2,dS)):
#             return 1
#         '''
#     # 如果实在还更新不了，就回去重新选择一个alpha2吧，当前的alpha2肯定是有毒
#     return 0
#
#
# '''
# 函数名称：SVM_with_SMO
# 函数功能：用SMO写的SVM的入口函数，里面采用了第一个启发式确定alpha2,即在全局遍历和非边界遍历之间来回repeat，直到不再有任何更新
# 输入参数：dS            dataStruct类的数据
# 返回参数：None
# 作者：Leo Ma
# 时间：2019.05.20
# '''
#
#
# def SVM_with_SMO(dS):
#     # 初始化控制变量，确保第一次要全局遍历
#     numChanged = 0
#     examineAll = 1
#
#     # 显然，如果全局遍历了一次，并且没有任何更新，此时examineAll和numChanged都会被置零，算法终止
#     while (numChanged > 0 or examineAll):
#         numChanged = 0
#         if (examineAll):
#             for i in range(dS.m):
#                 numChanged += examineExample(i, dS)
#         else:
#             for i in range(dS.m):
#                 if (dS.alphas[i] == 0 or dS.alphas[i] == dS.C): continue
#                 numChanged += examineExample(i, dS)
#         if (examineAll == 1):
#             examineAll = 0
#         elif (numChanged == 0):
#             examineAll = 1
#
#
# '''
# 函数名称：cal_W
# 函数功能：根据alpha和y来计算W
# 输入参数：dS         dataStruct类的数据
# 返回参数：W          超平名的法向量W
# 作者：Leo Ma
# 时间：2019.05.20
# '''
#
#
# def cal_W(dS):
#     W = np.dot(dS.dataMat.T, np.multiply(dS.labelMat, dS.alphas))
#     return W
#
#
# '''
# 函数名称：showClassifer
# 函数功能：画出原始数据点、超平面，并标出支持向量
# 输入参数：dS         dataStruct类的数据
#           W          超平名的法向量W
# 返回参数：None
# 作者：机器学习实践SVM chapter 6
# 修改：Leo Ma
# 时间：2019.05.20
# '''
#
#
# def showClassifer(dS, w):
#     # 绘制样本点
#     dataMat = dS.dataMat.tolist()
#     data_plus = []  # 正样本
#     data_minus = []  # 负样本
#     for i in range(len(dataMat)):
#         if dS.labelMat[i, 0] > 0:
#             data_plus.append(dataMat[i])
#         else:
#             data_minus.append(dataMat[i])
#     data_plus_np = np.array(data_plus)  # 转换为numpy矩阵
#     data_minus_np = np.array(data_minus)  # 转换为numpy矩阵
#     plt.scatter(np.transpose(data_plus_np)[0], np.transpose(data_plus_np)[1], s=30, alpha=0.7, c='r')  # 正样本散点图
#     plt.scatter(np.transpose(data_minus_np)[0], np.transpose(data_minus_np)[1], s=30, alpha=0.7, c='g')  # 负样本散点图
#     # 绘制直线
#     x1 = max(dataMat)[0]
#     x2 = min(dataMat)[0]
#     a1, a2 = w
#     b = float(dS.b)
#     a1 = float(a1[0])
#     a2 = float(a2[0])
#     y1, y2 = (-b - a1 * x1) / a2, (-b - a1 * x2) / a2
#     plt.plot([x1, x2], [y1, y2])
#     # 找出支持向量点
#     for i, alpha in enumerate(dS.alphas):
#         if abs(alpha) > 0.000000001:
#             x, y = dataMat[i]
#             plt.scatter([x], [y], s=150, c='none', alpha=0.7, linewidth=1.5, edgecolor='red')
#     plt.xlabel("happy 520 day, 2018.06.13")
#     plt.savefig("svm.png")
#     plt.show()
#
#
# if __name__ == '__main__':
#     dataMat, labelMat = loadData(r'C:\Users\sugar\Desktop\data\有用数据\Book1去除0_train.xlsx')
#     dS = dataStruct(dataMat, labelMat, 0.6, 0.001, 0.01)  # 初始化数据结构 dataMatIn, labelMatIn,C,toler,eps
#     for i in range(0, 1):  # 只需要执行一次，效果就非常不错
#         SVM_with_SMO(dS)
#     W = cal_W(dS)
#     showClassifer(dS, W.tolist())pip




