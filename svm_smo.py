import numpy as np
import random
import matplotlib.pyplot as plt


class osStruct:
    def __init__(self, dataMatIn, classlabels, C, toler):
        self.dataMatrix = dataMatIn
        self.labelMatrix = classlabels
        self.C = C
        self.toler = toler
        # shape[0]行，shape[1]列
        self.m = self.dataMatrix.shape[0]
        self.b = 0.0
        self.alphas = np.zeros((self.m, 1))
        self.eCache = np.zeros((self.m, 2))
        self.K = np.zeros((self.m, self.m))
        for i in range(self.m):
            self.K[i] = np.matmul(self.dataMatrix[i, :], self.dataMatrix.T)
            self.K[i].reshape(1,20)


def selectJRand(i, m):
    j = i
    while j == i:
        j = np.random.randint(0, m, 1)[0]
    return j


def clipAlpha(alpha, L, H):
    if alpha >= H:
        return H
    elif alpha <= L:
        return L
    else:
        return alpha


def calEi(obj, i):
    t=obj.labelMatrix.reshape(len(obj.labelMatrix),1)
    fxi = float(np.matmul((obj.alphas*t).T , obj.K[:, i])[0]) + obj.b
    Ek = fxi - obj.labelMatrix[i, 0]
    return Ek


def updateEi(obj, i):
    Ei = calEi(obj, i)
    obj.eCache[i] = [1, Ei]


def selectJIndex(obj, i, Ei):
    maxJ = -1
    maxdelta = -1
    Ek = -1
    obj.eCache[i] = [1, Ei]
    vaildEiList = np.nonzero(obj.eCache[:, 0])[0]
    if len(vaildEiList) > 1:
        for j in vaildEiList:
            if j == i:
                continue
            Ej = calEi(obj, j)
            delta = np.abs(Ei - Ej)
            if delta > maxdelta:
                maxdelta = delta
                maxJ = j
                Ek = Ej
    else:
        maxJ = selectJRand(i, obj.m)
        Ek = calEi(obj, maxJ)
    return Ek, maxJ


def innerLoop(obj, i):
    Ei = calEi(obj, i)
    if (obj.labelMatrix[i, 0] * Ei < -obj.toler and obj.alphas[i, 0] < obj.C) or \
            (obj.labelMatrix[i, 0] * Ei > obj.toler and obj.alphas[i, 0] > 0):
        Ej, j = selectJIndex(obj, i, Ei)
        alphaIold = obj.alphas[i, 0].copy()
        alphaJold = obj.alphas[j, 0].copy()
        if obj.labelMatrix[i, 0] == obj.labelMatrix[j, 0]:
            L = max(0, obj.alphas[i, 0] + obj.alphas[j, 0] - obj.C)
            H = min(obj.C, obj.alphas[i, 0] + obj.alphas[j, 0])
        else:
            L = max(0, obj.alphas[j, 0] - obj.alphas[i, 0])
            H = min(obj.C, obj.C - obj.alphas[i, 0] + obj.alphas[j, 0])
        if L == H:
            return 0
        eta = obj.K[i, i] + obj.K[j, j] - 2 * obj.K[i, j]
        if eta <= 0:
            return 0
        obj.alphas[j, 0] += obj.labelMatrix[j, 0] * (Ei - Ej) / eta
        print("before update alphas[j, 0]", obj.alphas[j, 0])
        obj.alphas[j, 0] = clipAlpha(obj.alphas[j, 0], L, H)
        print("update alphas[j, 0]",obj.alphas[j, 0],alphaJold)
        updateEi(obj, j)
        if np.abs(obj.alphas[j, 0] - alphaJold) < 0.00001:
            return 0
        obj.alphas[i, 0] += obj.labelMatrix[i, 0] * obj.labelMatrix[j, 0] * (alphaJold - obj.alphas[j, 0])
        updateEi(obj, i)
        b1 = -Ei - obj.labelMatrix[i, 0] * obj.K[i, i] * (obj.alphas[i, 0] - alphaIold) \
             - obj.labelMatrix[j, 0] * obj.K[i, j] * (obj.alphas[j, 0] - alphaJold) + obj.b
        b2 = -Ej - obj.labelMatrix[i, 0] * obj.K[i, j] * (obj.alphas[i, 0] - alphaIold) \
             - obj.labelMatrix[j, 0] * obj.K[j, j] * (obj.alphas[j, 0] - alphaJold) + obj.b
        if obj.alphas[i, 0] > 0 and obj.alphas[i, 0] < obj.C:
            obj.b = b1
        elif obj.alphas[j, 0] > 0 and obj.alphas[j, 0] < obj.C:
            obj.b = b2
        else:
            obj.b = (b1 + b2) / 2.0
        return 1
    else:
        return 0


def realSMO(trainSet, trainLabels, C, toler, maxIter=40):
    obj = osStruct(trainSet, trainLabels, C, toler)
    entrySet = True
    iterNum = 0
    alphapairschanged = 0
    while (iterNum < maxIter) and (alphapairschanged > 0 or entrySet):
        print("iterNum=",iterNum)
        alphapairschanged = 0
        if entrySet:
            for i in range(obj.m):
                alphapairschanged += innerLoop(obj, i)
            iterNum += 1
        else:
            vaildalphsaList = np.nonzero(obj.alphas<C)[0]
            #print("vaildalphsaList=",vaildalphsaList)
            for i in vaildalphsaList:
                alphapairschanged += innerLoop(obj, i)
            iterNum += 1
        if entrySet:
            entrySet = False
        elif alphapairschanged == 0:
            entrySet = True
        print("alphapairschanged=",alphapairschanged)
    return obj.alphas, obj.b


def calcWs(alphas, dataArr, classLabels):
    w = np.zeros((2, 1))
    for i in range(len(alphas)):
      t=dataArr[i, :].T.reshape(2,1)
      w += alphas[i] * classLabels[i] * t
    return w


def init_data():
    data = []
    labels = []
    points = []
    for i in range(20):
        x = random.random()
        y = random.random()
        if y > x:
            data.append([x, y, 1])
            labels.append([1])
            points.append([x, y])
        elif y < x:
            data.append([x, y, -1])
            labels.append([-1])
            points.append([x, y])
    labels=np.array(labels)
    points=np.array(points)
    alpha,b=realSMO(points,labels,1,0.0001)
    ws = calcWs(alpha, points, labels)
    print("alpha=",alpha)
    print("ws = \n", ws)
    print("b = \n", b)
    fig = plt.figure(figsize=(8, 6))
    ax1 = fig.add_subplot(111)
    # 设置标题
    ax1.set_title('Scatter Plot')
    # 设置X轴标签
    plt.xlabel('X')
    # 设置Y轴标签
    plt.ylabel('Y')
    for d in data:
        if d[2] > 0:
            ax1.scatter(d[0], d[1], c='r', s=5)
        else:
            ax1.scatter(d[0], d[1], c='b', s=5)
    ax1.plot([0, 1], [0, 1], color="black", linewidth=1)
    ax1.plot([0,1],[-b/ws[1],-b/ws[1]-ws[0]/ws[1]],color="blue", linewidth=1)
    plt.show()


init_data()



# def getSupportVectorandSupportLabel(trainSet, trainLabel, alphas):
#     vaildalphaList = np.nonzero(alphas.A)[0]
#     dataMatrix = np.array(trainSet, dtype=np.float)
#     labelMatrix = np.array(trainLabel, dtype=np.float).transpose()
#     sv = dataMatrix[vaildalphaList]  # 得到支持向量
#     svl = labelMatrix[vaildalphaList]
#     return sv, svl

# def predictLabel(data, sv, svl, alphas, b, kTup):
#     kernal = kernalTransfrom(sv, np.matrix(data, dtype=np.float), kTup).transpose()
#     fxi = np.multiply(svl.T, alphas[alphas != 0]) * kernal + b
#     return np.sign(float(fxi))
