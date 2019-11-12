import numpy as np
import matplotlib.pyplot as plt
import random as rd
import time
import cvxopt as cv
max_num=25 #样本最大数目
max_ZS_num = 6      #噪声点个数
#rd.seed(10)
def RandomData(): #随机数
    x=[]
    cnt=0
    while cnt<max_num :
          cnt+=1
          tmp_random=np.random.normal()#-1+2*rd.random()
          x.append(tmp_random)
    return x
def getDataSet(x:list,y:list): #获取数据集
    cnt=0
    x_simple=[]
    x1_1=[]
    x2_1=[]
    x1_2=[]
    x2_2=[]
    y_lable=[]
    w = [-1, -1]
    b = 0.3
    while cnt<max_num:
        y_lable_tmp=np.sign((b+w[0]*x[cnt]+w[1]*y[cnt]))
        if y_lable_tmp==0:
            continue
        y_lable.append(y_lable_tmp)
        if  (y_lable_tmp==1):
            x1_1.append(x[cnt])
            x2_1.append(y[cnt])
        elif y_lable_tmp==-1:
            x1_2.append(x[cnt])
            x2_2.append(y[cnt])
        x_simple.append([x[cnt],y[cnt],1.0*y_lable[cnt]])
        cnt+=1
    return x_simple
def getb(alpha,x_train,y_label):
    cnt=0
    sum=float(0.0)
    alpha=list(alpha)
    x_train=np.matrix(x_train)
    y_label=np.matrix(y_label)
    n,m=x_train.shape
    eps=1e-10
    minx=1e10
    maxx=1e-10
    support_vector=[]
    support_vector_index=[]
    for i in range(len(alpha)):
        if abs(alpha[i])<eps:
            continue
        asum=0.0
           # a=math.isclose(1e-100,0.0)
        for j in range(m):
            asum+=alpha[j]*y_label[j]*(x_train[:,j].transpose()*x_train[:,i])

        b=y_label[i]-asum
        if (b>maxx):
            maxx=b
        if (b<minx):
            minx=b
        sum+=b
        cnt+=1
        support_vector.append(x_train[:,i])
        support_vector_index.append(i)
    b=sum/cnt
    print("support_NUM: ",cnt)
    return  b,support_vector,support_vector_index
def getW(alpha,x_train,y_label):
    ans=np.array([[0],[0]])
    n,m=x_train.size
    for i in range(m):
        ans=ans+alpha[i]*y_label[i]*x_train[:,i]
    return ans
def SVMLinear(x_train,y_label): # 参数X=(x1,x2...xn) yt=(y1,y2...yn)
    n=len(y_label)
    x_train=cv.matrix(x_train)
    y_label=cv.matrix(y_label)
    k=x_train.trans()*x_train     # k=xt*x
    y=y_label*y_label.trans()     # y=y_label*y_labelt
    H=np.multiply(k,y)            #h=(xt*x).(y_label*y_labelt)
    H=cv.matrix(H)
    f=-np.ones((y_label.size))    #f=(-1,-1,..-1)t
    f=cv.matrix(f)
    b=np.zeros((y_label.size))    #b=(0,0,....0)t
    b=cv.matrix(b)
    A=-np.identity(n)             #A=-En(单位阵)
    A=cv.matrix(A)
    Aeq=y_label.trans()           #Aeq=yt
    Aeq=cv.matrix(Aeq)
    beq=cv.matrix([0.0])           #beq=0
    cv.solvers.options['show_progress']=False
    result=cv.solvers.qp(H,f,A,b,Aeq,beq)
    alpha=result['x']
    print("alpha:",alpha.T)
    w=getW(alpha,x_train,y_label)
    b,support_vector,support_vector_index=getb(alpha,x_train,y_label)
    print("support_vector:",support_vector)
    w=np.matrix(w)
    return w,b,support_vector,support_vector_index
def getLoss(test_w,test_b,x_train,test_index,y_label):    #计算第k组预测错误的样本个数
    Lenindex=len(test_index)
    WrongNUm=0
    x_train=np.matrix(x_train)
    for i in range(Lenindex):
        testyn=test_w.T*x_train[:,test_index[i]]+test_b
        if testyn<0 and y_label[test_index[i]]==1:
            WrongNUm+=1
        if testyn>=0 and y_label[test_index[i]]==-1:
            WrongNUm+=1
    return WrongNUm
def LeaveOneOut(x_train,y_label,k):
    C = [1e-5, 1e-4, 1e-3,0.0055, 1e-2, 0.055, 1e-1, 1, 10, 1e2, 1e3, 1e4, 1e5]
    c_len=len(C)
    n,m=x_train.shape
    groupLen=m//k
    index=[]
    for i in range(k-1):
        index.append([i*groupLen+j  for j in range(groupLen)])
    index.append([j for j in range((k-2)*groupLen+groupLen,m)])
    minCWronNum=1e10
    ansc=C[0]
    for i in range(c_len):
        cur_c=C[i]
        Wrongnum=0
        for j in range(k):
            cur_index=[]
            for l in range(k):
                if(l==j):
                    continue
                cur_index=cur_index+index[l]
            cur_x_train=x_train[:,cur_index]
            cur_y_label=y_label[cur_index]
            cur_w,cur_b,cur_support_V=SVMNOLinear(cur_x_train,cur_y_label,cur_c)
            Wrongnum+=getLoss(cur_w,cur_b,x_train,index[j],y_label)
        if minCWronNum>Wrongnum:
            minCWronNum=Wrongnum
            print("minWNum:",minCWronNum,"C:",C[i])
            ansc=C[i]
    return ansc
def SVMNOLinear(x_train,y_label,C): # 参数X=(x1,x2...xn) yt=(y1,y2...yn)
    n=len(y_label)
    x_train=cv.matrix(x_train)
    y_label=cv.matrix(y_label)
    k=x_train.trans()*x_train     # k=xt*x
    y=y_label*y_label.trans()     # y=y_label*y_labelt
    H=np.multiply(k,y)            #h=(xt*x).(y_label*y_labelt)
    H=cv.matrix(H)
    f=-np.ones((y_label.size))    #f=(-1,-1,..-1)t
    f=cv.matrix(f)
    b1=np.zeros((y_label.size))    #b=(0,0,....0)t
    b2=C*np.ones(y_label.size)
    b=np.vstack((b1,b2))
    b=cv.matrix(b)
    e1=-np.identity(n)             #A=-En(单位阵)
    e2=np.identity(n)
    A=np.vstack((e1,e2))
    A=cv.matrix(A)
    Aeq=y_label.trans()           #Aeq=yt
    Aeq=cv.matrix(Aeq)
    beq=cv.matrix([0.0])           #beq=0
    result=cv.solvers.qp(H,f,A,b,Aeq,beq)
    alpha=result['x']
    print("SVMNOalpha:",alpha.T,"cur_C:",C)
    w=getW(alpha,x_train,y_label)
    b,support_vector,support_vector_index=getb(alpha,x_train,y_label)
    # print("support_vector:\n",support_vector)
    w=np.matrix(w)
    return w,b,support_vector
def drawPlot(data_set,w_pla,b,xs,xxs,isNOlinear=False):
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    color_index=['blue' if x==1 else 'red' for x in data_set[:,2]]
    plt.scatter(data_set[:, 0], data_set[:, 1], c=color_index,marker='.')  # 绘制可视化图
    w_pla=w_pla.tolist()
    xs=xs.tolist()
    xxs=xxs.tolist()
    xx=np.linspace(-1,8)
    a=-w_pla[0][0]/w_pla[1][0]
    b=-b[0]/w_pla[1][0]
    sa=a
    sb=w_pla[0][0]*xs[0][0]+w_pla[1][0]*xs[1][0]
    sb=sb/w_pla[1][0]
    sxb = w_pla[0][0] * xxs[0][0] + w_pla[1][0] * xxs[1][0]
    sxb = sxb / w_pla[1][0]
    yyxs_svm=sa*xx+sxb
    yys_svm=sa*xx+sb
    yy_pla = a * xx + float(b[0])
    if isNOlinear:
        print("带松弛变量的SVM找到的直线:y=" + str(a) + "*x+" + str(float(b[0])))
    else :
        print("SVM 预测的直线:y=" + str(a) + "*x+" + str(float(b[0])))
    p4 = plt.plot(xx, yy_pla, c='green', label='SVM找到的直线')
    plt.plot(xx, yys_svm, c='purple', label='间距直线',linestyle=":")
    plt.plot(xx, yyxs_svm, c='red', label='间距直线', linestyle=":")
    plt.legend()
    plt.show()
def getXX(w,b,support_vector):
    n=len(support_vector)
    mindiss=1e10
    mindisx=1e10
    poss=0
    posx=0
    for i in range(n):
        yn=w.transpose()*np.matrix(support_vector[i])+b[0]
        dis=float(yn[0])/((float(w[0][0])**2+float(w[1][0])**2)**0.5)
        if (yn>=0 and dis<mindiss):
            mindiss=dis
            poss=i
        elif (yn<0 and (-dis)<mindisx):
            mindisx=-dis
            posx=i
    return support_vector[poss],support_vector[posx],poss,posx
def getPre(w,b,data):
    n,m=data.shape
    ans=[]
    for i in range(n):
        yn=data[i]*w+b
        if(yn>=0):
            ans.append(1)
        else :
            ans.append(-1)
    return ans
def main():
    k=5
    data_set = np.loadtxt("SVM_data.txt")
    train_data = data_set[:,0:2]   # 训练特征空间
    train_target = np.sign(data_set[:,2])  # 训练集类标号
    x_simple=train_data.T
    y_lable=train_target
    w,b,support_vector,support_vector_index=SVMLinear(x_simple,y_lable)             #线性可分集合 SVM
    xs,xx,poss,posx=getXX(w,b,support_vector)                  #获取距离直线最近的点
    y_lable[posx]=-1*y_lable[posx]
    drawPlot(data_set, w, b, xs, xx)                   #显示线性可分的情况下SVM效果图
    posx=support_vector_index[posx]
    poss=support_vector_index[poss]
    # # n, m = data_set.shape
    # # maxpp = 0
    # # posr = 0
    # # for i in range(n):
    # #     if (data_set[i][2] == 1):
    # #         continue
    # #     if maxpp < data_set[i][0]:
    # #         maxpp = data_set[i][0]
    # #         posr = i
    # # data_set[posr][2] = -1 * data_set[posr][2]
    data_set[posx][2]=-1*data_set[posx][2]                                                          #改变数据集，使其变为线性不可分的情况，通过松弛变量来找到C
    data_set[poss][2]=-1*data_set[poss][2]
    c = LeaveOneOut(x_simple, y_lable, k)                      #留一法调整参数c
    w_NOLineear, b_no, support_vector_no = SVMNOLinear(x_simple, y_lable, c)
    xs_no,xx_no,poss_no,posx_no=getXX(w_NOLineear,b_no,support_vector_no)
    print("w_NO:",w_NOLineear.T,"b_NO:",b_no,"C:",c)
    print("w:",w.T,"b",b)
    drawPlot(data_set,w_NOLineear,b_no,xs_no,xx_no,True)
if __name__ == '__main__':
    main()


# import random
# import matplotlib.pyplot as plt
# import numpy as np
#
# data = []
# labels = np.array()
# points = np.array()
# alpha = np.zeros((20, 1))
# error_cache = np.zeros((20, 2))
# b = 0
# # 常数c
# c=0
# # 容错率
# tol=0
#
# def init_data():
#     for i in range(20):
#         x = random.random()
#         y = random.random()
#         if y > x:
#             data.append([x, y, 1])
#             labels.append(np.array([1]))
#             points.append(np.array([x, y]))
#         elif y < x:
#             data.append([x, y, -1])
#             labels.append(np.array([-1]))
#             points.append(np.array([x, y]))
#
#
# def draw():
#     fig = plt.figure(figsize=(8, 6))
#     ax1 = fig.add_subplot(111)
#     # 设置标题
#     ax1.set_title('Scatter Plot')
#     # 设置X轴标签
#     plt.xlabel('X')
#     # 设置Y轴标签
#     plt.ylabel('Y')
#     # 画散点图
#     for d in data:
#         if d[2] > 0:
#             ax1.scatter(d[0], d[1], c='r', s=5)
#         else:
#             ax1.scatter(d[0], d[1], c='b', s=5)
#     ax1.plot([0, 1], [0, 1], color="black", linewidth=1)
#     plt.show()
#
#
# # 计算k处的误差值
# def cal_error(k):
#     fXk = float(np.multiply(alpha, labels).T * (points * points[k, :].T) + b)
#     eK = fXk - float(labels[k])
#     return eK
#
#
# # error_cache没有赋值，随机选另一个alpha j
# def select_j_rand(i, m):
#     j = i
#     while j == i:
#         j = int(random.uniform(0, m))
#     return j
#
#
# # 假定现在已经选取了第一个待优化的alpha i,选择另一个和i有较大差异的j
# def select_j(i, err_i):
#     max_k = -1
#     max_delta_err = 0
#     err_j = 0
#     error_cache[i] = np.array([1, err_i])
#     # .A 表示矩阵转数组,找到error_cache所有已初始化的
#     valid_err_cache_list = np.nonzero(error_cache[:, 0].A)
#     if len(valid_err_cache_list) > 1:
#         for k in valid_err_cache_list:
#             if k == i:
#                 continue
#             err_k = cal_error(k)
#             delta_err = np.abs(err_i - err_k)
#             if delta_err > max_delta_err:
#                 max_delta_err = delta_err
#                 max_k = k
#                 err_j = err_k
#     else:
#         max_k = select_j_rand(i, len(data))
#         err_j = cal_error(max_k)
#     return max_k, err_j
#
#
# def clip_alpha(j,h,l):
#     if j>h:
#         j=h
#     if l>j:
#         j=l
#     return j
#
#
# # 边界判定
# def inner_loop(i):
#     err_i=cal_error(i)
#     if (labels[i]*err_i<-tol and alpha[i]<c)or \
#         (labels[i]*err_i>tol and alpha[i]>0):
#         err_j,j=select_j(i,err_i)
#         i_old=alpha[i].copy()
#         j_old=alpha[j].copy()
#         if labels[i]==labels[j]:
#             l=max(0,alpha[i]+alpha[j]-c)
#             h=min(c,alpha[i]+alpha[j])
#         else:
#             l=max(0,alpha[j]-alpha[i])
#             h=min(c,c-alpha[i]+alpha[j])
#         if l==h:
#             return 0
#         eta = 2*points[i,:]*points[j,:].T-points[i,:]*points[i,:].T-\
#         points[j, :]*points[j,:].T
#         if eta >=0:
#             return 0
#         alpha[j]-=labels[j]*(err_i-err_j)/eta
#         alpha[j]=clip_alpha(alpha[j],h,l)
#         error_cache[j]=np.array([1,cal_error(j)])
#         if abs(alpha[j]-j_old)<0.0001:
#             return 0
#         alpha[i]+=labels[j]*labels[i]*(j_old-alpha[j])
#         error_cache[i]=np.array([1,cal_error(i)])
#         b1=b-err_i-labels[i]*(alpha[i]-i_old)*points[i,:]* \
#            points[i, :].T-labels[j]*(alpha[j]-j_old)*\
#             points[i,:]*points[j,:].T
#         b2=b-err_i-labels[i]*(alpha[i]-i_old)*points[i,:]* \
#            points[j, :].T-labels[j]*(alpha[j]-j_old)*\
#             points[j,:]*points[j,:].T
#         if alpha[i]>0 and c > alpha[i]:
#             b=b1
#         elif alpha[j]>0 and c>alpha[j]:
#             b=b2
#         else:
#             b=(b1+b2)/2.0
#         return 1
#     else:
#         return 0
#
#
#
# # 输出目标b和参数alpha
# def smo(max_iterator):
#     iterator=0
#     entire_set=True
#     alpha_change=0
#     while iterator<max_iterator and (alpha_change>0 or entire_set):
#         alpha_change = 0
#         if entire_set:
#             for i in range(len(data)):
#
#
#
# # def svm():
import numpy as np
data=np.zeros((5,1))
print(data)