import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import time
# from tqdm import tqdm

np.set_printoptions(suppress=True)
'''============分片蝴蝶混沌轨道堆栈聚集搜索算法  又称"Steins Gate算法",灵感来自于动漫《命运石之门》================'''

'''核心函数：产生三个轨道'''
def core(up,down,Z0,Step,Times,Aband):
    #lorenz
    Ox,Oy,Oz=1,1,Z0
    temp=Oz
    a,b,c=10,28,8/3
    x0,y0,z0=Ox,Oy,Oz


    ##rossler
    # Ox,Oy,Oz=0.1000, 0.1000, Z0
    # temp=Oz
    # a, b, c = 0.2, 0.2, 4.6

    TrackMat=[]

    P=6
    I=10**16
    step=Step
    times=Times
    aband=Aband
    MOD = 10 ** P

    upbond=up
    downbond=down
    for t in range(len(upbond)):
        track=[]
        Ox, Oy, Oz = 1, 1, temp+0.00001
        temp=Oz
        x0, y0, z0 = Ox, Oy, Oz

        for point in range(times):

            #lorenz
            Ox += step * (a * (y0 - x0))
            Oy += step * (b * x0 - x0 * z0 - y0)
            Oz += step * (x0 *y0 - c * z0)
            x0, y0, z0 = Ox, Oy, Oz

            # #rossler
            # Ox -= step * (Oy + Oz)
            # Oy += step * (Ox + a * Oy)
            # Oz += step * (b + Oz * (Ox - c))

            if Oz>=0:
                track.append((Oz*I)%MOD)
            else:
                track.append((-(Oz * I) % -MOD))
        del(track[:aband])
        TrackMat.append(track)
    TrackMat=np.array(TrackMat)

    #将散点压缩到指定区间内
    for row in range(len(TrackMat)):
        TrackMat[row]=TrackMat[row]*(upbond[row]-downbond[row])/MOD+downbond[row]

    return TrackMat

'''测试用函数'''
# # 简单峰值函数，求解域为xoy界面上-3~3的方形区间
# def F(mat):
#     return 3 * (1 - mat[0]) ** 2 * np.exp(-(mat[0] ** 2) - (mat[1] + 1) ** 2) - 10 * (mat[0] / 5 - mat[0] ** 3 - mat[1] ** 5) * np.exp(
#         -mat[0] ** 2 - mat[1] ** 2) - 1 / 3 ** np.exp(-(mat[0] + 1) ** 2 - mat[1] ** 2)

# #Rastrigin's 函数,求解域为xoy界面上-5~5的方形区间
# def F(mat):
#     return 20+mat[0]**2+mat[1]**2-10*(np.cos(2*3.14*mat[0])+np.cos(2*3.14*mat[1]))

# # SCHAFFER FUNCTION N. 2函数，求解域为xoy界面上-50~50的方形区间
# def F(mat):
#     return 0.5+(np.sin(mat[0]**2-mat[1]**2)**2-0.5)/((1+0.001*(mat[0]**2+mat[1]**2))**2)

# #大海捞针问题（极端）：初始搜索域为[-5.12~5.12]的方形区域
# def F(mat):
#     return (3/(0.05+(mat[0]**2+mat[1]**2)))**2+(mat[0]**2+mat[1]**2)**2

# #Schaffer函数:初始搜索域为[-4~4]的方形区域
# def F(mat):
#     return 0.5-(np.sin((mat[0]**2+mat[1]**2)**0.5)**2-0.5)/((1+0.001*(mat[0]**2+mat[1]**2))**2)
# #
# def F(mat):
#     return sum(mat**2)

# #Ackley函数：初始搜索域为[-50~50]的方形区域
# '''我要打十个！！！'''
# def F(mat):
#     MI=sum(mat**2)
#     MII=1/len(mat)*sum(np.cos(6.28*mat)**2)
#     return -(-20*np.exp(-0.2*((MI*(1/len(mat)))**0.5))-np.exp(MII)+20+np.exp(1))

# Griewank函数：初始搜索域为[-1000~1000]的方形区域
def F(mat):
    fitnessI=sum((mat**2)/4000)
    fitnessII=1
    for para in range(len(mat)):
        fitnessII=fitnessII*np.cos(mat[para]/((para+1)**0.5))
    return fitnessI-fitnessII+1

# #弱智问题
# def F(mat):
#     return mat[0] + 10*np.sin(5*mat[0]) + 7*np.cos(4*mat[0])

'''堆栈构建子算法(根据维数不同需要重新编写！考虑向全自动方向优化):遍历整个混沌轨道并依序将近优解压入栈中'''
#Mat是混沌轨道矩阵，即解集矩阵，一行为一个解，一列为一个变量
def stack_pushin(order,mat):
    transmat=np.array(mat).T
    Stack=[transmat[0]]#把第一个混沌解作为栈底
    Stack_value=[F(Stack[-1])]#给出栈底解值
    #检验函数是否正确
    # print(Stack_value)
    # time.sleep(10)
    #针对最小化问题
    if order == "0":
        for point in range(1,len(transmat)):
            Z=F(transmat[point])
            if Z<Stack_value[-1]:
                Stack.append(transmat[point])
                Stack_value.append(Z)
    #针对最大化问题
    elif order =="1":
        for point in range(1, len(transmat)):
            Z = F(transmat[point])
            if Z>Stack_value[-1]:
                Stack.append(transmat[point])
                Stack_value.append(Z)
    if len(Stack_value)<=1:
        pass
    else:
        del Stack[0]
        del Stack_value[0]

    Stack=np.array(Stack)
    return Stack,Stack_value



'''多维位面聚集出栈子算法'''
def assemble(Mat):
    #将栈中元素出栈，计算最密聚集中心
    S=np.zeros((len(Mat),len(Mat)))
    Sa=[]
    for row in range(len(Mat)):
        for col in range(len(Mat)):
            S[row][col]=np.sum((Mat[row]-Mat[col])**2)**0.5
    for row in range(len(S)-1):
        Sa.append(np.average(S[row][row+1:]))
    Track = Mat[-1]
    for point in range(len(Sa)-1,0,-1):
        if np.average(Sa[point-1:])>=Sa[point-1]:
            Track=np.average(Mat[point-1:],axis=0)
        else:
            break
    # print("Mat:",Mat)
    # # print("S:",S)
    # # print("Sa:",Sa)
    # print("Track:",Track)
    return Track


'''=========================================主运行控制代码=============================================='''
#搜索域初始化
# Mat=np.zeros(30)
# print(Mat)
# print(F(Mat))

print("请选择优化目标：（最小化输入0，最大化输入1）")
order=input()

MaxIteration = 1000
SSS = []
output = []
time_start = time.time()

can=[]
monitor=[]
timelog=[]
epochlog=[]
# ax = plt.figure().add_subplot(111, projection='3d')
# 初始搜索域
Digital = 19#10维！
bond = 1000#-1000~1000

originUp = up = t_up = [bond] * Digital
originDown = down = t_down = [-bond] * Digital
Scope = np.array(up) - np.array(down)

#震荡率
mutaterate=0.05

transsignal=0
plt.ion()
for iteration in (range(MaxIteration)):

    plt.cla()
    # print("新周期：",up)
    # print("新周期：",down)
    # 初始混沌轨道和停机精确度设定
    z0 = 0.9 + np.random.random() / 10
    accu = 0.000001
    bias=0.001
    speed = 0.7
    speed_M=0.4
    speed_G=0.6
    step = 0.01
    times = 1000#2000个混沌点
    aband = 100
    store=100#保留200底限

    times+=aband

    # 针对二维问题的初始化
    delta = [np.float('inf')]
    lastsulotion = 0
    count = 1
    originMap = []
    Mat = []
    points=[]


    # plt.ion()
    while(True):
        # plt.cla()
        # 产生对应多维切片混沌轨道
        chaoticindex = core(up,down,z0,step,times,aband)

        # 获取混沌轨道对应的解集（用于画图，此处不需要）
        originMap = F(chaoticindex)

        #基于切片混沌轨道运行堆栈算法，找出若干近优点的坐标矩阵
        track,value=stack_pushin(order,chaoticindex)
        track=np.mat(track)
        value=np.mat(value).T

        #构造用于聚集出栈的Mat
        Mat=np.hstack((track,value))
        Mat=np.array(Mat)

        #聚集出栈，获得一个解
        points=assemble(Mat)
        # points=Mat[-1]

        # ax.scatter(chaoticindex[0],chaoticindex[1],originMap,alpha = 1/4)
        # ax.scatter(Mat[:,0],Mat[:,1],Mat[:,-1], marker='^')
        # plt.tick_params(labelsize=15)
        # plt.tight_layout()
        # plt.pause(0.1)
        # print("Iteration:"+str(iteration)+"; ("+str(np.round(track1[-1],2))+ ","+str(np.round(track2[-1],2))+ ","+str(np.round(track3[-1],2))+ ","+str(np.round(track4[-1],2))+");F*="+str(np.round(value[-1],4)))

        delta.append(np.abs(lastsulotion - value[-1]))
        lastsulotion = F(points[:-1])
        '''停机条件'''
        # print("Δ'=",np.abs(delta[-1]))
        if np.abs(delta[-1]) <= accu:
            break
        count += 1

        '''规定新的搜索区域，超出解空间的放弃'''
        for index in range(len(up)):
            if points[index] + Scope[index] * (speed ** count) > originUp[index]:
                up[index]= originUp[index]
            else:
                up[index] = points[index] + Scope[index] * (speed ** count)
            if points[index] - Scope[index] * (speed ** count) < originDown[index]:
                down[index]=originDown[index]
            else:
                down[index] = points[index] - Scope[index] * (speed ** count)
        #密度指数式缩减防止堆栈过度聚集，同时提高算法速度
        times=(int)(times*speed_M)+store

        # "屏障内"的世界线跳动
        z0+=bias

    #添加“收束集”搜索结果
    SSS.append(np.round(points[:],4))
    #比较“世界线”优劣
    if len(SSS)>1:
        if order=="1":
            if SSS[-1][-1]>output[-1]:
                output=SSS[-1]
                timelog.append(time.time())
                epochlog.append(iteration)
                can.append(SSS[-1].copy())
                monitor.append(can[-1][-1])
                transsignal=1
                print("时间：",np.round(time.time()-time_start,2),";迭代：", iteration+1, ":", SSS[-1])
            else:
                monitor.append(can[-1][-1])
                print("时间：",np.round(time.time()-time_start,2),";迭代：", iteration+1, ":", "value=",SSS[-1][-1],",Miss Match...")
        else:
            if SSS[-1][-1]<output[-1]:
                output=SSS[-1]
                timelog.append(time.time())
                epochlog.append(iteration)
                can.append(SSS[-1].copy())
                monitor.append(can[-1][-1])
                print("时间：",np.round(time.time()-time_start,2),";迭代：", iteration+1, ":", SSS[-1])
                transsignal=1
            else:
                monitor.append(can[-1][-1])
                print("时间：",np.round(time.time()-time_start,2),";迭代：", iteration+1, ":", "value=",SSS[-1][-1], ",Miss Match...")
    else:
        output=SSS[-1]
        timelog.append(time.time())
        epochlog.append(iteration)
        can.append(SSS[-1].copy())
        transsignal=1
        monitor.append(can[-1][-1])
        print("时间：",np.round(time.time()-time_start,2),";迭代：",iteration+1,":",SSS[-1])

    '''自适应搜索域变异'''
    #变异几率
    odd=np.random.random()
    reset=np.random.random()
    #产生新解时
    if transsignal==1 or (reset>mutaterate*4 and reset<=mutaterate*5):
        #计算新的搜索域，并保存到临时容器
        originUp = up = [bond] * Digital
        originDown = down = [-bond] * Digital
        for index in range(len(up)):
            if can[-1][index] + Scope[index] * (speed_G ** (np.log2(1+len(can)))) > originUp[index]:
                t_up[index]= originUp[index]
            else:
                t_up[index] = can[-1][index] + Scope[index] * (speed_G ** (np.log2(1+len(can))))
            if can[-1][index] - Scope[index] * (speed_G ** (np.log2(1+len(can)))) < originDown[index]:
                t_down[index]=originDown[index]
            else:
                t_down[index] = can[-1][index] - Scope[index] * (speed_G ** (np.log2(1+len(can))))
        print("上下界缩减：", np.vstack([t_up,t_down]))
        # print("上界缩减：",t_up)
        # print("下界缩减：",t_down)
    elif odd<=mutaterate*4:
        middle=(np.array(t_up)+np.array(t_down))/2
        d=((np.array(t_up)-np.array(t_down))/2)*speed_G
        # print("验算1：",middle,d)
        t_up = middle+d
        t_down = middle-d
        # print("上界震荡：",t_up)
        # print("下界震荡：",t_down)
        print("上下界振荡：",  np.vstack([t_up,t_down]))
    up=t_up.copy()
    down=t_down.copy()

    transsignal=0
    plt.plot(monitor)
    plt.title("epoch:"+str(iteration+1)+";Solution="+str(can[-1]))
    plt.tick_params(labelsize=15)
    plt.pause(0.01)
time_end = time.time()
print("最终最优解：:", can[-1])
plt.ioff()
# print("共计迭代",count-1,"次，最优点：(", track1[-1], ",", track2[-1],",",track3[-1],",",track4[-1], "),最优值为：", value[-1])
print('用时：',timelog[-1]-time_start)
# plt.show()
# plt.plot(delta)
# plt.title("Delta")
# plt.show()

