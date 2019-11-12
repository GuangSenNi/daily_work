import time
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing
from multiprocessing import Manager

wide = 300
height = 200
count = 100
# 图形中心的位置
orig_x = wide * 2 / 3
orig_y = height / 2


def iteration(x, y):
    limit = count + 1
    a = (x - orig_x) / (wide / 3)
    b = (orig_y - y) / (height / 2)
    # complex生成复数
    c = complex(a, b)
    z = complex(0, 0)
    for m in range(limit):
        z = z * z + c
        if z.real > 2 or z.imag > 2:
            return 1
    return 0


# 使用消息队列 按行分任务
def sub_p_calculate(x, mq):
    for y in range(height):
        limit = count + 1
        a = (x - orig_x) / (wide / 3)
        b = (orig_y - y) / (height / 2)
        # complex生成复数
        c = complex(a, b)
        z = complex(0, 0)
        t = 0
        for m in range(limit):
            z = z * z + c
            if z.real > 2 or z.imag > 2:
                t = 1
                break
        mq.put([y, x, t])


# 使用map分任务
def sub_use_map(v):
    x = v[1]
    y = v[0]
    limit = count + 1
    a = (x - orig_x) / (wide / 3)
    b = (orig_y - y) / (height / 2)
    # complex生成复数
    c = complex(a, b)
    z = complex(0, 0)
    t = 0
    for m in range(limit):
        z = z * z + c
        if z.real > 2 or z.imag > 2:
            t = 1
            break
    return t


# 串行计算
def serial_cal():
    img = Image.new("RGB", (wide, height))
    img2 = np.array(img)
    t0 = time.perf_counter()
    for i in range(height):
        for j in range(wide):
            ite = iteration(j, i)
            if ite:
                img2[i, j, 0] = ite
                img2[i, j, 1] = abs(j - orig_x) / wide * 255
                img2[i, j, 2] = abs(i - orig_y) / height * 255
            else:
                img2[i, j, :] = 0
    t1 = time.perf_counter()
    print("单进程执行时间:", t1 - t0)
    return img2


# 进程池pool.apply_async+消息队列 进程通信耗时较大 按行分配任务以减小通信开销
def pool_use_mq():
    img1 = Image.new("RGB", (wide, height))
    img3 = np.array(img1)
    # 巨坑 进程池通信要用Manager下的queue
    q = Manager().Queue()
    pool = multiprocessing.Pool(processes=4)
    t2 = time.perf_counter()
    for j in range(wide):
        pool.apply_async(sub_p_calculate, (j, q,))
    pool.close()
    pool.join()
    while not q.empty():
        result = q.get()
        i = result[0]
        j = result[1]
        if result[2]:
            img3[i, j, 0] = 1
            img3[i, j, 1] = abs(j - orig_x) / wide * 255
            img3[i, j, 2] = abs(i - orig_y) / height * 255
        else:
            img3[i, j, :] = 0
    t3 = time.perf_counter()
    print("并行使用消息队列执行时间:", t3 - t2)
    return img3


# pool使用map优化，进一步减小通信开销
def pool_use_map():
    pool = multiprocessing.Pool(processes=4)
    t4 = time.perf_counter()
    arr = []
    for i in range(height):
        for j in range(wide):
            arr.append([i, j])
    result = pool.map_async(sub_use_map, arr)
    pool.close()
    pool.join()
    t5 = time.perf_counter()
    img = Image.new("RGB", (wide, height))
    img2 = np.array(img)
    for k in range(len(arr)):
        j = arr[k][1]
        i = arr[k][0]
        if result.get()[k]:
            img2[i, j, 0] = 1
            img2[i, j, 1] = abs(j - orig_x) / wide * 255
            img2[i, j, 2] = abs(i - orig_y) / height * 255
        else:
            img2[i, j, :] = 0
    print("并行使用map运行时间", t5 - t4)
    return img2


# 进程切换及消息队列都会严重影响运行时间 迭代次数越多并行优势越明显
if __name__ == '__main__':
    img2 = serial_cal()
    plt.imshow(img2)
    plt.show()

    # fig = plt.figure()
    # sub1 = fig.add_subplot(211)
    # sub1.imshow(img2)
    # sub2 = fig.add_subplot(212)
    # sub2.imshow(img3)
    # plt.tight_layout()
    # plt.show()
