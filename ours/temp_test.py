
import os
import numpy as np
from pprint import pprint,pformat
from collections import defaultdict

from queue import PriorityQueue




def test_1():
    data = [3,4,2,1]
    b = data[[0,8]]
    print(b)


def test_2():
    a = {1:"mml"}
    b = a.get(2)
    print()

def xcycwh_to_x1y1x2y2(bbox,W,H):
    xc = bbox[0]
    yc = bbox[1]
    w = bbox[2]
    h = bbox[3]

    # 1. 归一化 -> 像素
    x_c = xc * W
    y_c = yc * H
    bw  = w  * W
    bh  = h  * H

    # 2. 中心 -> 左上 / 右下
    x1 = x_c - bw / 2
    y1 = y_c - bh / 2
    x2 = x_c + bw / 2
    y2 = y_c + bh / 2

    # 3. 转 int + 裁剪
    x1 = max(0, min(W - 1, int(round(x1))))
    y1 = max(0, min(H - 1, int(round(y1))))
    x2 = max(0, min(W - 1, int(round(x2))))
    y2 = max(0, min(H - 1, int(round(y2))))

    return [x1,y1,x2,y2]

def test_3():
    data = [1]
    b = data[:-1]
    print(b)

def test_4():
    a = [1,"b.j"]
    print(a)

def test_5():
    a = ["a","b","c"]
    print(a[-2:])
def test_6():
    data = [1]*10
    print(data)

def test_7():
    def get_all_img_name():
        img_dir = "/data/mml/data_debugging_data/datasets/VOC2012-yolo/train/images"
        img_name_list = []
        for filename in os.listdir(img_dir):
            filepath = os.path.join(img_dir, filename)
            if os.path.isfile(filepath):
                img_name_list.append(filename)
        return img_name_list
    img_name_list = get_all_img_name()

def test_8():
    
    from collections import defaultdict

    path = "/data/mml/data_debugging_data/datasets/VisDrone-yolo/train/labels/9999987_00000_d_0000049.txt"
    arr = np.loadtxt(path).reshape(-1, 5)          # [cls, x, y, w, h]
    arr = np.round(arr, 6)                         # 适当四舍五入避免格式差异

    m = defaultdict(list)
    for i, row in enumerate(arr, 1):
        m[tuple(row.tolist())].append(i)

    for k, idxs in m.items():
        if len(idxs) > 1:
            print(f"重复行号 {idxs}: {k}")

def test_9():
    data = {
    "name": "张三",
    "age": 25,
    "address_list": [{
        "city": "北京",
        "street": "朝阳路",
        "postcode": "100000"
    },
    {
        "city": "北京",
        "street": "朝阳路",
        "postcode": "100000"
    }],
    "hobbies": ["阅读", "游泳", "编程"]
}   
    print(data)
    # 直接打印
    pprint(data, indent=2)

    # 获取格式化的字符串
    formatted = pformat(data, indent=2)
    print(formatted)

def test10():
    # 创建一个优先级队列
    priority_queue = PriorityQueue()

    # 添加元素到队列中，格式为(优先级, 值)
    priority_queue.put((1, '任务1'))
    priority_queue.put((2, '任务2'))
    priority_queue.put((0, '任务0'))

    # 获取并弹出优先级最高的元素
    while not priority_queue.empty():
        priority, task = priority_queue.get()
        print(f"处理任务: {task}")

def test11():
    clusters = defaultdict(list)
    print(list(clusters.values()))
if __name__ == "__main__":
    # test_2()
    test11()
    # bbox = [0.499,0.4866666666666667,0.106,0.14666666666666667]
    # W = 500
    # H = 375
    # b = xcycwh_to_x1y1x2y2(bbox,W,H)
    # print()