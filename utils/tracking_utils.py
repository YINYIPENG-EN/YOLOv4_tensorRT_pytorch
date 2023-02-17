'''
采用iou进行目标的跟踪，对前一帧和当前帧iou判断是否为同一目标
'''
import copy

def compute_iou(box_a, box_b, diou=True):
    S_rec1 = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    S_rec2 = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    # 并集面积
    sum_area = S_rec1 + S_rec2

    left_X = max(box_a[1], box_b[1])  # 相交的左上角
    left_Y = max(box_a[0], box_b[0])

    right_X = min(box_a[3], box_b[3])  # 相交的右下角
    right_Y = min(box_a[2], box_b[2])

    # ----------------用来算diou内容--------------------
    center_a_x = (box_a[3] - box_a[1]) / 2 + box_a[1]
    center_a_y = (box_a[2] - box_a[0]) / 2 + box_a[0]

    center_b_x = (box_b[3] - box_b[1]) / 2 + box_b[1]
    center_b_y = (box_b[2] - box_b[0]) / 2 + box_b[0]
    # 计算两个框中心点的欧式距离
    center_distance = (center_b_x-center_a_x)**2 + (center_b_y-center_a_y) ** 2

    # 计算两个框最小矩形的左上角和右下角
    closebox_min_x = min(box_a[1], box_b[1])
    closebox_min_y = min(box_a[0], box_b[0])
    closebox_max_x = max(box_a[3], box_b[3])
    closebox_max_y = max(box_a[2], box_b[2])

    # 计算两个框最小矩形的对角线距离
    closebox_distance = (closebox_min_x-closebox_max_x)**2 + (closebox_min_y- closebox_max_y)**2
    # -------------------------------------------------

    if left_X >= right_X or left_Y >= right_Y:
        return 0.
    else:
        # 相交的面积
        inter_area = (right_X - left_X) * (right_Y - left_Y)
        iou = inter_area / (sum_area - inter_area)
        return iou - (center_distance/closebox_distance) if diou else iou


def tracking_box(box_data, target_dict, track_index, iou_thre=0.8):
    reg_dict = {}  # 用来记录识别id以及信息
    # box_data是列表形式
    for bbox in box_data:  # box_data会保存目标坐标，每个元素bbox是元组的形式，包含内容(x1.y1,x2,y2,score)
        xa0, ya0, xa1, ya1, score = bbox  # 当前帧目标的box
        is_tracke = False  # 状态初始化
        for k_ in target_dict.keys():
            xb0, yb0, xb1, yb1, s, _ = target_dict[k_]  # 前一帧的box
            iou_ = compute_iou((ya0, xa0, ya1, xa1), (yb0, xb0, yb1, xb1))
            if iou_ > iou_thre:  # 跟踪目标成功
                reg_dict[k_] = (xa0, ya0, xa1, ya1, score, iou_)
                is_tracke = True  # 跟踪√
                # print('iou_', iou_)
        if not is_tracke:  # 表示新的目标
            reg_dict[track_index] = (xa0, ya0, xa1, ya1, score, 0.)  # 记录目标id以及信息
            track_index += 1
            if track_index >= 65535:  # 越界归零
                track_index = 0
            if track_index >= 100:
                track_index = 0
    target_dict = copy.deepcopy(reg_dict)  # 记录target信息
    # print(target_dict)
    return target_dict, track_index

def target_tracking(box_data, target_dict, track_index):
    target_dict, track_index = tracking_box(box_data, target_dict, track_index)
    return target_dict, track_index