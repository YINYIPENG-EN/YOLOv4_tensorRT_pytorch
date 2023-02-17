import cv2


def object_track_pipeline(img, target_dict):
    for idx, id_ in enumerate(sorted(target_dict.keys(), key=lambda x:x, reverse=False)):
        x_min, y_min, x_max, y_max, score, iou_ = target_dict[id_]
        '''
        cv2.putText传入参数：图片，添加的文字，左上角坐标，字体，字体大小，颜色，字体粗细
        '''
        cv2.putText(img, 'ID {}'.format(id_), (int(x_min + 2), int(y_min + 15)), cv2.FONT_HERSHEY_COMPLEX, 0.45, (255, 0, 0), 5)
        cv2.putText(img, 'ID {}'.format(id_), (int(x_min + 2), int(y_min + 15)), cv2.FONT_HERSHEY_COMPLEX, 0.45,
                    (173, 255, 73))
    return img
