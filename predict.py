import time
import cv2
from yolo import YOLO, LOGGER, np, Image
import argparse


def run(opt):
    weights, predict, video, video_path, fps, ONNX, TRT, input_shape, conf, nms = opt.weights, opt.img, opt.video, \
                                    opt.video_path, opt.fps, opt.onnx, opt.engine, opt.input_shape, opt.conf, opt.nms
    yolo = YOLO(weights, input_shape, conf, nms, ONNX, TRT)
    video_fps = 25.0
    video_save_path = "results.mp4"
    test_interval = 100
    if type(eval(video_path)) == int:
        video_path = eval(video_path)
    if predict:
        '''
        1、如果想要进行检测完的图片的保存，利用r_image.save("img.jpg")即可保存，直接在predict.py里进行修改即可。 
        2、如果想要获得预测框的坐标，可以进入yolo.detect_image函数，在绘图部分读取top，left，bottom，right这四个值。
        3、如果想要利用预测框截取下目标，可以进入yolo.detect_image函数，在绘图部分利用获取到的top，left，bottom，right这四个值
        在原图上利用矩阵的方式进行截取。
        4、如果想要在预测图上写额外的字，比如检测到的特定目标的数量，可以进入yolo.detect_image函数，在绘图部分对predicted_class进行判断，
        比如判断if predicted_class == 'car': 即可判断当前目标是否为车，然后记录数量即可。利用draw.text即可写字。
        '''
        while True:
            img = input('Input image filename:')
            try:
                # image = Image.open(img) # 采用Image用trt推理的时候精度下降非常厉害。具体原因不明
                image = cv2.imread(img)
            except:
                LOGGER.info('Open Error! Try again!')
                continue
            else:
                r_image = yolo.detect_image(image)
                r_image.show()
                r_image.save('results.jpg')  # jpg

    elif video:
        capture = cv2.VideoCapture(video_path)
        if video_save_path != "":
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            size = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            out = cv2.VideoWriter(video_save_path, fourcc, video_fps, size)

        ref, frame = capture.read()
        if not ref:
            raise ValueError("未能正确读取摄像头（视频），请注意是否正确安装摄像头（是否正确填写视频路径）。")

        fps = 0.0
        while (True):
            t1 = time.time()
            ref, frame = capture.read()
            if not ref:
                break
            frame = np.array(yolo.detect_image(frame))  # detect_image返回的是Image
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            fps = (fps + (1. / (time.time() - t1))) / 2
            print("fps= %.2f" % (fps))
            frame = cv2.putText(frame, "fps= %.2f" % (fps), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow("video", frame)
            c = cv2.waitKey(1) & 0xff
            if video_save_path != "":
                out.write(frame)

            if c == 27:
                capture.release()
                break

        LOGGER.info("Video Detection Done!")
        capture.release()
        if video_save_path != "":
            LOGGER.info("Save processed video to the path :" + video_save_path)
            out.release()
        cv2.destroyAllWindows()
    elif fps:
        img = Image.open('img/street.jpg')
        tact_time = yolo.get_FPS(img, test_interval)
        LOGGER.info(str(tact_time) + ' seconds, ' + str(1 / tact_time) + 'FPS, @batch_size 1')


if __name__ == "__main__":
    parser = argparse.ArgumentParser("TRT YOLOv4 ")
    parser.add_argument('--weights', type=str, default='model_data/yolo4_weights.pth', help='weight path')
    parser.add_argument('--img', action='store_true', default=False, help='image predict')
    parser.add_argument('--video', action='store_true', default=False, help='video predict')
    parser.add_argument('--video_path', type=str, default='0', help='video path')
    parser.add_argument('--fps', action='store_true', default=False, help='fps test')
    parser.add_argument('--onnx', action='store_true', default=False, help='onnx inference')
    parser.add_argument('--engine', action='store_true', default=False, help='tensorrt inference')
    parser.add_argument('--input_shape', type=tuple, default=(608, 608), help='input shape')
    parser.add_argument('--conf', type=float, default=0.5, help='confidence threshold')
    parser.add_argument('--nms', type=float, default=0.4, help='nms threshold')
    opt = parser.parse_args()
    LOGGER.info(opt)
    run(opt)
