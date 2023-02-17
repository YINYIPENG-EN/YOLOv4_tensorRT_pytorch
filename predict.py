import time
import cv2
from yolo import YOLO, LOGGER, np, Image
import argparse
from multiprocessing import Manager, Process
from utils.audio_recognize import Audio_process
from utils.object_track_fuction import object_track_pipeline

def run(opt, infor_dict, img=None):
    weights, predict, video, video_path, fps, ONNX, TRT, input_shape, conf, nms, audio, audio_class = opt.weights, opt.img, opt.video, \
                                    opt.video_path, opt.fps, opt.onnx, opt.engine, opt.input_shape, opt.conf, opt.nms, opt.audio, opt.audio_class
    yolo = YOLO(weights, input_shape, conf, nms, ONNX, TRT, audio, audio_class)
    video_fps = 25.0
    video_save_path = "results.mp4"
    test_interval = 100

    if type(eval(video_path)) == int:
        video_path = eval(video_path)
    if predict:
            try:
                # image = Image.open(img) # 采用Image用trt推理的时候精度下降非常厉害。具体原因不明
                image = cv2.imread(img)
            except:
                LOGGER.info('Open Error! Try again!')
            else:
                r_image, _, _ = yolo.detect_image(image, infor_dict)
                r_image.show()
                r_image.save('results.jpg')  # jpg

    elif video:
        target_dict = {}
        track_index = 0
        capture = cv2.VideoCapture(video_path)
        if video_save_path != "":
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            size = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            out = cv2.VideoWriter(video_save_path, fourcc, video_fps, size)

        ref, frame = capture.read()
        if not ref:
            raise ValueError("未能正确读取摄像头（视频），请注意是否正确安装摄像头（是否正确填写视频路径）。")

        fps = 0.0
        print('\033[1;31m Yolov4 process ready~\033[0m')
        print("\033[1;32m Press ESC to exit the program\033[0m")
        infor_dict['detection_procss_ready'] = True
        while (True):
            t1 = time.time()
            ref, frame = capture.read()
            if not ref:
                break
            frame, target_dict, track_index = yolo.detect_image(frame, infor_dict, target_dict, track_index)  # detect_image返回的是Image
            frame = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)
            frame = object_track_pipeline(frame, target_dict)
            fps = (fps + (1. / (time.time() - t1))) / 2
            # print("fps= %.2f" % (fps))
            frame = cv2.putText(frame, "fps= %.2f" % (fps), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow("video", frame)
            c = cv2.waitKey(1) & 0xff
            if video_save_path != "":
                out.write(frame)

            if c == 27:
                capture.release()
                infor_dict['break'] = True
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
        LOGGER.info(str(int(tact_time * 1000)) + ' ms, ' + str(int(1 / tact_time)) + 'FPS, @batch_size 1')


if __name__ == "__main__":
    parser = argparse.ArgumentParser("TRT YOLOv4 ")
    parser.add_argument('--weights', type=str, default='model_data/yolov4.engine', help='weight path')
    parser.add_argument('--img', action='store_true', default=False, help='image predict')
    parser.add_argument('--video', action='store_true', default=True, help='video predict')
    parser.add_argument('--video_path', type=str, default='0', help='video path')
    parser.add_argument('--fps', action='store_true', default=False, help='fps test')
    parser.add_argument('--onnx', action='store_true', default=False, help='onnx inference')
    parser.add_argument('--engine', action='store_true', default=True, help='tensorrt inference')
    parser.add_argument('--input_shape', type=tuple, default=(608, 608), help='input shape')
    parser.add_argument('--conf', type=float, default=0.5, help='confidence threshold')
    parser.add_argument('--nms', type=float, default=0.4, help='nms threshold')

    # 增添语音识别
    parser.add_argument('--audio', action='store_true', default=False, help='audio detection')
    parser.add_argument('--audio_class', type=str, default='cell phone', help='audio recognize class')

    # 添加目标跟踪
    parser.add_argument('--track', action='store_true', default=True, help='object track')
    parser.add_argument('--track_class', type=str, default='person', help='track class')
    opt = parser.parse_args()


    LOGGER.info(opt)

    time.sleep(0.2)
    if opt.img:
        img = input("Input image filename:")
    else:
        img = None
    # 建立共享字典记录检测状态
    infor_dict = Manager().dict()
    infor_dict['detection_procss_ready'] = False  # 进程间开启同步信号
    infor_dict['res_class'] = False
    infor_dict['track'] = opt.track
    infor_dict['track_class'] = opt.track_class
    infor_dict['break'] = False  # 进程间退出信号

    process_list = []
    # 目标检测进程
    t = Process(target=run, args=(opt, infor_dict, img), name='object detection')
    process_list.append(t)
    # 是否开启audio功能
    if opt.audio:
        t = Process(target=Audio_process, args=(infor_dict, ), name='audio_recognize')
        process_list.append(t)

    for key_, val in infor_dict.items():
        print("- >", key_, ' ', val)

    for i in range(len(process_list)):
        process_list[i].start()
    for i in range(len(process_list)):
        process_list[i].join()

    del process_list