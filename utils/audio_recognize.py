from playsound import playsound
import time
import os

'''
开启识别语音
'''


def Audio_process(infor_dict):
    import speech
    print('\033[1;31m Audio process ready~\033[0m')
    print('Audio process pid :', os.getpid())
    while (infor_dict['detection_procss_ready'] == False):
        time.sleep(2)

    while True:
        if infor_dict['res_class'] == True:
            # print("res_class :", infor_dict['res_class'])
            playsound("./materials/audio/ObjectMayBeIdentified.mp3")
            speech.say(infor_dict['class'])
        infor_dict['res_class'] = False  # 每次语音播报后进行状态的重置
        if infor_dict['break']:
            print('\033[1;31m 语音报警进程准备销毁!\033[0m')
            break
