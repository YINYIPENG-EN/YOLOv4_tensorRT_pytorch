```
git clone
```

# torch2onnx

改功能的实现在torch2onnx.py中。

你需要修改的地方有：

simplity:是否开启simplity模式

output_path: # 输出onnx路径

num_classes: # 类的数量

ckpt_path: # torch权重路径

input_shape: # 输入网络图像尺寸大小

```python
python torch2onnx.py
```

```
转换过程如下：
graph(%images : Float(1:1108992, 3:369664, 608:608, 608:1, requires_grad=0, device=cpu),
      %yolo_head3.1.weight : Float(255:256, 256:1, 1:1, 1:1, requires_grad=1, device=cpu),
      %yolo_head3.1.bias : Float(255:1, requires_grad=1, device=cpu),
      %yolo_head2.1.weight : Float(255:512, 512:1, ...
```

可以通过Netron工具对onnx可视化

------

# onnx2engine

改功能的实现在onnx2trt.py中。

你需要修改的地方有：

onnx_path: # onnx文件路径

engine_path: # 输出engine路径

```
python onnx2trt.py
```

生成engine的时间比较长，在我电脑上大概用了10分钟左右，如果用trtexec也可以生成engine，这个构建时间会快很多。

------

推理功能在predict.py中。

```
参数说明：
--weights: # 权重路径
--img: # 开启图像预测
--video: # 开启视频预测
--video_path: # 视频路径
--fps: # FPS测试
--onnx: # 开启onnx推理
--engine: # 开启trt预测
--input_shape: # 输入大小，默认608 * 608
--conf: # 置信度阈值
--nms: # NMS阈值
```



# torch推理

图像推理：

```
python predict.py --weights model_data/yolo4_weights.pth --img --conf 0.5 --nms 0.4
```

视频推理：

```
python predict.py --weights model_data/yolo4_weights.pth --video --video_path 0
```

FPS测试：

```
python predict.py --weights model_data/yolo4_weights.pth --fps
```

# onnx推理

图像推理：

```
python predict.py --weights model_data/yolov4.onnx --img --conf 0.5 --nms 0.4
```

视频推理：

```
python predict.py --weights model_data/yolov4.onnx --video --video_path 0
```

FPS测试：

```
python predict.py --weights model_data/yolov4.onnx --fps
```

# engine推理

图像推理：

```
python predict.py --weights model_data/yolov4.engine --img --conf 0.5 --nms 0.4
```

视频推理：

```
python predict.py --weights model_data/yolov4.engine --video --video_path 0
```

FPS测试：

```
python predict.py --weights model_data/yolov4.engine --fps
```

