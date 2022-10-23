import torch
from nets.yolo import YoloBody
from onnxsim import simplify
import onnx

Simplity = False
output_path = 'model_data/yolov4.onnx'
num_classes = 80
anchors_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
ckpt_path = 'model_data/yolo4_weights.pth'
input_shape = (608, 608)
model = YoloBody(anchors_mask, num_classes)
device = torch.device('cpu')
ckpt = torch.load(ckpt_path, map_location=device)
model.load_state_dict(ckpt)
model.to(device)
test_input = torch.randn(1, 3, *input_shape).to(device)
input_name = ['images']
output_name = ['output0', 'outpupt1', 'output2']
torch.onnx.export(model, test_input, output_path,verbose=True,opset_version=12,input_names=input_name,
                  output_names=output_name)
if Simplity:
    onnx_model = onnx.load(output_path)  # load onnx model
    model_simp, check = simplify(onnx_model)
    assert check, "Simplified ONNX model could not be validated"
    onnx.save(model_simp, output_path)
    print('finished exporting onnx')
