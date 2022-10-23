import tensorrt as trt


def build_engine(onnx_file_path,engine_file_path,half=False):
    """Takes an ONNX file and creates a TensorRT engine to run inference with"""
    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    config = builder.create_builder_config()
    config.max_workspace_size = 4 * 1 << 30#这决定了可用的内存量
    flag = (1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    network = builder.create_network(flag)
    parser = trt.OnnxParser(network, logger)
    if not parser.parse_from_file(str(onnx_file_path)):
        raise RuntimeError(f'failed to load ONNX file: {onnx_file_path}')
    half &= builder.platform_has_fast_fp16
    if half:
        config.set_flag(trt.BuilderFlag.FP16)
    with builder.build_engine(network, config) as engine, open(engine_file_path, 'wb') as t:
        t.write(engine.serialize())
    return engine_file_path


if __name__ =="__main__":
    onnx_path = 'model_data/yolov4.onnx'
    engine_path = 'model_data/yolov4.engine'
    build_engine(onnx_path, engine_path,True)
