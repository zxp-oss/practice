import tensorrt as trt

G_LOGGER = trt.Logger()

batch_size = 1
input_h = 640
input_w = 640


def get_engine(onnx_model_name, trt_model_name):
    explicit_batch = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    with trt.Builder(G_LOGGER) as builder, \
            builder.create_network(explicit_batch) as network, \
            trt.OnnxParser(network, G_LOGGER) as parser, \
            builder.create_builder_config() as config:

        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 << 30)
        print('Loading ONNX file from path {}...'.format(onnx_model_name))

        with open(onnx_model_name, 'rb') as model:
            print('Beginning ONNX file parsing')
            if not parser.parse(model.read()):
                for error in range(parser.num_errors):
                    print(parser.get_error(error))

        print('Completed parsing of ONNX file')
        print('Building an engine from file {}; this may take a while...'.format(onnx_model_name))

        # Enable FP16 mode if possible
        config.set_flag(trt.BuilderFlag.FP16)

        print("Number of layers:", network.num_layers)

        # Set input shape
        network.get_input(0).shape = [batch_size, 3, input_h, input_w]

        # Build serialized engine
        serialized_engine = builder.build_serialized_network(network, config)
        if serialized_engine is None:
            print("Failed to build the engine!")
            return None

        print("Completed creating Engine")

        # Deserialize engine to CUDA engine
        runtime = trt.Runtime(G_LOGGER)
        engine = runtime.deserialize_cuda_engine(serialized_engine)
        if engine is None:
            print("Failed to deserialize the engine!")
            return None

        # Save engine to file
        with open(trt_model_name, "wb") as f:
            f.write(serialized_engine)

        return engine


def main():
    onnx_file_path = './v8m-obb-best.onnx'
    engine_file_path = './yolov8n-obb.trt'
    engine = get_engine(onnx_file_path, engine_file_path)


if __name__ == '__main__':
    print("This is main ...")
    main()
