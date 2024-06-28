import pycuda.driver as cuda
import pycuda.autoinit
import tensorrt as trt
import os

# Classes and function to run YOLO
class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()


def allocate_buffers(engine):
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()
    for binding in engine:
        size = trt.volume(engine.get_binding_shape(binding)[1:]) * engine.max_batch_size
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        binding_index = engine.get_binding_index(binding)

        # Allocate host and device buffers
        host_mem = cuda.pagelocked_empty(size, dtype)
        # Only bytes, no need for size
        device_mem = cuda.mem_alloc(host_mem.nbytes)

        # Append the device buffer to device bindings.
        bindings.append(int(device_mem))

        # Append to the appropriate list.
        if engine.binding_is_input(binding):
            inputs.append(HostDeviceMem(host_mem, device_mem))
        else:
            outputs.append(HostDeviceMem(host_mem, device_mem))
    return inputs, outputs, bindings, stream

def load_engine(engine_file_path, trt_logger):
  assert os.path.exists(engine_file_path)
  print("Reading engine from file {}".format(engine_file_path))
  trt.init_libnvinfer_plugins(trt_logger, "")
  with open(engine_file_path, "rb") as f, trt.Runtime(trt_logger) as runtime:
    serialized_engine = f.read()
    engine = runtime.deserialize_cuda_engine(serialized_engine)
    return engine
