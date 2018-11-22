import pytest
from tensorflow.python.client import device_lib


@pytest.fixture
def tensorflow_gpu_devices():
    return [d.name for d in device_lib.list_local_devices()
            if d.device_type == 'GPU']
