"""GPU stress-test & evaluation pipeline.

Two front-ends share the same runner, monitor, evaluator and reporter:

* ``gpu_stress.torch_pipeline`` - full PyTorch suite (ResNet50 training,
  torch.compile, mixed precision, ...).  Needs the ``torch`` extra.
* ``gpu_stress.lite_pipeline``  - zero-dependency suite that drives the CUDA
  driver API through ctypes with pre-built PTX kernels.  Needs only the
  NVIDIA driver.
"""

__version__ = "0.2.0"
