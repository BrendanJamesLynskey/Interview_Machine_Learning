# ONNX and Inference Runtimes

## Prerequisites
- Neural network training with PyTorch or TensorFlow
- Quantisation basics (`quantisation_and_pruning.md`)
- Familiarity with hardware accelerators (GPU, CPU SIMD, NPU)

---

## Concept Reference

### The Framework-to-Deployment Gap

Training frameworks (PyTorch, TensorFlow, JAX) are optimised for flexibility and
automatic differentiation during research and development. They carry substantial overhead:
Python interpreter, autograd graph construction, dynamic operator dispatch, and memory
profiling. In deployment, none of this overhead is needed -- you want the fastest possible
execution of a fixed graph on a specific piece of hardware.

Inference runtimes bridge this gap by:
1. Accepting a frozen model graph as input (no gradients, no dynamic graph construction).
2. Applying hardware-specific operator fusion, memory layout optimisation, and kernel
   selection.
3. Executing the optimised graph with minimal overhead on the target hardware.

---

### ONNX: Open Neural Network Exchange

**ONNX** (Open Neural Network Exchange) is an open standard that defines a common
intermediate representation (IR) for neural network models. It specifies:

- A **protobuf schema** describing the computational graph (nodes, edges, data types,
  tensor shapes).
- A **standard operator set** (opset) with versioned semantics covering common operations:
  Conv, MatMul, GEMM, Relu, Softmax, BatchNormalization, LSTM, Attention, etc.
- A versioned **opset** system: `opset_version=17` means the model uses operator semantics
  defined in ONNX opset 17.

#### Why ONNX Matters

The combination of multiple training frameworks and multiple deployment targets (CPU, GPU,
NPU, FPGA, mobile, browser) creates an N×M compatibility problem. ONNX solves this by
providing a single intermediate format:

```
PyTorch ----\                        /---- ONNX Runtime (CPU/GPU)
TensorFlow --+-- ONNX (IR) ----------+---- TensorRT (NVIDIA GPU)
JAX ---------/                       \---- OpenVINO (Intel)
                                      \--- CoreML (Apple)
                                       \-- TFLite (Mobile)
```

#### Exporting a PyTorch Model to ONNX

```python
import torch
import torch.onnx

model = MyModel()
model.eval()

dummy_input = torch.randn(1, 3, 224, 224)   # Batch size 1, 3-channel 224x224 image

torch.onnx.export(
    model,
    dummy_input,
    "model.onnx",
    export_params=True,           # Include trained weights in the file
    opset_version=17,             # Target ONNX opset version
    do_constant_folding=True,     # Fold constant subexpressions at export time
    input_names=["input"],        # Name the input tensor
    output_names=["output"],      # Name the output tensor
    dynamic_axes={                # Mark dimensions that can vary at runtime
        "input":  {0: "batch_size"},
        "output": {0: "batch_size"},
    }
)
```

**`do_constant_folding`:** Pre-computes subgraphs that depend only on constants (e.g.,
fixed positional embeddings, learned batch-norm statistics folded into conv weights).
Reduces the runtime graph size and improves execution speed.

**`dynamic_axes`:** Without this, the exported model is fixed to the exact input shape
used during export. Setting `dynamic_axes` marks dimensions (e.g., batch size, sequence
length) as variable, allowing the runtime to handle variable-size inputs.

#### ONNX Graph Inspection and Verification

```python
import onnx
import onnxruntime as ort
import numpy as np

# Verify the ONNX model is well-formed
model_proto = onnx.load("model.onnx")
onnx.checker.check_model(model_proto)

# Run inference with ONNX Runtime
session = ort.InferenceSession("model.onnx")
input_name = session.get_inputs()[0].name
outputs = session.run(None, {input_name: np.random.randn(1, 3, 224, 224).astype(np.float32)})
```

#### Common ONNX Export Issues

| Issue                          | Cause                                               | Fix                                                  |
|-------------------------------|-----------------------------------------------------|------------------------------------------------------|
| Unsupported op                 | Custom Python operation with no ONNX equivalent     | Implement a custom ONNX operator or refactor the op  |
| Shape mismatch at runtime      | Dynamic shapes not declared                         | Add `dynamic_axes` for variable dimensions           |
| Opset version incompatibility  | Runtime supports older opset than model uses        | Lower `opset_version` to runtime's maximum           |
| Numerical discrepancy          | PyTorch and runtime differ in op precision          | Compare outputs; tolerance of 1e-5 FP32 is typical  |
| Control flow not exported      | Python `if`/`for` traced with fixed dummy values    | Use `torch.jit.script` or ONNX `If`/`Loop` ops       |

---

### ONNX Runtime

**ONNX Runtime (ORT)** is Microsoft's open-source inference engine for ONNX models. It
is one of the most widely deployed inference runtimes and is used in production by Azure
ML, Hugging Face Transformers inference, and Windows ML.

#### Execution Providers

ORT uses **execution providers (EPs)** to dispatch operations to specific hardware:

| Execution Provider       | Hardware                        | Notes                                         |
|-------------------------|---------------------------------|-----------------------------------------------|
| CPUExecutionProvider    | CPU (any)                       | Default. Uses MLAS kernels (Microsoft's BLAS).|
| CUDAExecutionProvider   | NVIDIA GPU                      | cuBLAS, cuDNN. Requires CUDA toolkit.         |
| TensorrtExecutionProvider| NVIDIA GPU (TensorRT backend)  | Higher latency optimisation. See TensorRT.    |
| OpenVINOExecutionProvider| Intel CPU/GPU/VPU              | Uses OpenVINO under the hood.                 |
| CoreMLExecutionProvider | Apple CPU/GPU/Neural Engine     | macOS/iOS deployment.                         |
| DnnlExecutionProvider   | Intel CPU (oneDNN)              | Optimised Intel SIMD kernels.                 |

ORT selects the best EP for each node in the graph (graph partitioning), falling back to
CPU for unsupported ops.

#### Graph Optimisations in ORT

ORT applies a multi-level graph optimisation pipeline:

- **Level 1 (Basic):** Constant folding, redundant node elimination, shape inference.
- **Level 2 (Extended):** Operator fusion. Examples: Conv + BatchNorm folded into a
  single Conv (BN parameters absorbed into Conv weights). Attention = MatMul + Softmax +
  MatMul fused.
- **Level 3 (Layout optimisation):** NCHW to NHWC layout transformation for CPU kernels
  that prefer contiguous channel data.

---

### TensorRT

**TensorRT** is NVIDIA's deep learning inference SDK. It takes a trained model (via ONNX
or its own API) and produces a highly optimised engine for a specific NVIDIA GPU.

#### TensorRT Optimisation Pipeline

```
Input ONNX graph
    -> Layer fusion (Conv + BN + ReLU -> single fused kernel)
    -> Kernel auto-tuning (select fastest CUDA kernel from a library of implementations)
    -> Precision calibration (FP32 -> FP16 or INT8 with calibration data)
    -> Memory optimisation (in-place operations, tensor reuse)
    -> Engine serialisation (.plan file) for fast loading at inference time
```

**Key optimisations:**
- **Kernel auto-tuning:** For each operation (e.g., a specific convolution shape), TensorRT
  benchmarks multiple CUDA kernel implementations and selects the fastest for that exact
  GPU and input shape. This is why TensorRT build time can take minutes -- it runs
  benchmarks internally.
- **INT8 calibration:** Runs a calibration dataset through the FP32 network, collects
  activation histograms, and computes optimal quantisation scales per layer.
- **FP16 mixed precision:** FP16 tensor cores on Volta/Turing/Ampere provide 2-8x speedup
  for matrix operations with minimal accuracy loss.

#### TensorRT Build vs Runtime

```
Build phase (offline, once):
  - Parse ONNX
  - Apply optimisations and kernel selection
  - Serialize to .plan file (GPU-specific, non-portable)

Runtime phase (fast, repeated):
  - Load .plan file
  - Allocate GPU buffers
  - Execute engine
```

The `.plan` file is specific to the GPU architecture and TensorRT version. A plan built
on an A100 cannot run on a T4. Store plans alongside the GPU type that built them.

---

### OpenVINO

**OpenVINO** (Open Visual Inference and Neural network Optimisation) is Intel's inference
toolkit for deployment on Intel CPUs, integrated GPUs, VPUs (Myriad X), and FPGAs.

Key steps:
1. Convert ONNX (or TensorFlow/PaddlePaddle) model to OpenVINO IR (a `.xml` graph file
   and `.bin` weights file) using the **Model Optimizer**.
2. Load and execute with the **Inference Engine**.

OpenVINO applies similar optimisations to TensorRT (layer fusion, INT8 calibration) but
targets Intel hardware. The INT8 Post-Training Optimisation Toolkit (POT) uses the same
principles as TensorRT calibration.

---

### Inference Optimisation Strategies

Beyond choosing a runtime, additional techniques improve inference speed:

#### Operator Fusion
Merge consecutive operations into a single kernel. Eliminates intermediate memory reads
and writes. Most runtimes apply this automatically (e.g., Conv+BN+ReLU, LayerNorm,
MultiHeadAttention).

#### Batching
Process multiple inputs simultaneously. GPU utilisation is typically low for batch size 1;
larger batches amortise kernel launch overhead and fully utilise SIMD/tensor core
parallelism. Throughput scales roughly linearly with batch size until memory bandwidth or
compute is saturated.

#### Asynchronous Execution and Pipelining
Overlap data transfer (CPU-to-GPU) with GPU execution. Use CUDA streams or TensorRT
async APIs to pipeline preprocessing, inference, and postprocessing.

#### Memory Layout Optimisation
Ensure contiguous memory access patterns. For convolutions, NHWC (batch, height, width,
channels) is often faster than NCHW on CPUs due to better cache locality for the channel
dimension. For NVIDIA cuDNN, NCHW is often preferred. Most runtimes perform layout
transformation automatically.

---

## Interview Questions by Difficulty

### Fundamentals

**Q1.** What is ONNX and why was it created? What problem does it solve?

**Answer:**

ONNX is an open standard intermediate representation for neural network models. It was
created to solve the N×M compatibility problem between training frameworks (PyTorch,
TensorFlow, JAX, MXNet) and deployment runtimes and hardware targets (CPU, NVIDIA GPU,
Intel CPU, mobile NPU, FPGA, browser).

Without ONNX, each framework-to-hardware combination requires a separate converter or
exporter, creating a combinatorial maintenance burden. With ONNX, any framework that
can export to ONNX can be deployed on any runtime that supports ONNX, reducing the
problem to N exporters + M importers instead of N*M bespoke converters.

---

**Q2.** What does `do_constant_folding=True` do during ONNX export?

**Answer:**

Constant folding pre-computes subgraphs of the model that depend only on constant values
(i.e., learned weights and fixed hyperparameters, with no dependence on the runtime input).
For example, in a Transformer, learned positional embeddings and the folding of batch
normalisation parameters (scale and bias) into the preceding convolution's weights and
biases are constant-foldable.

The result is a smaller ONNX graph with fewer nodes, faster load times, and reduced
runtime computation because the constant subgraphs are evaluated once at export time
rather than on every inference call.

---

### Intermediate

**Q3.** Your team exports a Transformer model to ONNX and finds inference is 3x slower
in ONNX Runtime than in PyTorch with CUDA. What are the most likely causes and how
would you investigate?

**Answer:**

Most likely causes:

1. **TorchScript/CUDA graph optimisations in PyTorch not replicated in ORT.** PyTorch
   2.0 `torch.compile` and CUDA graphs can dramatically accelerate inference in ways ORT
   does not automatically replicate. Ensure you compare ORT against `model.eval()` without
   `torch.compile`, not against a compiled model.

2. **No CUDAExecutionProvider configured.** ORT defaults to CPU. Confirm CUDA EP is
   active: `ort.InferenceSession("model.onnx", providers=["CUDAExecutionProvider"])`.

3. **Attention is not fused.** Transformer attention (Q*K^T / sqrt(d) -> softmax -> *V)
   may export as many individual ONNX nodes rather than a fused multi-head attention op.
   Check whether ORT's `BertFusedAttention` or Flash Attention kernel is being used via
   `ort.SessionOptions` graph optimisation level.

4. **FP32 vs FP16 mismatch.** PyTorch may be running in FP16 (AMP) while ORT is running
   in FP32. Verify data types.

**Investigation steps:**
- Profile with ORT's execution profiling: `session_options.enable_profiling = True`.
- Compare per-op timing to identify the slow nodes.
- Try TensorrtExecutionProvider as a drop-in replacement for better GPU utilisation.

---

### Advanced

**Q4.** Describe TensorRT's engine build process. Why is a TensorRT plan file not
portable across GPU generations, and what are the implications for a CI/CD pipeline?

**Answer:**

**Build process:**
TensorRT takes an ONNX graph and performs:
1. **Graph parsing and validation:** Converts ONNX ops to TensorRT layer types.
2. **Layer fusion:** Combines consecutive compatible ops into a single fused layer
   (e.g., Conv+BN+ReLU, multi-head attention patterns).
3. **Kernel auto-tuning:** For each unique layer configuration (kernel size, tensor shape,
   data type), TensorRT benchmarks a library of CUDA kernel implementations and records
   the fastest for this specific GPU. This is the slow step: it can take 5-30 minutes.
4. **Precision calibration:** If INT8 is selected, TensorRT runs calibration data through
   the FP32 graph and computes per-layer quantisation scales using KL-divergence or
   percentile calibration.
5. **Serialisation:** The optimised plan (a binary blob) is written to a `.plan` file.

**Why .plan files are non-portable:**
The kernel selection step hardcodes the fastest CUDA kernel for the exact GPU SM
architecture, the exact CUDA and cuDNN versions, and the exact TensorRT version. A kernel
that is fastest on an A100 (sm80) may not even exist on a T4 (sm75). Running a
sm80-compiled plan on a sm75 GPU will produce incorrect results or fail at load time.

**CI/CD implications:**
1. **Build plans as part of deployment, not as build artefacts.** The plan must be built
   on the same GPU class that will run inference.
2. **Cache plans per (model-hash, GPU-type, TensorRT-version) tuple.** Rebuilding is
   expensive; caching avoids repeated builds for the same combination.
3. **Validate numerics after every build.** Compare TensorRT outputs to a reference
   (ORT CPU) on a validation set and assert max absolute error is below a tolerance.
4. **Version-pin the TensorRT and CUDA versions** in the deployment container to ensure
   plan compatibility.
