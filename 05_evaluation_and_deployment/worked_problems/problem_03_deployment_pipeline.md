# Problem 03: Designing an ML Deployment Pipeline

**Topic:** End-to-end machine learning deployment architecture  
**Difficulty:** Fundamentals (Parts A-B), Intermediate (Parts C-D), Advanced (Parts E-F)  
**Prerequisites:** `onnx_and_inference_runtimes.md`, `quantisation_and_pruning.md`

---

## Problem Statement

Your company has trained a PyTorch image classification model (ResNet-50 fine-tuned on
product images) to be used in a real-time e-commerce recommendation system. The model
must:

- Classify product images into 500 categories.
- Serve predictions with p99 latency <= 50 ms on an NVIDIA T4 GPU.
- Handle 500 requests/second peak throughput.
- Support rolling model updates (new fine-tuned models every 2 weeks) with zero downtime.
- Allow rollback to the previous version within 5 minutes if the new model underperforms.

---

### Part A (Fundamentals)

What is the difference between a model artefact, a model serving endpoint, and a model
registry? How do these three components interact in a deployment pipeline?

---

### Part B (Fundamentals)

List the steps you would take to convert a trained PyTorch ResNet-50 model for optimised
inference on the T4 GPU. Specify the tools and formats involved at each step.

---

### Part C (Intermediate)

Design the real-time serving infrastructure. Your design should address:
(a) How HTTP requests are handled and batched.
(b) How to achieve the 500 req/s throughput target with a single T4 GPU.
(c) How to monitor latency and catch regressions.

---

### Part D (Intermediate)

Design the model update and rollback mechanism. Specifically:
(a) What constitutes a "safe" new model version before it receives live traffic?
(b) How would you implement zero-downtime deployment with rollback capability?
(c) What metrics trigger an automatic rollback?

---

### Part E (Advanced)

During a load test at 600 req/s (20 % above peak), p99 latency spikes to 280 ms.
Identify five potential root causes and for each, describe how to diagnose and remediate
it using profiling tools or architectural changes.

---

### Part F (Advanced)

The product team requests a feature: display a confidence score to customers
("85 % match"). The model's raw softmax outputs are poorly calibrated -- for samples
where the model outputs p=0.85, only 62 % are actually correct. Design a calibration
pipeline that integrates into the deployment workflow.

---

## Solutions

### Part A Solution

**Model artefact:**
The serialised, static output of a training job -- e.g., a PyTorch `.pt` checkpoint,
a TensorRT `.plan` file, or an ONNX `.onnx` file. An artefact is immutable after
creation. It encodes the model's weights, architecture, and preprocessing parameters
(normalisation statistics). Artefacts are stored in object storage (S3, GCS) with a
unique version identifier (e.g., a hash of the training job's configuration and data).

**Model registry:**
A metadata store that tracks model artefacts across their lifecycle:
- Maps model names and version numbers to artefact locations.
- Stores associated metadata: training data version, training metrics, evaluation
  scores, who approved the model, when it was deployed.
- Tracks status transitions: Staging -> Production -> Archived.
- Examples: MLflow Model Registry, AWS SageMaker Model Registry, Vertex AI Model Registry.

**Model serving endpoint:**
A running service that loads a model artefact and exposes it via an API (REST or gRPC).
The endpoint accepts feature vectors or raw inputs, runs inference, and returns predictions.
It is the live, stateful component that serves traffic.

**How they interact:**
```
Training job
    |
    v
Model artefact (S3/GCS: s3://models/resnet50/v23/model.plan)
    |
    v
Model registry (records: "resnet50-v23 -> s3://...plan, eval_acc=92.1%, status=Staging")
    |
    v [after approval gate]
    v
Deployment pipeline (pulls artefact from registry, loads into serving container)
    |
    v
Serving endpoint (receives HTTP, runs TensorRT inference, returns JSON)
```

The registry is the source of truth for "which artefact is deployed where." Rolling back
means updating the registry's active pointer and restarting the serving container with
the previous artefact.

---

### Part B Solution

**Conversion pipeline for optimised T4 inference:**

**Step 1: Export PyTorch model to ONNX**
```python
import torch
import torch.onnx

model = ResNet50Classifier(num_classes=500)
model.load_state_dict(torch.load("resnet50_v23.pt"))
model.eval()

dummy = torch.randn(1, 3, 224, 224)

torch.onnx.export(
    model, dummy, "resnet50_v23.onnx",
    opset_version=17,
    do_constant_folding=True,
    input_names=["image"],
    output_names=["logits"],
    dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}},
)
```

**Step 2: Validate ONNX export**
Run ONNX checker and compare outputs between PyTorch and ONNX Runtime on a test batch.
Assert max absolute error < 1e-5 for FP32.

**Step 3: Build TensorRT engine**
```python
import tensorrt as trt

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
builder = trt.Builder(TRT_LOGGER)
network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
parser = trt.OnnxParser(network, TRT_LOGGER)

with open("resnet50_v23.onnx", "rb") as f:
    parser.parse(f.read())

config = builder.create_builder_config()
config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 << 30)  # 2 GB workspace

# Enable FP16 (T4 supports FP16 Tensor Cores)
config.set_flag(trt.BuilderFlag.FP16)

# INT8 calibration (optional, further speedup)
# config.set_flag(trt.BuilderFlag.INT8)
# config.int8_calibrator = MyCalibrator(calibration_data)

# Dynamic shapes for variable batch size
profile = builder.create_optimization_profile()
profile.set_shape("image", (1,3,224,224), (16,3,224,224), (32,3,224,224))
config.add_optimization_profile(profile)

serialized_engine = builder.build_serialized_network(network, config)
with open("resnet50_v23_t4.plan", "wb") as f:
    f.write(serialized_engine)
```

**Step 4: Validate TensorRT engine**
Load the .plan file, run on the validation set, compare predictions to the PyTorch
model. Assert top-1 accuracy matches within 0.1 percentage points.

**Step 5: Benchmark latency and throughput**
Use `trtexec --loadEngine=resnet50_v23_t4.plan --batch=16` or a custom Python benchmark
to measure p50/p99 latency and max throughput. Confirm p99 < 50 ms for the target batch
size before proceeding.

---

### Part C Solution

**(a) HTTP request handling and batching:**

A direct request-per-inference pattern wastes GPU utilisation: a single image at batch
size 1 uses a tiny fraction of the T4's compute. Instead, use **dynamic batching**:

```
Client -> Load balancer -> [Triton Inference Server or TorchServe]
                                |
                          Batching queue
                          (collect requests up to:
                            max_batch=32 OR
                            max_wait=5ms)
                                |
                          TensorRT engine (batch inference)
                                |
                          Dispatch results back to waiting clients
```

NVIDIA Triton Inference Server natively supports dynamic batching with configurable
`preferred_batch_size` and `max_queue_delay_microseconds`.

**(b) Achieving 500 req/s on a single T4:**

The T4 has 65 TOPS INT8 / 8.1 TFLOPS FP16. ResNet-50 requires ~4 GFLOPs per image.

Rough throughput upper bound (FP16):
```
8.1 TFLOPS / 4 GFLOPS per image = ~2025 images/s (theoretical peak, no overhead)
```

In practice, memory bandwidth and kernel overhead reduce this to ~400-800 images/s for
batch size 16-32 with TensorRT FP16.

To achieve 500 req/s:
1. Use TensorRT FP16 (2x speedup over FP32).
2. Set dynamic batching to prefer batch=16.
3. Use concurrent CUDA streams so preprocessing (CPU) overlaps with inference (GPU).
4. If single T4 is insufficient, add a second T4 with load balancer (trivial horizontal
   scaling since inference is stateless).

**(c) Latency monitoring and regression detection:**

Instrument the serving layer to emit latency percentiles (p50, p95, p99, p999) per
endpoint and batch size. Use Prometheus + Grafana dashboards.

Alert conditions:
- p99 > 40 ms (warn) or > 50 ms (critical) over a 5-minute window.
- GPU utilisation > 95 % for > 2 minutes (saturation indicator).
- Request queue depth > 50 (indicates batching cannot keep up).

After every model update, run a load test (1.2x expected peak) before routing live traffic
and gate on: p99 < 50 ms AND throughput > 500 req/s.

---

### Part D Solution

**(a) What constitutes a "safe" new model version:**

Before receiving any live traffic, a new model version must pass:

1. **Correctness gate:** Evaluation on a held-out test set. Top-1 accuracy >= baseline
   model accuracy - 0.5 %. AUC-PR on a labelled validation set within tolerance.
2. **Latency gate:** TensorRT benchmarking confirms p99 < 50 ms at batch=16 on the T4.
3. **Numerical parity check:** Max output difference between the ONNX and TensorRT
   versions < 1e-3 on 1000 test images (ensures quantisation did not corrupt logic).
4. **Shadow mode evaluation:** Route 1 % of live traffic to the new model in "shadow mode"
   (run inference but discard results). Compare live distribution of prediction
   confidence scores to the baseline model to detect distribution shift.

**(b) Zero-downtime deployment with rollback:**

Use a **blue-green deployment** or **canary deployment**:

```
Blue-green:
  Current (blue): 100 % of traffic
  Deploy new (green): start containers, run health checks, latency benchmarks
  Switch: update load balancer weights to 0 % blue / 100 % green (instantaneous)
  Monitor for 15 minutes; if alert fires, switch back to 100 % blue (rollback in <1 min)

Canary:
  Ramp traffic to new model: 1% -> 5% -> 25% -> 100% over 30 minutes
  Monitor business metrics (CTR, conversion) and technical metrics (latency, error rate)
  at each step; auto-rollback if degradation detected
```

Rollback is achieved by updating the load balancer weights and pointing the active model
pointer in the registry back to the previous version. Both versions' TensorRT engines
are kept loaded in memory until the rollback window has passed.

**(c) Metrics that trigger automatic rollback:**

| Metric               | Rollback condition                                   |
|----------------------|-----------------------------------------------------|
| p99 latency          | > 50 ms for > 2 minutes on the new model            |
| Error rate           | > 0.1 % HTTP 5xx errors (inference failures)        |
| Prediction accuracy  | Online evaluation accuracy drops > 2 % vs baseline  |
| Score distribution   | KL-divergence of confidence score distributions > 0.1 |
| GPU memory           | OOM errors on the new model                         |

All checks run in a monitoring loop with 30-second evaluation windows. Rollback is
automated for infrastructure metrics (latency, errors); business metrics (accuracy, CTR)
may require a human approval step.

---

### Part E Solution

Five potential root causes of p99 = 280 ms at 600 req/s:

**1. Batching queue saturation**
*Diagnosis:* Queue depth metric > max_batch consistently. New requests wait in queue
longer than max_queue_delay.
*Remediation:* Add a second T4 GPU behind the load balancer. Alternatively, reduce
max_batch to flush the queue faster at the cost of lower GPU utilisation.

**2. Memory bandwidth bottleneck (not compute-bound)**
*Diagnosis:* Use `nvidia-smi dmon` or `nsys profile`: GPU compute utilisation < 70 %
but memory bandwidth utilisation is > 90 %. Small batch sizes (e.g., batch=1 during
burst) cause many small memory transactions.
*Remediation:* Ensure dynamic batching is aggregating requests effectively. Increase
`max_queue_delay_microseconds` to allow larger batches to accumulate during bursts.

**3. Preprocessing CPU bottleneck**
*Diagnosis:* GPU is idle waiting for preprocessed batches. CPU util = 100 %. The image
decode and resize pipeline (JPEG decode + resize to 224x224 + normalise) is the
bottleneck.
*Remediation:* Move preprocessing to GPU using DALI (NVIDIA Data Loading Library) or
torchvision GPU ops. Alternatively, add more CPU preprocessing threads.

**4. TensorRT engine not built for the observed batch sizes**
*Diagnosis:* At peak load, batch sizes exceed the `optShapes` used during engine build.
TensorRT falls back to dynamic kernel selection which is slower.
*Remediation:* Rebuild the TensorRT engine with the peak batch size included in the
optimization profile: `profile.set_shape("image", (1,...), (16,...), (64,...))`.

**5. TCP/HTTP overhead at high connection rate**
*Diagnosis:* Profiling shows most latency is in network I/O, not GPU compute. Many
short-lived TCP connections at 600 req/s cause kernel TCP stack overhead.
*Remediation:* Use HTTP/2 multiplexing (one TCP connection for many requests) or gRPC
(HTTP/2 with Protocol Buffers). Enable keep-alive on the load balancer. Alternatively,
place a lightweight L7 proxy (Envoy, nginx) in front of the inference server to handle
connection pooling.

---

### Part F Solution

**Calibration pipeline design:**

**Why calibration is needed:**
ResNet-50 trained with cross-entropy is typically overconfident on training-domain samples
and miscalibrated after fine-tuning on a small domain dataset. The raw softmax output
is not a reliable probability estimate.

**Calibration method: Temperature scaling** (simplest and most effective in practice)

Temperature scaling learns a single scalar parameter T that divides the logits before
softmax:
```
p_calibrated = softmax(logits / T)
```
T > 1 flattens the distribution (reduces confidence); T < 1 sharpens it.
T is found by minimising the NLL loss on a held-out calibration set (never used in
training):

```python
from torch.optim import LBFGS
import torch.nn.functional as F

T = torch.nn.Parameter(torch.ones(1))
optimiser = LBFGS([T], lr=0.01, max_iter=50)

def closure():
    optimiser.zero_grad()
    scaled_logits = logits_val / T
    loss = F.cross_entropy(scaled_logits, y_val)
    loss.backward()
    return loss

optimiser.step(closure)
print(f"Optimal temperature: {T.item():.3f}")
```

**Integration into deployment workflow:**

```
Training job:
  1. Fine-tune ResNet-50 -> produce logits on calibration set (10 % of labelled data)
  2. Fit temperature T on calibration set
  3. Package (model weights, T) as the artefact

Serving pipeline:
  4. Export PyTorch model to ONNX (include T scaling as a Div node before Softmax)
  5. Build TensorRT engine from calibrated ONNX graph
  6. At inference: raw logits / T -> Softmax -> calibrated probabilities

Monitoring:
  7. Log predicted confidence bins (0-10%, 10-20%, ..., 90-100%) and actual accuracy
     within each bin on a rolling window of scored-and-labeled transactions.
  8. Reliability diagram: plot predicted confidence vs actual accuracy.
     A well-calibrated model follows the diagonal.
  9. Alert if Expected Calibration Error (ECE) > 5 % (indicates distribution shift
     or the temperature T is no longer optimal for the live distribution).
  10. Re-calibrate T every 2 weeks alongside the model update cycle using recently
      logged and labelled predictions.
```

**Expected Calibration Error (ECE):**
```
ECE = sum over bins b: (|b| / N) * |accuracy(b) - confidence(b)|
```
A well-calibrated model has ECE < 2 %. If ECE > 5 %, display confidence scores to
customers with a caveat or re-run the calibration step before the next display.
