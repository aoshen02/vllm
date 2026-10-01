// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <ATen/dlpack.h>

#include <atomic>
#include <cstdint>
#include <limits>
#include <new>

namespace {
std::atomic<long> live{0};
std::atomic<long> released{0};

struct Owner {
  DLManagedTensor managed{};
  int64_t shape[2];
  int64_t strides[2];
};

void release(DLManagedTensor* managed) {
  auto* owner = static_cast<Owner*>(managed->manager_ctx);
  delete owner;
  --live;
  ++released;
}

void capsule_release(PyObject* capsule) {
  if (PyCapsule_IsValid(capsule, "dltensor")) {
    auto* managed = static_cast<DLManagedTensor*>(
        PyCapsule_GetPointer(capsule, "dltensor"));
    release(managed);
  }
}

PyObject* make_capsule(PyObject*, PyObject* args) {
  unsigned long long pointer;
  long long rows, columns, row_stride;
  int device_type, device_id;
  if (!PyArg_ParseTuple(args, "KLLLii", &pointer, &rows, &columns, &row_stride,
                        &device_type, &device_id)) {
    return nullptr;
  }
  if (!pointer || rows <= 0 || columns <= 0 || row_stride < columns ||
      (device_type != kDLCPU && device_type != kDLCUDA) || device_id < 0 ||
      rows - 1 > (std::numeric_limits<int64_t>::max() - columns) / row_stride) {
    PyErr_SetString(PyExc_ValueError, "invalid uint8 strided workspace");
    return nullptr;
  }
  auto* owner = new (std::nothrow) Owner;
  if (!owner) return PyErr_NoMemory();
  owner->shape[0] = rows;
  owner->shape[1] = columns;
  owner->strides[0] = row_stride;
  owner->strides[1] = 1;
  auto& tensor = owner->managed.dl_tensor;
  tensor.data = reinterpret_cast<void*>(static_cast<uintptr_t>(pointer));
  tensor.device = {static_cast<DLDeviceType>(device_type), device_id};
  tensor.ndim = 2;
  tensor.dtype = {kDLUInt, 8, 1};
  tensor.shape = owner->shape;
  tensor.strides = owner->strides;
  tensor.byte_offset = 0;
  owner->managed.manager_ctx = owner;
  owner->managed.deleter = release;
  ++live;
  auto* capsule = PyCapsule_New(&owner->managed, "dltensor", capsule_release);
  if (!capsule) release(&owner->managed);
  return capsule;
}

PyObject* counts(PyObject*, PyObject*) {
  return Py_BuildValue("ll", live.load(), released.load());
}

PyMethodDef methods[] = {
    {"make_capsule", make_capsule, METH_VARARGS,
     "Wrap borrowed uint8 memory with native DLPack metadata ownership."},
    {"counts", counts, METH_NOARGS, "Live and released metadata owners."},
    {nullptr, nullptr, 0, nullptr}};
PyModuleDef module = {PyModuleDef_HEAD_INIT, "_mnnvl_C", nullptr, -1, methods};
}  // namespace

PyMODINIT_FUNC PyInit__mnnvl_C() { return PyModule_Create(&module); }
