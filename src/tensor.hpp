#include <stdint.h>
#include <stddef.h>

namespace smollnet {

// Temp aliases
using Device  = int32_t;
using DataType  = int32_t;
using Layout  = int32_t;
using AutogradMeta  = int32_t;

struct Storage {
  void *ptr;
  size_t bytes;
  Device device;
  int refcount;
  ~Storage();
};

struct TensorImpl {
  Storage *storage;
  int64_t *sizes;
  int64_t *strides;
  int64_t ndim;
  DataType dtype;
  Layout layout;
  Device device;
  bool requires_grad;
  AutogradMeta *grad;
};

class Tensor {
  TensorImpl *p_;

public:
  TensorImpl *impl() const noexcept { return p_; }
  int64_t size(int d) const noexcept { return p_->sizes[d]; }
  void *data() const noexcept { return static_cast<char *>(p_->storage->ptr); }
};

} // namespace smollnet
