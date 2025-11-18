#pragma once

#include "tensor.hpp"

#include <cstddef>
#include <cstdint>
#include <iterator>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace smollnet {

struct DatasetBatch {
  DatasetBatch() : inputs(), targets() {}
  DatasetBatch(Tensor batch_inputs, Tensor batch_targets)
      : inputs(std::move(batch_inputs)), targets(std::move(batch_targets)) {}

  Tensor inputs;
  Tensor targets;
};

class TensorDataset {
public:
  TensorDataset() = default;
  TensorDataset(Tensor inputs, Tensor targets);

  size_t size() const noexcept;
  const Tensor &inputs() const noexcept;
  const Tensor &targets() const noexcept;

  DatasetBatch batch(size_t start, size_t count) const;
  DatasetBatch batch(const std::vector<size_t> &indices) const;

private:
  Tensor inputs_;
  Tensor targets_;
};

using TensorDatasetPtr = std::shared_ptr<TensorDataset>;

struct DataLoaderOptions {
  size_t batch_size = 1;
  bool shuffle = false;
  bool drop_last = false;
  uint32_t seed = 1234U;
};

class DataLoader {
public:
  class Iterator {
  public:
    using iterator_category = std::input_iterator_tag;
    using value_type = DatasetBatch;
    using difference_type = std::ptrdiff_t;
    using pointer = DatasetBatch *;
    using reference = DatasetBatch &;

    Iterator() = default;
    explicit Iterator(DataLoader *loader);

    reference operator*() noexcept;
    pointer operator->() noexcept;
    Iterator &operator++();
    bool operator==(const Iterator &other) const noexcept;
    bool operator!=(const Iterator &other) const noexcept;

  private:
    void load_next();

    DataLoader *loader_ = nullptr;
    DatasetBatch current_;
  };

  DataLoader(TensorDatasetPtr dataset, DataLoaderOptions options);
  DataLoader(TensorDataset dataset, DataLoaderOptions options);
  DataLoader(TensorDatasetPtr dataset, size_t batch_size, bool shuffle = false,
             bool drop_last = false, uint32_t seed = 1234U);
  DataLoader(TensorDataset dataset, size_t batch_size, bool shuffle = false,
             bool drop_last = false, uint32_t seed = 1234U);

  void reset();
  bool has_next() const noexcept;
  DatasetBatch next();
  Iterator begin();
  Iterator end() noexcept;

  size_t batch_size() const noexcept;
  size_t num_batches() const noexcept;
  size_t dataset_size() const noexcept;

private:
  void prepare_epoch();

  TensorDatasetPtr dataset_;
  DataLoaderOptions options_;
  std::vector<size_t> order_;
  size_t cursor_ = 0;
  uint64_t epoch_ = 0;
};

struct CSVLoaderOptions {
  bool has_header = true;
  char delimiter = ',';
  size_t target_columns = 1;
  DataType dtype = DataType::f32;
  Device device = Device::CPU;
  bool requires_grad = false;
};

TensorDatasetPtr load_csv_dataset(const std::string &path,
                                  const CSVLoaderOptions &options = {});

} // namespace smollnet
