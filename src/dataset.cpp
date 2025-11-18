#include "dataset.hpp"

#include "dtype_utils.hpp"
#include "helpers.hpp"

#include <algorithm>
#include <cctype>
#include <cstring>
#include <exception>
#include <fstream>
#include <iterator>
#include <memory>
#include <numeric>
#include <random>
#include <set>
#include <sstream>
#include <string_view>
#include <unordered_map>
#include <utility>

#include <cuda_runtime.h>
#include <fmt/core.h>

namespace smollnet {
namespace {

struct CSVRow {
  size_t line_number = 0;
  std::vector<std::string> cells;
};

struct CategoricalFeature {
  size_t column = 0;
  std::vector<std::string> categories;
  std::unordered_map<std::string, size_t> index_by_category;
};

constexpr size_t kNoCategoricalFeature = static_cast<size_t>(-1);

DataLoaderOptions make_loader_options(size_t batch_size, bool shuffle,
                                      bool drop_last, uint32_t seed) {
  DataLoaderOptions options;
  options.batch_size = batch_size;
  options.shuffle = shuffle;
  options.drop_last = drop_last;
  options.seed = seed;
  return options;
}

bool is_dense_contiguous(const Tensor &tensor) {
  int64_t expected_stride = 1;
  for (int64_t dim = tensor.ndims(); dim-- > 0;) {
    if (tensor.strides()[dim] != expected_stride) {
      return false;
    }
    expected_stride *= tensor.size(dim);
  }
  return true;
}

size_t sample_elements(const Tensor &tensor) {
  ASSERT(tensor.ndims() >= 1, "Dataset tensors must have a batch dimension");
  ASSERT(tensor.size(0) > 0, "Dataset tensors must not be empty");
  return tensor.numel() / static_cast<size_t>(tensor.size(0));
}

void validate_dataset_tensor(const Tensor &tensor, const char *name) {
  ASSERT(tensor.initialized(), fmt::format("{} tensor is uninitialized", name));
  ASSERT(tensor.ndims() >= 1,
         fmt::format("{} tensor must have at least one dimension", name));
  ASSERT(tensor.size(0) > 0,
         fmt::format("{} tensor must have at least one sample", name));
  ASSERT(is_dense_contiguous(tensor),
         fmt::format("{} tensor must be dense and contiguous", name));
}

Tensor copy_rows(const Tensor &source, const std::vector<size_t> &indices) {
  ASSERT(!indices.empty(), "Cannot create an empty dataset batch");

  TensorShape batch_dims = source.dims();
  batch_dims[0] = static_cast<int64_t>(indices.size());

  Tensor output = empty(batch_dims.data(), source.ndims(), source.dtype(),
                        source.device(), source.requires_grad());
  const size_t row_bytes = sample_elements(source) * element_size(source.dtype());
  const auto *src = static_cast<const char *>(source.data());
  auto *dst = static_cast<char *>(output.data());

  for (size_t out_row = 0; out_row < indices.size(); ++out_row) {
    ASSERT(indices[out_row] < static_cast<size_t>(source.size(0)),
           fmt::format("Dataset index {} out of range for {} samples",
                       indices[out_row], source.size(0)));

    const size_t src_offset = indices[out_row] * row_bytes;
    const size_t dst_offset = out_row * row_bytes;

    if (source.device() == Device::CUDA) {
      CHECK_CUDA(cudaMemcpy(dst + dst_offset, src + src_offset, row_bytes,
                            cudaMemcpyDeviceToDevice));
    } else {
      std::memcpy(dst + dst_offset, src + src_offset, row_bytes);
    }
  }

  return output;
}

bool is_blank(std::string_view value) {
  return std::all_of(value.begin(), value.end(), [](unsigned char ch) {
    return std::isspace(ch) != 0;
  });
}

std::string trim(std::string_view value) {
  size_t begin = 0;
  while (begin < value.size() &&
         std::isspace(static_cast<unsigned char>(value[begin])) != 0) {
    ++begin;
  }

  size_t end = value.size();
  while (end > begin &&
         std::isspace(static_cast<unsigned char>(value[end - 1])) != 0) {
    --end;
  }

  return std::string(value.substr(begin, end - begin));
}

std::vector<std::string> parse_csv_cells(std::string_view line, char delimiter,
                                         size_t line_number) {
  std::vector<std::string> cells;
  std::stringstream stream{std::string(line)};
  std::string cell;

  while (std::getline(stream, cell, delimiter)) {
    const std::string cleaned = trim(cell);
    ASSERT(!cleaned.empty(),
           fmt::format("Empty CSV value at line {}", line_number));
    cells.push_back(cleaned);
  }

  ASSERT(!cells.empty(), fmt::format("Empty CSV row at line {}", line_number));
  return cells;
}

float parse_float(std::string_view value, size_t line_number, size_t column) {
  const std::string text(value);
  size_t parsed_chars = 0;

  try {
    const float parsed = std::stof(text, &parsed_chars);
    ASSERT(parsed_chars == text.size(),
           fmt::format("Invalid float value '{}' at line {}, column {}", text,
                       line_number, column + 1));
    return parsed;
  } catch (const std::exception &) {
    ASSERT(false,
           fmt::format("Invalid float value '{}' at line {}, column {}", text,
                       line_number, column + 1));
  }

  return 0.0f;
}

std::vector<size_t>
validate_categorical_columns(const CSVLoaderOptions &options,
                             size_t feature_columns) {
  std::vector<size_t> categorical_columns = options.categorical_columns;
  std::sort(categorical_columns.begin(), categorical_columns.end());

  for (size_t idx = 0; idx < categorical_columns.size(); ++idx) {
    const size_t column = categorical_columns[idx];
    ASSERT(column < feature_columns,
           fmt::format("Categorical column {} must be an input feature column "
                       "before the last {} target column(s)",
                       column, options.target_columns));
    ASSERT(idx == 0 || categorical_columns[idx - 1] != column,
           fmt::format("Duplicate categorical column {}", column));
  }

  return categorical_columns;
}

std::vector<CategoricalFeature>
collect_categorical_features(const std::vector<CSVRow> &rows,
                             const std::vector<size_t> &columns) {
  std::vector<CategoricalFeature> features;
  features.reserve(columns.size());

  for (size_t column : columns) {
    std::set<std::string> unique_categories;
    for (const auto &row : rows) {
      unique_categories.insert(row.cells[column]);
    }

    CategoricalFeature feature;
    feature.column = column;
    feature.categories.assign(unique_categories.begin(),
                              unique_categories.end());
    for (size_t idx = 0; idx < feature.categories.size(); ++idx) {
      feature.index_by_category.emplace(feature.categories[idx], idx);
    }

    features.push_back(std::move(feature));
  }

  return features;
}

std::vector<size_t>
make_categorical_lookup(size_t feature_columns,
                        const std::vector<CategoricalFeature> &features) {
  std::vector<size_t> lookup(feature_columns, kNoCategoricalFeature);

  for (size_t idx = 0; idx < features.size(); ++idx) {
    lookup[features[idx].column] = idx;
  }

  return lookup;
}

size_t
expanded_feature_columns(size_t feature_columns,
                         const std::vector<CategoricalFeature> &features) {
  size_t expanded = feature_columns;
  for (const auto &feature : features) {
    expanded += feature.categories.size() - 1;
  }
  return expanded;
}

void append_encoded_feature(std::vector<float> &input_values,
                            const CSVRow &row, size_t column,
                            const std::vector<CategoricalFeature> &features,
                            size_t feature_index) {
  if (feature_index == kNoCategoricalFeature) {
    input_values.push_back(parse_float(row.cells[column], row.line_number,
                                       column));
    return;
  }

  const auto &feature = features[feature_index];
  const auto category = feature.index_by_category.find(row.cells[column]);
  ASSERT(category != feature.index_by_category.end(),
         fmt::format("Unknown category '{}' at line {}, column {}",
                     row.cells[column], row.line_number, column + 1));

  for (size_t idx = 0; idx < feature.categories.size(); ++idx) {
    input_values.push_back(idx == category->second ? 1.0f : 0.0f);
  }
}

Tensor tensor_from_values(const std::vector<float> &values, int64_t rows,
                          int64_t columns, DataType dtype, Device device,
                          bool requires_grad) {
  const int64_t dims[2] = {rows, columns};
  Tensor cpu_tensor = empty(dims, 2, dtype, Device::CPU, requires_grad);

  for (size_t idx = 0; idx < values.size(); ++idx) {
    store_scalar(cpu_tensor.data(), cpu_tensor.dtype(), idx, values[idx]);
  }

  if (device == Device::CUDA) {
    return cpu_tensor.cuda();
  }

  return cpu_tensor;
}

} // namespace

TensorDataset::TensorDataset(Tensor inputs, Tensor targets)
    : inputs_(std::move(inputs)), targets_(std::move(targets)) {
  validate_dataset_tensor(inputs_, "inputs");
  validate_dataset_tensor(targets_, "targets");
  ASSERT(inputs_.size(0) == targets_.size(0),
         fmt::format("Input/target sample mismatch: {} vs {}", inputs_.size(0),
                     targets_.size(0)));
  ASSERT(inputs_.device() == targets_.device(),
         fmt::format("Input/target device mismatch: {} vs {}",
                     get_device_name(inputs_.device()),
                     get_device_name(targets_.device())));
}

size_t TensorDataset::size() const noexcept {
  return inputs_.initialized() ? static_cast<size_t>(inputs_.size(0)) : 0;
}

const Tensor &TensorDataset::inputs() const noexcept { return inputs_; }

const Tensor &TensorDataset::targets() const noexcept { return targets_; }

DatasetBatch TensorDataset::batch(size_t start, size_t count) const {
  ASSERT(start <= size(), fmt::format("Batch start {} exceeds dataset size {}",
                                      start, size()));
  ASSERT(count <= size() - start,
         fmt::format("Batch [{}:{}) exceeds dataset size {}", start,
                     start + count, size()));

  std::vector<size_t> indices(count);
  std::iota(indices.begin(), indices.end(), start);
  return batch(indices);
}

DatasetBatch TensorDataset::batch(const std::vector<size_t> &indices) const {
  return {copy_rows(inputs_, indices), copy_rows(targets_, indices)};
}

DataLoader::Iterator::Iterator(DataLoader *loader) : loader_(loader) {
  load_next();
}

DataLoader::Iterator::reference DataLoader::Iterator::operator*() noexcept {
  return current_;
}

DataLoader::Iterator::pointer DataLoader::Iterator::operator->() noexcept {
  return &current_;
}

DataLoader::Iterator &DataLoader::Iterator::operator++() {
  load_next();
  return *this;
}

bool DataLoader::Iterator::operator==(const Iterator &other) const noexcept {
  return loader_ == other.loader_;
}

bool DataLoader::Iterator::operator!=(const Iterator &other) const noexcept {
  return !(*this == other);
}

void DataLoader::Iterator::load_next() {
  if (loader_ == nullptr || !loader_->has_next()) {
    loader_ = nullptr;
    current_ = DatasetBatch{};
    return;
  }

  current_ = loader_->next();
}

DataLoader::DataLoader(TensorDatasetPtr dataset, DataLoaderOptions options)
    : dataset_(std::move(dataset)), options_(options) {
  ASSERT(dataset_, "DataLoader dataset must not be null");
  ASSERT(options_.batch_size > 0,
         "DataLoader batch_size must be greater than zero");
  prepare_epoch();
}

DataLoader::DataLoader(TensorDataset dataset, DataLoaderOptions options)
    : DataLoader(std::make_shared<TensorDataset>(std::move(dataset)),
                 options) {}

DataLoader::DataLoader(TensorDatasetPtr dataset, size_t batch_size,
                       bool shuffle, bool drop_last, uint32_t seed)
    : DataLoader(std::move(dataset),
                 make_loader_options(batch_size, shuffle, drop_last, seed)) {}

DataLoader::DataLoader(TensorDataset dataset, size_t batch_size, bool shuffle,
                       bool drop_last, uint32_t seed)
    : DataLoader(std::make_shared<TensorDataset>(std::move(dataset)),
                 make_loader_options(batch_size, shuffle, drop_last, seed)) {}

void DataLoader::prepare_epoch() {
  order_.resize(dataset_->size());
  std::iota(order_.begin(), order_.end(), size_t{0});

  if (options_.shuffle) {
    std::mt19937 generator(options_.seed + static_cast<uint32_t>(epoch_));
    std::shuffle(order_.begin(), order_.end(), generator);
  }

  cursor_ = 0;
}

void DataLoader::reset() {
  ++epoch_;
  prepare_epoch();
}

bool DataLoader::has_next() const noexcept {
  const size_t remaining = dataset_->size() - cursor_;
  if (options_.drop_last) {
    return remaining >= options_.batch_size;
  }
  return remaining > 0;
}

DatasetBatch DataLoader::next() {
  ASSERT(has_next(), "DataLoader has no remaining batches");

  const size_t remaining = dataset_->size() - cursor_;
  const size_t current_batch_size = std::min(options_.batch_size, remaining);
  std::vector<size_t> indices(order_.begin() + static_cast<int64_t>(cursor_),
                              order_.begin() +
                                  static_cast<int64_t>(cursor_ +
                                                       current_batch_size));

  cursor_ += current_batch_size;
  return dataset_->batch(indices);
}

DataLoader::Iterator DataLoader::begin() {
  if (cursor_ != 0) {
    reset();
  }
  return Iterator(this);
}

DataLoader::Iterator DataLoader::end() noexcept { return Iterator(); }

size_t DataLoader::batch_size() const noexcept { return options_.batch_size; }

size_t DataLoader::num_batches() const noexcept {
  if (options_.drop_last) {
    return dataset_->size() / options_.batch_size;
  }
  return (dataset_->size() + options_.batch_size - 1) / options_.batch_size;
}

size_t DataLoader::dataset_size() const noexcept { return dataset_->size(); }

TensorDatasetPtr load_csv_dataset(const std::string &path,
                                  const CSVLoaderOptions &options) {
  ASSERT(options.target_columns > 0,
         "CSVLoaderOptions::target_columns must be greater than zero");

  std::ifstream input(path);
  ASSERT(input.is_open(), fmt::format("Unable to open CSV dataset '{}'", path));

  std::vector<float> input_values;
  std::vector<float> target_values;
  std::vector<CSVRow> csv_rows;
  size_t feature_columns = 0;
  size_t total_columns = 0;
  std::string line;
  size_t line_number = 0;

  if (options.has_header && std::getline(input, line)) {
    ++line_number;
  }

  while (std::getline(input, line)) {
    ++line_number;
    if (is_blank(line)) {
      continue;
    }

    std::vector<std::string> cells =
        parse_csv_cells(line, options.delimiter, line_number);
    if (csv_rows.empty()) {
      total_columns = cells.size();
      ASSERT(total_columns > options.target_columns,
             fmt::format("CSV dataset needs at least one feature column and {} "
                         "target column(s), got {} columns",
                         options.target_columns, total_columns));
      feature_columns = total_columns - options.target_columns;
    } else {
      ASSERT(cells.size() == total_columns,
             fmt::format("CSV row {} has {} columns, expected {}", line_number,
                         cells.size(), total_columns));
    }

    csv_rows.push_back({line_number, std::move(cells)});
  }

  ASSERT(!csv_rows.empty(),
         fmt::format("CSV dataset '{}' has no data rows", path));

  const std::vector<size_t> categorical_columns =
      validate_categorical_columns(options, feature_columns);
  const std::vector<CategoricalFeature> categorical_features =
      collect_categorical_features(csv_rows, categorical_columns);
  const std::vector<size_t> categorical_lookup =
      make_categorical_lookup(feature_columns, categorical_features);
  const size_t input_columns =
      expanded_feature_columns(feature_columns, categorical_features);

  input_values.reserve(csv_rows.size() * input_columns);
  target_values.reserve(csv_rows.size() * options.target_columns);

  for (const auto &row : csv_rows) {
    for (size_t column = 0; column < feature_columns; ++column) {
      append_encoded_feature(input_values, row, column, categorical_features,
                             categorical_lookup[column]);
    }

    for (size_t column = feature_columns; column < total_columns; ++column) {
      target_values.push_back(parse_float(row.cells[column], row.line_number,
                                          column));
    }
  }

  Tensor inputs = tensor_from_values(input_values,
                                     static_cast<int64_t>(csv_rows.size()),
                                     static_cast<int64_t>(input_columns),
                                     options.dtype, options.device,
                                     options.requires_grad);
  Tensor targets = tensor_from_values(
      target_values, static_cast<int64_t>(csv_rows.size()),
      static_cast<int64_t>(options.target_columns), options.dtype,
      options.device, options.requires_grad);

  return std::make_shared<TensorDataset>(std::move(inputs),
                                         std::move(targets));
}

} // namespace smollnet
