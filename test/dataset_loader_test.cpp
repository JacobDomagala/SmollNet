#include <smollnet.hpp>

#include <helpers.hpp>

#include <filesystem>
#include <fstream>

using namespace smollnet;

namespace {

float tensor_value(const Tensor &tensor, size_t index) {
  return static_cast<float *>(tensor.cpu().data())[index];
}

void write_csv_fixture(const std::filesystem::path &path) {
  std::ofstream out(path);
  out << "x0,x1,y\n";
  out << "1.0,2.0,10.0\n";
  out << "3.0,4.0,20.0\n";
  out << "5.0,6.0,30.0\n";
}

} // namespace

int main() {
  const auto path = std::filesystem::temp_directory_path() /
                    "smollnet_dataset_loader_test.csv";
  write_csv_fixture(path);

  CSVLoaderOptions options;
  options.device = Device::CPU;
  options.has_header = true;
  options.target_columns = 1;

  auto dataset = load_csv_dataset(path.string(), options);
  ASSERT(dataset->size() == 3, "Dataset should contain three rows");
  ASSERT(dataset->inputs().size(0) == 3, "Input batch dimension mismatch");
  ASSERT(dataset->inputs().size(1) == 2, "Input feature dimension mismatch");
  ASSERT(dataset->targets().size(1) == 1, "Target dimension mismatch");

  DatasetBatch second_row = dataset->batch(1, 1);
  ASSERT(tensor_value(second_row.inputs, 0) == 3.0f,
         "Unexpected first feature in row batch");
  ASSERT(tensor_value(second_row.inputs, 1) == 4.0f,
         "Unexpected second feature in row batch");
  ASSERT(tensor_value(second_row.targets, 0) == 20.0f,
         "Unexpected target in row batch");

  DataLoaderOptions loader_options;
  loader_options.batch_size = 2;

  DataLoader loader(dataset, loader_options);
  ASSERT(loader.num_batches() == 2, "DataLoader should keep the tail batch");

  size_t batch_index = 0;
  for (auto &batch : loader) {
    if (batch_index == 0) {
      ASSERT(batch.inputs.size(0) == 2, "First batch should have two rows");
      ASSERT(tensor_value(batch.inputs, 0) == 1.0f,
             "Unexpected first value in first batch");
      ASSERT(tensor_value(batch.targets, 1) == 20.0f,
             "Unexpected second target in first batch");
    } else {
      ASSERT(batch.inputs.size(0) == 1, "Tail batch should have one row");
      ASSERT(tensor_value(batch.inputs, 0) == 5.0f,
             "Unexpected first value in tail batch");
    }
    ++batch_index;
  }
  ASSERT(batch_index == 2, "Range loop should visit two batches");
  ASSERT(!loader.has_next(), "DataLoader should be exhausted");

  batch_index = 0;
  for (auto &batch : loader) {
    (void)batch;
    ++batch_index;
  }
  ASSERT(batch_index == 2, "Range loop should reset between epochs");

  DataLoaderOptions drop_tail_options;
  drop_tail_options.batch_size = 2;
  drop_tail_options.drop_last = true;

  DataLoader dropped_tail(dataset, drop_tail_options);
  ASSERT(dropped_tail.num_batches() == 1,
         "drop_last should remove the tail batch");

  std::filesystem::remove(path);
  return 0;
}
