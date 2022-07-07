#include "models.h"

#include <torch/torch.h>

#include <cstddef>
#include <cstdio>
#include <iostream>
#include <string>
#include <vector>

int main() {
  const char *kDataRoot = "/home/nhmaruthi/.pytorch/FMNIST/FashionMNIST/raw/";
  const int64_t kTrainBatchSize = 64;
  const int64_t kTestBatchSize = 1000;
  const int64_t kNumberofEpochs = 10;
  const int64_t kLogInterval = 10;

  torch::manual_seed(1);
  torch::DeviceType device_type;

  if (torch::cuda::is_available()) {
    std::cout << "CUDA available! Training on GPU." << std::endl;
    device_type = torch::kCUDA;
  } else {
    std::cout << "Training on CPU." << std::endl;
    device_type = torch::kCPU;
  }

  torch::Device device(device_type);

  Net model(784, 128, 64, 10);
  model.to(device);

  auto train_dataset =
      torch::data::datasets::MNIST(kDataRoot)
          .map(torch::data::transforms::Normalize<>(0.1307, 0.3081))
          .map(torch::data::transforms::Stack<>());

  const size_t train_dataset_size = train_dataset.size().value();

  auto train_loader =
      torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
          std::move(train_dataset), kTrainBatchSize);

  auto test_dataset =
      torch::data::datasets::MNIST(kDataRoot,
                                   torch::data::datasets::MNIST::Mode::kTest)
          .map(torch::data::transforms::Normalize<>(0.1307, 0.3081))
          .map(torch::data::transforms::Stack<>());

  const std::size_t test_dataset_size = test_dataset.size().value();

  auto test_loader =
      torch::data::make_data_loader(std::move(test_dataset), kTrainBatchSize);

  torch::optim::SGD optimizer(model.parameters(),
                              torch::optim::SGDOptions(0.01).momentum(0.5));

  for (std::size_t epoch = 1; epoch < kNumberofEpochs; ++epoch) {
    train(epoch, model, device, *train_loader, optimizer, train_dataset_size);
    test(model, device, *test_loader, test_dataset_size);
  }

  return 0;
}