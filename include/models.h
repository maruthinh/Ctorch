#pragma once
#include <torch/torch.h>

struct Net : torch::nn::Module {
  Net(int n_fc1, int n_fc2, int n_fc3, int num_labels);
  torch::Tensor forward(torch::Tensor x);
  torch::nn::Linear fc1{nullptr}, fc2{nullptr}, output{nullptr};
  int m_n_fc1, m_n_fc2, m_n_fc3, m_num_labels;
};

// template <typename DataLoader>
// void train(size_t epoch, Net &model, torch::Device &device,
//            DataLoader &data_loader, torch::optim::Optimizer &optimizer,
//            std::size_t data_size);

// template <typename DataLoader>
// void test(Net &net, torch::Device &device, DataLoader &data_loader,
//           std::size_t data_size);

template <typename DataLoader>
void train(size_t epoch, Net &model, torch::Device &device,
           DataLoader &data_loader, torch::optim::Optimizer &optimizer,
           std::size_t dataset_size, std::size_t kLogInterval = 10) {
  model.train();
  std::size_t batch_idx = 0;

  for (auto &batch : data_loader) {
    auto data = batch.data.to(device), targets = batch.target.to(device);
    optimizer.zero_grad();
    auto output = model.forward(data);
    auto loss = torch::nll_loss(output, targets);
    AT_ASSERT(!std::isnan(loss.template item<float>()));
    loss.backward();
    optimizer.step();

    if (batch_idx++ % kLogInterval == 0) {
      std::cout << epoch << "\t" << batch_idx * batch.data.size(0) << "\t"
                << dataset_size << "\t" << loss.template item<float>() << "\n";
    }
  }
}

template <typename DataLoader>
void test(Net &model, torch::Device device, DataLoader &data_loader,
          std::size_t dataset_size) {
  torch::NoGradGuard no_grad;
  model.eval();
  double test_loss = 0;
  int32_t correct = 0;

  for (const auto &batch : data_loader) {
    auto data = batch.data.to(device), targets = batch.target.to(device);
    auto output = model.forward(data);
    test_loss += torch::nll_loss(output, targets, {}, torch::Reduction::Sum)
                     .template item<float>();
    auto pred = output.argmax(1);
    correct += pred.eq(targets).sum().template item<int64_t>();
  }

  test_loss /= dataset_size;
  std::cout << test_loss << "\t" << static_cast<double>(correct) / dataset_size
            << "\n";
}