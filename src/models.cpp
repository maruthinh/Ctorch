#include "models.h"
#include "models.ipp"
#include <torch/torch.h>

Net::Net(int n_fc1, int n_fc2, int n_fc3, int num_labels)
    : m_n_fc1(n_fc1), m_n_fc2(n_fc2), m_n_fc3(n_fc3), m_num_labels(num_labels),
      fc1(n_fc1, n_fc2), fc2(n_fc2, n_fc3), output(n_fc3, num_labels) {
  register_module("fc1", fc1);
  register_module("fc2", fc2);
  register_module("output", output);
}

torch::Tensor Net::forward(torch::Tensor x) {
  x = x.view({-1, m_n_fc1});
  x = torch::relu(fc1(x));
  x = torch::relu(fc2(x));
  x = output(x);
  x = torch::log_softmax(x, 1);

  return x;
}


