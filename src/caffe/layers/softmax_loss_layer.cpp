// Copyright 2014 BVLC and contributors.

#include <algorithm>
#include <cfloat>
#include <vector>
#include <iostream>

#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/util/math_functions.hpp"

using std::max;

namespace caffe {

template <typename Dtype>
void SoftmaxWithLossLayer<Dtype>::SetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  CHECK_EQ(bottom.size(), 2) << "SoftmaxLoss Layer takes two blobs as input.";
  CHECK_EQ(top->size(), 0) << "SoftmaxLoss Layer takes no blob as output.";
  softmax_bottom_vec_.clear();
  softmax_bottom_vec_.push_back(bottom[0]);
  softmax_top_vec_.push_back(&prob_);
  softmax_layer_->SetUp(softmax_bottom_vec_, &softmax_top_vec_);
}

template <typename Dtype>
Dtype SoftmaxWithLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {
  // The forward pass computes the softmax prob values.
  // what is _prob ? looks like instantiation of some class
  softmax_bottom_vec_[0] = bottom[0];
  softmax_layer_->Forward(softmax_bottom_vec_, &softmax_top_vec_);
  const Dtype* prob_data = prob_.cpu_data();
  const Dtype* label = bottom[1]->cpu_data();
  int num = prob_.num();
  int dim = prob_.count() / num;
  Dtype loss = 0;

  std::cout << std::endl << "SL layer" << std::endl;
  std::cout << "output probs:" << std::endl;
  int n = std::min(20,num);
  for (int i=0; i<n; i++) {
    if (static_cast<int>(label[i]) == 0)
      std::cout << "min class ";
   std::cout << "case " << i << ": ";
    for (int neur = 0; neur < dim; neur++)
      std::cout << prob_data[i*dim + neur] << ", ";
    std::cout << std::endl;
  }
  
  for (int i = 0; i < num; ++i) {
    //taking from prob_data the outputted probability of the correct label
    //FLT_MIN is smallest nonzero float (don't want to give 0 to log)
    loss += -log(max(prob_data[i * dim + static_cast<int>(label[i])],
                     Dtype(FLT_MIN)));
  }
  return loss / num;
}

template <typename Dtype>
void SoftmaxWithLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const bool propagate_down,
    vector<Blob<Dtype>*>* bottom) {
  // Compute the diff
  Dtype* bottom_diff = (*bottom)[0]->mutable_cpu_diff();
  const Dtype* prob_data = prob_.cpu_data();
  memcpy(bottom_diff, prob_data, sizeof(Dtype) * prob_.count());
  const Dtype* label = (*bottom)[1]->cpu_data();
  int num = prob_.num();
  int dim = prob_.count() / num;
  for (int i = 0; i < num; ++i) {
    bottom_diff[i * dim + static_cast<int>(label[i])] -= 1;
  }
  // Scale down gradient
  caffe_scal(prob_.count(), Dtype(1) / num, bottom_diff);
}


INSTANTIATE_CLASS(SoftmaxWithLossLayer);


}  // namespace caffe
