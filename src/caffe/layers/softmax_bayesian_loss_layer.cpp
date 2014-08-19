#include <algorithm>
#include <cfloat>
#include <vector>
#include<iostream>

#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/util/math_functions.hpp"

using std::max;

namespace caffe {

template <typename Dtype>
void SoftmaxWithBayesianLossLayer<Dtype>::SetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  CHECK_EQ(bottom.size(), 2) << "SoftmaxLoss Layer takes two blobs as input.";
  CHECK_EQ(top->size(), 0) << "SoftmaxLoss Layer takes no blob as output.";
  CHECK_EQ(bottom[0]->num(), bottom[1]->num()) << "The data and label should have the same number.";
  CHECK_EQ(bottom[1]->channels(), 1);
  CHECK_EQ(bottom[1]->height(), 1);
  CHECK_EQ(bottom[1]->width(), 1);
  labels_.Reshape(bottom[1]->num(), 1, 1, 1);
  softmax_bottom_vec_.clear();
  softmax_bottom_vec_.push_back(bottom[0]);
  softmax_top_vec_.push_back(&prob_);
  softmax_layer_->SetUp(softmax_bottom_vec_, &softmax_top_vec_);
}

template <typename Dtype>
Dtype SoftmaxWithBayesianLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {
  // The forward pass computes the softmax prob values.
  // what is _prob ? looks like instantiation of some class
  softmax_bottom_vec_[0] = bottom[0];
  softmax_layer_->Forward(softmax_bottom_vec_, &softmax_top_vec_);
  const Dtype* prob_data = prob_.cpu_data();
  const Dtype* label = bottom[1]->cpu_data();
  int num = prob_.num();
  int dim = prob_.count() / num;
  // int label_count = bottom[1]->count();
  
  Dtype* prior = labels_.mutable_cpu_data();
  std::cout << "labels for this batch: ";
  for (int i = 0; i < num; ++i) {
    prior[static_cast<int>(label[i])] += 1.0 / num;
    std::cout << label[i] << " ";
  } 
  std::cout << std::endl << std::endl << "prior for this batch is: ";
  for (int i = 0; i < dim; ++i)
    std::cout << prior[i] << ", ";
  std::cout << std::endl;
  caffe_set(labels_.count(), Dtype(FLT_MIN), prior);

  Dtype loss = 0;
  for (int i = 0; i < num; ++i) {
    //taking from prob_data the outputted probability of the correct label
    //FLT_MIN is smallest nonzero float (don't want to give 0 to log)
    loss += -log(max(prob_data[i * dim + static_cast<int>(label[i])], Dtype(FLT_MIN))) / static_cast<float>(prior[static_cast<int>(label[i])]);
  }
  return loss / (dim*num);
}

template <typename Dtype>
// computes dE/dz for every neuron input vector z = <x,w>+b
// this does NOT update the weights, it merely calculates dy/dz
void SoftmaxWithBayesianLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const bool propagate_down,
    vector<Blob<Dtype>*>* bottom) {
  // Compute the diff
  Dtype* bottom_diff = (*bottom)[0]->mutable_cpu_diff();
  const Dtype* prior = labels_.cpu_data();
  const Dtype* prob_data = prob_.cpu_data();
  //bottom_diff starts off as the outputted probabilities, not the loss!
  //loss actually never backpropped... wtf
  memcpy(bottom_diff, prob_data, sizeof(Dtype) * prob_.count());
  const Dtype* label = (*bottom)[1]->cpu_data();
  int num = prob_.num();         //batchSize, num imgs
  int dim = prob_.count() / num; //num neurons, dimensionality
  
  std::cout << "bottom_diff before backward_pass is the outputted probabilities:" << std::endl;
  //for some reason, computing a gradient for each case; I guess they get averaged in the code
  //for weight update
  for (int i = 0; i < num; ++i)  {
    for (int j = 0; j < dim; ++j) 
      std::cout << "bottom_diff[" << i << "*" << dim << "+" <<j << "]: " << bottom_diff[i*dim+j]<< ",  ";
    std::cout << std::endl;
  }
  std::cout << std::endl;
  
  for (int i = 0; i < num; ++i) {
    // softmax gradient: bit.ly/1tmehE9
    bottom_diff[i * dim + static_cast<int>(label[i])] -= 1;
    // bottom_diff[i * dim + static_cast<int>(label[i])] /= static_cast<float>(prior[i])*dim;
  }
  // Scale down gradient
  // multiply the 1st prob_.count() elements of bottom_diff by 1/dim
  caffe_scal(prob_.count(), Dtype(1) / dim, bottom_diff);

  std::cout << "bottom_diff after backward_pass:" << std::endl;
  for (int i = 0; i < num; ++i)  {
    for (int j = 0; j < dim; ++j) 
      std::cout << "bottom_diff[" << i << "*" << dim << "+" <<j << "]: " << bottom_diff[i*dim+j]<< ",  ";
    std::cout << std::endl;
  }
  std::cout << std::endl;

  for (int i = 0; i < num; ++i)  {
    for (int j = 0; j < dim; ++j) 
      bottom_diff[i * dim + j] /= static_cast<float>(prior[label[i]]); //*dim;
  }
  
  std::cout << "bottom_diff after bayesian stuff:" << std::endl;
  for (int i = 0; i < num; ++i)  {
    for (int j = 0; j < dim; ++j) 
      std::cout << "bottom_diff[" << i << "*" << dim << "+" <<j << "]: " << bottom_diff[i*dim+j]<< ",  ";
    std::cout << std::endl;
  }
  std::cout << std::endl;
}
/*
template <typename Dtype>
void SoftmaxWithBayesianLossLayer<Dtype>::Backward_cpu_old(const vector<Blob<Dtype>*>& top,
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
*/

INSTANTIATE_CLASS(SoftmaxWithBayesianLossLayer);


}  // namespace caffe
