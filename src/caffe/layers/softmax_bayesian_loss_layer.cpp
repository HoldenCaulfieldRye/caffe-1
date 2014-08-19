#include <algorithm>
#include <cfloat>
#include <vector>

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
  int label_count = bottom[1]->count();
  LOG(INFO) << "num: " << num;
  LOG(INFO) << "dim: " << dim;
  LOG(INFO) << "dim: " << dim;
  LOG(INFO) << "labels_.count(): " << labels_.count();
  LOG(INFO) << "bottom[1]->count(): " << bottom[1]->count();

  Dtype* prior = labels_.mutable_cpu_data();
  caffe_set(labels_.count(), Dtype(FLT_MIN), prior);
  LOG(INFO) << "label_count" << labels_.count();
  for (int i = 0; i < label_count; ++i) {
    prior[static_cast<int>(label[i])] += 1.0 / label_count;
    LOG(INFO) << "label" << i << " " << label[i]; 
  } 
  // for (int i = 0; i < label_count; i++)
  //   LOG(INFO) << "the prior for label" << i << "is" << prior[static_cast<int>(label[i])];

  Dtype loss = 0;
  for (int i = 0; i < num; ++i) {
    //taking from prob_data the outputted probability of the correct label
    //FLT_MIN is smallest nonzero float (don't want to give 0 to log)
    loss += -log(max(prob_data[i * dim + static_cast<int>(label[i])], Dtype(FLT_MIN))) / static_cast<float>(prior[static_cast<int>(label[i])]);
  }
  return loss / (dim*num);
}

template <typename Dtype>
void SoftmaxWithBayesianLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
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
  for (int i = 0; i < num; ++i) {
    for (int j = 0; j < dim; ++j) {
      bottom_diff[i * dim + j] /= static_cast<float>(prior[i])*dim;
    }
  }
}


// template <typename Dtype>
// void SoftmaxWithBayesianLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
//     const bool propagate_down,
//     vector<Blob<Dtype>*>* bottom) {
//   // Compute the diff
//   Dtype* bottom_diff = (*bottom)[0]->mutable_cpu_diff();
//   const Dtype* prior = labels_.cpu_data();
//   const Dtype* prob_data = prob_.cpu_data();
//   memcpy(bottom_diff, prob_data, sizeof(Dtype) * prob_.count());
//   const Dtype* label = (*bottom)[1]->cpu_data();
//   int num = prob_.num();
//   int dim = prob_.count() / num;
//   for (int i = 0; i < num; ++i) {
//     bottom_diff[i * dim + static_cast<int>(label[i])] -= 1;
//     //    bottom_diff[i * dim + static_cast<int>(label[i])] /= static_cast<float>(prior[static_cast<int>(label[i])])*static_cast<float>(dim);
//   }
//   // for (int i = 0; i < num; ++i)  {
//   //   for (int j = 0; j < dim; ++j) {
//   //     bottom_diff[i * dim + j] /= static_cast<float>(prior[i])*static_cast<float>(dim);
//   //   }
//   // }
//   // Scale down gradient
//   caffe_scal(prob_.count(), Dtype(1) / dim, bottom_diff);
// }
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
