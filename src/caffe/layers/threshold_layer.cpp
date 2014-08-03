#include <algorithm>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/util/math_functions.hpp"

using std::max;

namespace caffe {

template <typename Dtype>
void ThresholdLayer<Dtype>::SetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) { // create prior counts here. Not sure if it's the right place. When is setup called?
  CHECK_EQ(bottom.size(), 2) << "Threshold Layer takes the parameters to modify and the labels from which to derive the priors as input.";
  CHECK_EQ(top->size(), 2) << "Softmax Layer gives the modified parameters and the unmodified labels as output.";
  CHECK_EQ(bottom[0]->num(), bottom[1]->num()) << "The data and label should have the same number.";

  (*top)[0]->ReshapeLike(*bottom[0]);
  (*top)[1]->ReshapeLike(*bottom[1]);
  
  labels_.Reshape(1, bottom[0]->channels(),
      bottom[0]->height(), bottom[0]->width());

  int label_count = bottom[1]->count();
  Dtype* labels_data = labels_.mutable_cpu_data();
  const Dtype* bottom_data = bottom[1]->cpu_data();

  caffe_set(labels_.count(), Dtype(0), labels_data); // is this neccesary? Why isn't caffe_set using cblas in general? (doesn't matter for 0 case)
  for (int i = 0; i < label_count; ++i) { // assumes labels are integers starting from 0
    labels_data[int(bottom_data[i])] += 1;
  }

  prior_.Reshape(bottom[0]->num(), bottom[0]->channels(),
      bottom[0]->height(), bottom[0]->width());
  
  int num = prior_.num();
  int count = prior_.count();
  int dim = count / num;

  Dtype* prior_data = prior_.mutable_cpu_data();
  for (int i = 0; i < num; ++i) {
    for (int j = 0; j < dim; ++j) {
      prior_data[i * dim + j] = labels_data[j];
    }
  }

  caffe_cpu_scale(count, Dtype(1.0 / label_count), prior_data, prior_data);

}
template <typename Dtype>
Dtype ThresholdLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    vector<Blob<Dtype>*>* top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* prior_data = prior_.cpu_data();
  Dtype* top_data = (*top)[0]->mutable_cpu_data();
  const int count = bottom[0]->count();
  caffe_mul(count, bottom_data, prior_data, top_data);

  //caffe_copy(count, bottom[1]->cpu_data(), (*top)[1]->mutable_cpu_data());
  return Dtype(0);
}

template <typename Dtype>
void ThresholdLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top, //not sure if you need to do anything for the data layer
    const bool propagate_down,
    vector<Blob<Dtype>*>* bottom) {
  if (propagate_down) {
    const Dtype* prior_data = prior_.cpu_data();
    const Dtype* top_diff = top[0]->cpu_diff();
    Dtype* bottom_diff = (*bottom)[0]->mutable_cpu_diff();
    const int count = (*bottom)[0]->count();
    caffe_mul(count, top_diff, prior_data, bottom_diff);
  }
}

INSTANTIATE_CLASS(ThresholdLayer);

}  // namespace caffe
