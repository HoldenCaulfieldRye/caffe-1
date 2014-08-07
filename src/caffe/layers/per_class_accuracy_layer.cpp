// Copyright 2014 BVLC and contributors.

#include <algorithm>
#include <cmath>
#include <cfloat>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/io.hpp"

using std::max;

namespace caffe {

  //NEXT STEPS: labels_ available in SetUp, so should move computation
  //to that side. copy the way Razvan allocates computation between
  //SetUp and Forward_cpu
  
template <typename Dtype>
void AccuracyLayer<Dtype>::SetUp(
  const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {
  CHECK_EQ(bottom.size(), 2) << "Accuracy Layer takes two blobs as input.";
  CHECK_EQ(top->size(), 1) << "Accuracy Layer takes 1 output.";
  CHECK_EQ(bottom[0]->num(), bottom[1]->num())
      << "The data and label should have the same number.";
  CHECK_EQ(bottom[1]->channels(), 1);
  CHECK_EQ(bottom[1]->height(), 1);
  CHECK_EQ(bottom[1]->width(), 1);
  (*top)[0]->Reshape(1, 2, 1, 1);
}

template <typename Dtype>
Dtype AccuracyLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    vector<Blob<Dtype>*>* top) {
  Dtype accuracy = 0;
  Dtype logprob = 0;
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* bottom_label = bottom[1]->cpu_data(); //Razvan calls this bottom_data
  int num = bottom[0]->num();
  int dim = bottom[0]->count() / bottom[0]->num();

  int label_count = bottom[1]->count();
  //need a Dtype array of length label_count initialised to zeros
  //need to look at how syntax works with Dtype
  //this is probably doubly wrong
  Dtype* labels_correct = labels_.mutable_cpu_data();
  Dtype* labels_count = labels_.mutable_cpu_data();
  Dtype* labels_accuracy = labels_.mutable_cpu_data();
  //labels_accuracy[i] = float(labels_correct[i]/labels_count[i]);
  
  caffe_set(labels_.count(), Dtype(0), labels_count);
  for (int i = 0; i < label_count; ++i)
    labels_count[int(bottom_label[i])] += 1;
  
  for (int i = 0; i < num; ++i) {
    // Accuracy
    Dtype maxval = -FLT_MAX;
    int max_id = 0;
    for (int j = 0; j < dim; ++j) {
      if (bottom_data[i * dim + j] > maxval) {
        maxval = bottom_data[i * dim + j];
        max_id = j;
      }
    }
    if (max_id == static_cast<int>(bottom_label[i]))
      labels_correct[int(bottom_label[i])] += 1;

    Dtype prob = max(bottom_data[i * dim + static_cast<int>(bottom_label[i])],
                     Dtype(kLOG_THRESHOLD));
    logprob -= log(prob);
  }
  for (int i=0; i<label_count; ++i) {
    labels_accuracy[i] = static_cast<float>(labels_correct[i]/labels_count[i]);
    accuracy += labels_accuracy[i];
  }
  accuracy = static_cast<float>(accuracy/label_count);
  // LOG(INFO) << "Accuracy: " << labels_accuracy; //is << defined for Dtype?  
  (*top)[0]->mutable_cpu_data()[0] = accuracy; //test score 0
  (*top)[0]->mutable_cpu_data()[1] = logprob / num;  //test score 1
  // Accuracy layer should not be used as a loss function.
  return Dtype(0);
}

INSTANTIATE_CLASS(AccuracyLayer);

}  // namespace caffe
