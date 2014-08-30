// Copyright 2014 BVLC and contributors.
// #include <iostream>
#include <algorithm>
#include <cmath>
#include <cfloat>
#include <vector>
#include <iostream>

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
void PerClassAccuracyLayer<Dtype>::SetUp(
  const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {
  CHECK_EQ(bottom.size(), 2) << "Accuracy Layer takes two blobs as input.";
  CHECK_EQ(top->size(), 1) << "Accuracy Layer takes 1 output.";
  CHECK_EQ(bottom[0]->num(), bottom[1]->num())
      << "The data and label should have the same number.";
  CHECK_EQ(bottom[1]->channels(), 1);
  CHECK_EQ(bottom[1]->height(), 1);
  CHECK_EQ(bottom[1]->width(), 1);
  (*top)[0]->Reshape(1, 2, 1, 1);

  //std::cout << "PER CLASS ACCURACY LAYER:" << std::endl;

  //std::cout << "bottom[0] dimensions:" << std::endl;
  //std::cout << "num: " << bottom[0]->num() << std::endl;
  //std::cout << "channels: " << bottom[0]->channels() << std::endl;
  //std::cout << "height: " << bottom[0]->height() << std::endl;
  //std::cout << "width: " << bottom[0]->width() << std::endl;
  //std::cout << "count: " << bottom[0]->count() << std::endl;

  //std::cout << std::endl;
  
  //std::cout << "top dimensions:" << std::endl;
  //std::cout << "num: " << (*top)[0]->num() << std::endl;
  //std::cout << "channels: " << (*top)[0]->channels() << std::endl;
  //std::cout << "height: " << (*top)[0]->height() << std::endl;
  //std::cout << "width: " << (*top)[0]->width() << std::endl;
  //std::cout << "count: " << (*top)[0]->count() << std::endl;  

  labels_.Reshape(bottom[1]->num(), 1, 1, 1);
}

template <typename Dtype>
Dtype PerClassAccuracyLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    vector<Blob<Dtype>*>* top) {
  Dtype accuracy = 0;
  Dtype logprob = 0;
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* bottom_label = bottom[1]->cpu_data(); //threshold_layer calls this bottom_data
  int num = bottom[0]->num();
  int dim = bottom[0]->count() / bottom[0]->num();

  int label_count = bottom[1]->count();
  
  Dtype* labels_count = labels_.mutable_cpu_data();
  caffe_set(labels_.count(), Dtype(FLT_MIN), labels_count);
  ////std::cout << "label_count" << labels_.count();
  for (int i = 0; i < label_count; ++i) {
    labels_count[static_cast<int>(bottom_label[i])] += 1.0;
    ////std::cout << "bottom_label" << i << " " << bottom_label[i];
  } 


  Dtype* prior = labels_.mutable_cpu_data();
  prior[0] = 0;
  prior[1] = 0;
  ////std::cout << "labels for this batch: ";
  for (int i = 0; i < num; ++i) {
    prior[static_cast<int>(label[i])] += 1.0 / num;
    ////std::cout << label[i] << " ";
  } 

  // const Dtype* prob_data = prob_.cpu_data();
  const Dtype* label = bottom[1]->cpu_data();
  //std::cout << std::endl << "per class accuracy layer" << std::endl;
  //std::cout << "output probs (bottom_data):" << std::endl;
  int n = std::min(20,num);
  // for (int i=0; i<n; i++) {
  //   if (static_cast<int>(label[i]) == 0)
      //std::cout << "min class ";
   //std::cout << "case " << i << ": ";
    // for (int neur = 0; neur < dim; neur++)
      //std::cout << bottom_data[i*dim + neur] << ", ";
    //std::cout << std::endl;
  // }  
  
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
    if (max_id == static_cast<int>(bottom_label[i])){
      accuracy += 1.0/static_cast<float>(labels_count[max_id]);
      //////std::cout << "acMF " << accuracy << " " << labels_count[max_id] << " " << num << "\n";
    }
    
    Dtype prob = max(bottom_data[i * dim + static_cast<int>(bottom_label[i])],
                     Dtype(kLOG_THRESHOLD));
    logprob -= log(prob) / prior[static_cast<int>(label[i])];
  }
  //////std::cout << "accF " << accuracy << "\n";
  // LOG(INFO) << "Accuracy: " << labels_accuracy; //is << defined for Dtype?
  //////std::cout << dim << " " << bottom[0]->count();  
  (*top)[0]->mutable_cpu_data()[0] = accuracy / static_cast<float>(dim); //test score 0
  (*top)[0]->mutable_cpu_data()[1] = logprob / num;  //test score 1
  // Accuracy layer should not be used as a loss function.
  return Dtype(0);
}

INSTANTIATE_CLASS(PerClassAccuracyLayer);

}  // namespace caffe
