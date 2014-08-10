
// Copyright 2014 BVLC and contributors.

#include <cmath>
#include <cstring>
#include <vector>

#include "cuda_runtime.h"
#include "gtest/gtest.h"
#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

#include "caffe/test/test_caffe_main.hpp"

namespace caffe {

extern cudaDeviceProp CAFFE_TEST_CUDA_PROP;

template <typename Dtype>
class ThresholdLayerTest : public ::testing::Test {
 protected:
  ThresholdLayerTest()
      : blob_bottom_data_(new Blob<Dtype>(10, 5, 1, 1)),
        blob_bottom_label_(new Blob<Dtype>(10, 5, 1, 1)),
        blob_top_data_(new Blob<Dtype>(10,5,1,1)),
        blob_top_label_(new Blob<Dtype>(10,5,1,1)) {
    // fill the values
    FillerParameter filler_param;
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_data_);
    blob_bottom_vec_.push_back(blob_bottom_data_);
    filler.Fill(this->blob_bottom_label_);
    blob_bottom_vec_.push_back(blob_bottom_label_);
    blob_top_vec_.push_back(blob_top_data_);
    blob_top_vec_.push_back(blob_top_label_);
}
  /*virtual ~ThresholdLayerTest() {
    delete blob_bottom_data_;
    delete blob_bottom_label_;
    delete blob_top_data_;
    delete blob_top_label_;
 }*/
  Blob<Dtype>* const blob_bottom_data_;
  Blob<Dtype>* const blob_bottom_label_;
  Blob<Dtype>* const blob_top_data_;
  Blob<Dtype>* const blob_top_label_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

typedef ::testing::Types<float, double> Dtypes;
TYPED_TEST_CASE(ThresholdLayerTest, Dtypes);

TYPED_TEST(ThresholdLayerTest, TestGradientCPU) {
  LayerParameter layer_param;
  Caffe::set_mode(Caffe::CPU);
  ThresholdLayer<TypeParam> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, &(this->blob_top_vec_));
  GradientChecker<TypeParam> checker(1e-2, 1e-2, 1701);
  checker.CheckGradientEltwise(&layer, &(this->blob_bottom_vec_), &(this->blob_top_vec_));
}

}  // namespace caffe