#include <cmath>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/vision_layers.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

#define SUBJECT_COUNT 8
#define FEATURE_DIMENSIONALITY 10

namespace caffe {

template <typename TypeParam>
class ClusterLossLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  ClusterLossLayerTest()
      : blob_bottom_data_(new Blob<Dtype>(SUBJECT_COUNT * 2, FEATURE_DIMENSIONALITY, 1, 1)),
        blob_bottom_label_(new Blob<Dtype>(SUBJECT_COUNT * 2, 1, 1, 1)),
        blob_top_loss_(new Blob<Dtype>()) {
    // fill the values
    Caffe::set_random_seed(1701);
    FillerParameter filler_param;
    filler_param.set_std(10);

    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_data_);
    int vec_len = blob_bottom_data_->shape(1);
    for (int i = 0; i < blob_bottom_data_->shape(0); i++) {
        for (int vec_i = 0; vec_i < vec_len; vec_i++) {
            blob_bottom_data_->mutable_cpu_data()[vec_len*i + vec_i] =
                vec_i*i;
        }
    }
    blob_bottom_vec_.push_back(blob_bottom_data_);

    for (int i = 0; i < blob_bottom_label_->count(); ++i) {
        if (i < SUBJECT_COUNT)
            blob_bottom_label_->mutable_cpu_data()[i] = i;
        else
            blob_bottom_label_->mutable_cpu_data()[i] = i - SUBJECT_COUNT;
    }
    blob_bottom_vec_.push_back(blob_bottom_label_);

    blob_top_vec_.push_back(blob_top_loss_);
  }
  virtual ~ClusterLossLayerTest() {
    delete blob_bottom_data_;
    delete blob_bottom_label_;
    delete blob_top_loss_;
  }
  Blob<Dtype>* const blob_bottom_data_;
  Blob<Dtype>* const blob_bottom_label_;
  Blob<Dtype>* const blob_top_loss_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(ClusterLossLayerTest, TestDtypesAndDevices);

TYPED_TEST(ClusterLossLayerTest, TestGradientL1) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  ClusterLossLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1, 1701, 1, 0.01);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_, 0);
}

TYPED_TEST(ClusterLossLayerTest, TestGradientL2) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  ClusterLossLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1, 1701);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_, 0);
}

}  // namespace caffe
