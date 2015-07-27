#include <algorithm>
#include <cfloat>
#include <cmath>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void MapLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
    float lambda = this->layer_param_.map_loss_param().lambda();
    int num_el = bottom[0]->shape(0)*bottom[0]->shape(1)*bottom[0]->shape(2);
    const Dtype* data = bottom[0]->cpu_data();
    const Dtype* label = bottom[1]->cpu_data();

    Dtype* loss = top[0]->mutable_cpu_data();
    // store loss in loss[0]
    loss[0] = 0;
    for (int i = 0; i < num_el; i++) {
        loss[0] += pow(data[i], 1 - label[i]) * pow(1 - data[i], label[i]) +
                   0.5 * lambda * pow(data[i], 2);
    }
}

template <typename Dtype>
void MapLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    if (propagate_down[1]) {
        LOG(FATAL) << this->type() << " Layer cannot backpropagate to label inputs.";
    }
    if (propagate_down[0]) {
        float lambda = this->layer_param_.map_loss_param().lambda();
        int num_el = bottom[0]->shape(0)*bottom[0]->shape(1)*bottom[0]->shape(2);
        const Dtype* data = bottom[0]->cpu_data();
        const Dtype* label = bottom[1]->cpu_data();
        Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
        
        for (int i = 0; i < num_el; i++) {
            bottom_diff[i] = - (pow(1 - data[i], label[i] - 1) / pow(data[i], label[i])) * (data[i] + label[i] - 1)
                             + lambda * data[i];
        }
    }
}

INSTANTIATE_CLASS(MapLossLayer);
REGISTER_LAYER_CLASS(MapLoss);

}  // namespace caffe
