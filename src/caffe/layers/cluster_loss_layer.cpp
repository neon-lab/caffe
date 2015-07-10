#include <algorithm>
#include <cfloat>
#include <cmath>
#include <vector>
#include <map>
#include <utility>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

#define CLUSTER_LOSS_DELTA 1
#define CLUSTER_LOSS_LAMBDA 0.001

typedef std::pair<int, int> index_pair;

namespace caffe {

template <typename Dtype>
static void collect_pairs(std::map<int, index_pair> *labels, Blob<Dtype> *label_blob) {
    const Dtype* label = label_blob->cpu_data();
    for (int  i = 0; i < label_blob->shape(0); i++) {
        if (label[i] != (int) label[i]) {
            LOG(FATAL) << "Non-integer label " << label[i];
        }
        std::map<int, index_pair>::iterator found_it = labels->find((int) label[i]);
        if (found_it == labels->end()) {
            // haven't seen this label before
            (*labels)[(int) label[i]] = index_pair(i, -1);
        }
        else if (found_it->second.second == -1) {
            // saw one sample before. insert second sample
            found_it->second.second = i;
        }
    }
}

template <typename Dtype>
static int compute_vector_len(Blob<Dtype> *data_blob) {
    int len = 1;
    for (int i = 1; i < data_blob->num_axes(); i++) {
        len *= data_blob->shape(i);
    }
    return len;
}

template <typename Dtype>
Dtype ClusterLossLayer<Dtype>::get_data_dot_product(int r, int c) {
    return data_dot_products[data_vec_count*r + c];
}

template <typename Dtype>
void ClusterLossLayer<Dtype>::set_data_dot_product(int r, int c, Dtype v) {
    // TODO: consider vectorizing the dot products to compute all of these in one swoop
    data_dot_products[data_vec_count*r + c] = 
    data_dot_products[data_vec_count*c + r] = v;
}

template <typename Dtype>
void ClusterLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
    std::map<int, index_pair> labels; // set with all the labels in current mini-batch
    collect_pairs(&labels, bottom[1]);

    int vec_len = compute_vector_len(bottom[0]);
    const Dtype* bottom_data = bottom[0]->cpu_data();
    Dtype* loss = top[0]->mutable_cpu_data(); // store loss in loss[0]

    data_vec_count = bottom[0]->shape(0);
    data_dot_products.reset(new Dtype[data_vec_count*data_vec_count]);
    std::fill(data_dot_products.get(), data_dot_products.get() + data_vec_count*data_vec_count, NAN);

    for (std::map<int, index_pair>::iterator it = labels.begin(); it != labels.end(); it++) {
        int a = it->second.first;
        int b = it->second.second;

        if (a == -1 || b == -1) {
            LOG(FATAL) << "Invalid pair " << a << ", " << b;
        }

        Dtype dot_data_ab = get_data_dot_product(a, b);
        if (isnan(dot_data_ab)) {
            dot_data_ab = caffe_cpu_dot(
                vec_len, &bottom_data[vec_len*a], &bottom_data[vec_len*b]
            );
            set_data_dot_product(a, b, dot_data_ab);
        }

        for (int j = 0; j < bottom[1]->shape(0); j++) {
            if (j == a || j == b) continue;
            Dtype dot_data_ja = caffe_cpu_dot(
                vec_len, &bottom_data[vec_len*j], &bottom_data[vec_len*a]
            );
            Dtype dot_data_jb = caffe_cpu_dot(
                vec_len, &bottom_data[vec_len*j], &bottom_data[vec_len*b]
            );
            loss[0] +=
                std::max((Dtype) 0, dot_data_ja - dot_data_ab + CLUSTER_LOSS_DELTA) +
                std::max((Dtype) 0, dot_data_jb - dot_data_ab + CLUSTER_LOSS_DELTA);
        }
    }
    Dtype N = bottom[1]->shape(0) / 2;
    loss[0] /= N*(N - 1);
    Dtype data_squared_sum = 0;
    for (int i = 0; i < data_vec_count*vec_len; i++) {
        data_squared_sum += pow(bottom_data[i], 2);
    }
    loss[0] += CLUSTER_LOSS_LAMBDA / (2 * data_vec_count * vec_len) * data_squared_sum;
}

template <typename Dtype>
void ClusterLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    if (propagate_down[1]) {
        LOG(FATAL) << this->type() << " Layer cannot backpropagate to label inputs.";
    }
    if (propagate_down[0]) {
        Dtype N = bottom[0]->shape(0) / 2;
        std::map<int, index_pair> labels; // set with all the labels in current mini-batch
        collect_pairs(&labels, bottom[1]);

        int vec_len = compute_vector_len(bottom[0]);
        const Dtype* bottom_data = bottom[0]->cpu_data();
        const Dtype* label = bottom[1]->cpu_data();
        Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
        std::fill(bottom_diff, bottom_diff + bottom[0]->shape(0)*vec_len, 0);

        for (int a = 0; a < 2*N; a++) {
            // find some other index b with the same label as a
            int b =
                (a == labels[label[a]].first) ? labels[label[a]].second : labels[label[a]].first;
            if (b == -1) {
                LOG(FATAL) << "Invalid pair " << a << ", " << b;
            }
            // iterate through all other samples
            for (int i = 0; i < 2*N; i++) {
                if (i == a || i == b) continue;
                // find index j with same label as i
                int j =
                    (i == labels[label[i]].first) ? labels[label[i]].second : labels[label[i]].first;
                if (i == j || j == -1 || label[i] != label[j]) {
                    LOG(FATAL) << "Invalid pair " << i << ", " << j;
                }
                // compute indicators
                Dtype dot_data_ab = get_data_dot_product(a, b);
                if (isnan(dot_data_ab)) {
                    dot_data_ab = caffe_cpu_dot(
                        vec_len, &bottom_data[vec_len*a], &bottom_data[vec_len*b]
                    );
                    set_data_dot_product(a, b, dot_data_ab);
                }

                Dtype dot_data_ai = get_data_dot_product(a, i);
                if (isnan(dot_data_ai)) {
                    dot_data_ai = caffe_cpu_dot(
                        vec_len, &bottom_data[vec_len*a], &bottom_data[vec_len*i]
                    );
                    set_data_dot_product(a, i, dot_data_ai);
                }

                Dtype dot_data_bi = get_data_dot_product(b, i);
                if (isnan(dot_data_bi)) {
                    dot_data_bi = caffe_cpu_dot(
                        vec_len, &bottom_data[vec_len*b], &bottom_data[vec_len*i]
                    );
                    set_data_dot_product(b, i, dot_data_bi);
                }

                Dtype dot_data_ij = get_data_dot_product(i, j);
                if (isnan(dot_data_ij)) {
                    dot_data_ij = caffe_cpu_dot(
                        vec_len, &bottom_data[vec_len*i], &bottom_data[vec_len*j]
                    );
                    set_data_dot_product(i, j, dot_data_ij);
                }

                int del_a_i = dot_data_ai - dot_data_ab + CLUSTER_LOSS_DELTA > 0 ? 1 : 0;
                int del_i_a = dot_data_ai - dot_data_ij + CLUSTER_LOSS_DELTA > 0 ? 1 : 0;
                int del_b_i = dot_data_bi - dot_data_ab + CLUSTER_LOSS_DELTA > 0 ? 1 : 0;
                // scale and sum vectors to buf
                for (int vec_i = 0; vec_i < vec_len; vec_i++) {
                    bottom_diff[vec_len*a + vec_i] +=
                        bottom_data[vec_len*i + vec_i]*(del_a_i + del_i_a) -
                        bottom_data[vec_len*b + vec_i]*(del_a_i + del_b_i); 
                }
            }
            for (int vec_i = 0; vec_i < vec_len; vec_i++) {
                bottom_diff[vec_len*a + vec_i] /= (N*(N - 1));
                bottom_diff[vec_len*a + vec_i] += CLUSTER_LOSS_LAMBDA * bottom_data[vec_len*a + vec_i] / (data_vec_count * vec_len);
            }
        }
/*
        // print out the gradients
        for (int a = 0; a < bottom[0]->shape(0); a++) {
            for (int i = 0; i < vec_len; i++) {
                LOG(INFO) << "> " << bottom_diff[a*vec_len + i];
            }
            LOG(INFO) << "<<<";
        }
*/
    }
    data_dot_products.reset(NULL);
}

INSTANTIATE_CLASS(ClusterLossLayer);
REGISTER_LAYER_CLASS(ClusterLoss);

}  // namespace caffe

