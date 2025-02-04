#include "ggml-backend.h"
#include "ggml.h"
#include "mnist-common.h"


#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <string>
#include <thread>
#include <vector>

typedef struct resnet_model {
    ggml_backend_t backend;
    struct ggml_context *ctx_weight = nullptr;
    struct ggml_context *ctx_compute = nullptr;

    resnet_model(const std::string &backend_name) {
        ggml_backend_dev_t dev = ggml_backend_dev_by_name(backend_name.c_str());
        if (dev == nullptr) {
            fprintf(stderr, "%s: ERROR: backend %s not found, available:\n", __func__, backend_name.c_str());
            for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
                ggml_backend_dev_t this_dev = ggml_backend_dev_get(i);
                fprintf(stderr, "  - %s (%s)\n", ggml_backend_dev_name(this_dev), ggml_backend_dev_description(this_dev));
            }
            exit(1);
        }

        fprintf(stderr, "%s: using %s (%s) backend\n", __func__, ggml_backend_dev_name(dev), ggml_backend_dev_description(dev));
        backend = ggml_backend_dev_init(dev, NULL);
        if (ggml_backend_is_cpu(backend)) {
            const int ncores_logical = std::thread::hardware_concurrency();
            ggml_backend_cpu_set_n_threads(backend, std::min(ncores_logical, (ncores_logical + 4)/2));
        }

        {
            const size_t size_meta = 1024*ggml_tensor_overhead();
            struct ggml_init_params params = {
                /*.mem_size   =*/ size_meta,
                /*.mem_buffer =*/ nullptr,
                /*.no_alloc   =*/ true,
            };
            ctx_weight = ggml_init(params);
        }

        {
            const size_t size_meta = GGML_DEFAULT_GRAPH_SIZE*ggml_tensor_overhead() + 3*ggml_graph_overhead();
            struct ggml_init_params params = {
                /*.mem_size   =*/ size_meta,
                /*.mem_buffer =*/ nullptr,
                /*.no_alloc   =*/ true,
            };
            ctx_compute = ggml_init(params);
        }
    }
} resnet_model;

int main(int argc, char ** argv) {
    // if (argc != 4 && argc != 5) {
    //     fprintf(stderr, "Usage: %s mnist-fc-f32.gguf data/MNIST/raw/t10k-images-idx3-ubyte data/MNIST/raw/t10k-labels-idx1-ubyte [CPU/CUDA0]\n", argv[0]);
    //     exit(1);
    // }

    resnet_model model("CPU");
    struct gguf_context *ctx; {
        struct gguf_init_params params = {
            true,
            &model.ctx_weight
        };
        ctx = gguf_init_from_file("/media/hbdesk/hb_desk_ext/weapons_classifier/weapon_classifier.gguf", params);
    };
}
