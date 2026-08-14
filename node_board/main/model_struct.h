#pragma once

#include <stdint.h>
#include <stddef.h>
#include "esp_err.h"
#include "bitlinear.h"

struct TransformerLayer
{
    LayerWeights w_q;
    LayerWeights w_k;
    LayerWeights w_v;
    LayerWeights w_o;

    LayerWeights w_gate;
    LayerWeights w_up;
    LayerWeights w_down;

    const float* rms_norm_1_weight; // be4 attent
    const float* rms_norm_2_weight; // be4 MLP
};
