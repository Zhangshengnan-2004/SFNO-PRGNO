| Dataset | Model | Params (M) | Core ×4 (G) | Core ×8 (G) | Major estimate ×4 (G) | Major estimate ×8 (G) |
|---|---|---:|---:|---:|---:|---:|
| cave | SFNO | 0.244645 | 2.4820 | 2.4820 | 2.4865 | 2.4865 |
| cave | SPGFNet (V3-stage3) | 0.614036 | 3.5386 | 3.5117 | 3.6767 | 3.6476 |
| cave | SSR-NET | 0.026041 | 0.2126 | 0.2126 | 0.2134 | 0.2134 |
| pavia | SFNO | 0.450020 | 8.1187 | 8.1187 | 8.1333 | 8.1333 |
| pavia | SPGFNet (V3-stage3) | 0.828570 | 9.4141 | 9.3872 | 9.7717 | 9.7426 |
| pavia | SSR-NET | 0.281215 | 2.3012 | 2.3012 | 2.3041 | 2.3041 |
| chikusei | SFNO | 0.565327 | 11.3898 | 11.3898 | 11.4081 | 11.4081 |
| chikusei | SPGFNet (V3-stage3) | 0.947231 | 12.7658 | 12.7389 | 13.2038 | 13.1748 |
| chikusei | SSR-NET | 0.442753 | 3.6239 | 3.6239 | 3.6275 | 3.6275 |

Batch=1, HR=64×64, FP32 CPU, multiply-add=2 FLOPs. Core: PyTorch 2.6 FlopCounterMode supported operators (mainly convolution/matrix products). Major estimate adds approximate FFT, interpolation and grid sampling costs. Neither column is a complete FLOPs count; elementwise, normalization, activation, reduction, softmax, pooling and indexing costs are not fully included. details.json lists unaccounted operators. Warm forward excludes cached geometry setup. Weights do not affect these static counts. SPGFNet is mapped to deform_stage3. Params includes registered but unused modules.
