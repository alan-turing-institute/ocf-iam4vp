# ocf-iam4vp
Application of the Implicit Stacked Autoregressive Model for Video Prediction (IAM4VP) to cloudcasting


## Model summary

The settings below are being used to train this model

```
- batch-size: 2
- hidden-channels-space: 32
- hidden-channels-time: 64
- num-convolutions-space: 4
- num-convolutions-time: 4
- num-forecast-steps: 12
- num-history-steps: 24
=========================================================================================================
Layer (type:depth-idx)                                  Output Shape              Param #
=========================================================================================================
IAM4VP                                                  [2, 11, 372, 614]         10,999,296
├─Encoder: 1-1                                          [48, 32, 93, 154]         --
│    └─Sequential: 2-1                                  --                        --
│    │    └─ConvSC: 3-1                                 [48, 32, 372, 614]        3,264
│    │    └─ConvSC: 3-2                                 [48, 32, 186, 307]        9,312
│    │    └─ConvSC: 3-3                                 [48, 32, 186, 307]        9,312
│    │    └─ConvSC: 3-4                                 [48, 32, 93, 154]         9,312
├─TimeMLP: 1-2                                          [2, 32]                   --
│    └─SinusoidalPosEmb: 2-2                            [2, 32]                   --
│    └─Linear: 2-3                                      [2, 128]                  4,224
│    └─GELU: 2-4                                        [2, 128]                  --
│    └─Linear: 2-5                                      [2, 32]                   4,128
├─Predictor: 1-3                                        [2, 24, 32, 93, 154]      --
│    └─Conv2d: 2-6                                      [2, 64, 93, 154]          98,368
│    └─Sequential: 2-7                                  --                        --
│    │    └─ConvNextTimeEmbed: 3-5                      [2, 64, 93, 154]          38,592
│    │    └─ConvNextTimeEmbedLKA: 3-6                   [2, 64, 93, 154]          47,616
│    │    └─ConvNextTimeEmbedLKA: 3-7                   [2, 64, 93, 154]          47,616
│    │    └─ConvNextTimeEmbedLKA: 3-8                   [2, 64, 93, 154]          47,616
│    └─Conv2d: 2-8                                      [2, 768, 93, 154]         49,920
├─Decoder: 1-4                                          [48, 11, 372, 614]        --
│    └─Sequential: 2-9                                  --                        --
│    │    └─ConvSC: 3-9                                 [48, 32, 186, 308]        37,056
│    │    └─ConvSC: 3-10                                [48, 32, 186, 308]        9,312
│    │    └─ConvSC: 3-11                                [48, 32, 372, 616]        37,056
│    │    └─ConvSC: 3-12                                [48, 32, 372, 614]        18,528
│    └─Conv2d: 2-10                                     [48, 11, 372, 614]        363
├─SpatioTemporalRefinement: 1-5                         [2, 11, 372, 614]         --
│    └─Attention: 2-11                                  [2, 264, 372, 614]        --
│    │    └─Conv2d: 3-13                                [2, 264, 372, 614]        69,960
│    │    └─GELU: 3-14                                  [2, 264, 372, 614]        --
│    │    └─LargeKernelAttention: 3-15                  [2, 264, 372, 614]        90,024
│    │    └─Conv2d: 3-16                                [2, 264, 372, 614]        69,960
│    └─Conv2d: 2-12                                     [2, 11, 372, 614]         2,915
├─Sigmoid: 1-6                                          [2, 11, 372, 614]         --
=========================================================================================================
Total params: 11,706,950
Trainable params: 11,706,950
Non-trainable params: 0
Total mult-adds (Units.GIGABYTES): 565.48
=========================================================================================================
Input size (MB): 482.40
Forward/backward pass size (MB): 29351.09
Params size (MB): 2.78
Estimated Total Size (MB): 29836.27
=========================================================================================================
```
