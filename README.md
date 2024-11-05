# ocf-iam4vp
Application of the Implicit Stacked Autoregressive Model for Video Prediction (IAM4VP) to cloudcasting


## Model summary

```
Settings:

- batch-size: 1
- hidden-channels-space: 32
- hidden-channels-time: 256
- num-convolutions-space: 4
- num-convolutions-time: 6
- num-forecast-steps: 6
- num-history-steps: 24
=========================================================================================================
Layer (type:depth-idx)                                  Output Shape              Param #
=========================================================================================================
IAM4VP                                                  [1, 11, 372, 614]         11,030,496
├─Encoder: 1-1                                          [24, 32, 93, 154]         --
│    └─Sequential: 2-1                                  --                        --
│    │    └─ConvSC: 3-1                                 [24, 32, 372, 614]        3,264
│    │    └─ConvSC: 3-2                                 [24, 32, 186, 307]        9,312
│    │    └─ConvSC: 3-3                                 [24, 32, 186, 307]        9,312
│    │    └─ConvSC: 3-4                                 [24, 32, 93, 154]         9,312
├─TimeMLP: 1-2                                          [1, 32]                   --
│    └─SinusoidalPosEmb: 2-2                            [1, 32]                   --
│    └─Linear: 2-3                                      [1, 128]                  4,224
│    └─GELU: 2-4                                        [1, 128]                  --
│    └─Linear: 2-5                                      [1, 32]                   4,128
├─Predictor: 1-3                                        [1, 24, 32, 93, 154]      --
│    └─Conv2d: 2-6                                      [1, 256, 93, 154]         393,472
│    └─Sequential: 2-7                                  --                        --
│    │    └─ConvNextTimeEmbed: 3-5                      [1, 256, 93, 154]         547,584
│    │    └─ConvNextTimeEmbedLKA: 3-6                   [1, 256, 93, 154]         632,832
│    │    └─ConvNextTimeEmbedLKA: 3-7                   [1, 256, 93, 154]         632,832
│    │    └─ConvNextTimeEmbedLKA: 3-8                   [1, 256, 93, 154]         632,832
│    │    └─ConvNextTimeEmbedLKA: 3-9                   [1, 256, 93, 154]         632,832
│    │    └─ConvNextTimeEmbedLKA: 3-10                  [1, 256, 93, 154]         632,832
│    └─Conv2d: 2-8                                      [1, 768, 93, 154]         197,376
├─Decoder: 1-4                                          [24, 11, 372, 614]        --
│    └─Sequential: 2-9                                  --                        --
│    │    └─ConvSC: 3-11                                [24, 32, 186, 308]        37,056
│    │    └─ConvSC: 3-12                                [24, 32, 186, 308]        9,312
│    │    └─ConvSC: 3-13                                [24, 32, 372, 616]        37,056
│    │    └─ConvSC: 3-14                                [24, 32, 372, 614]        18,528
│    └─Conv2d: 2-10                                     [24, 11, 372, 614]        363
├─SpatioTemporalRefinement: 1-5                         [1, 11, 372, 614]         --
│    └─Attention: 2-11                                  [1, 264, 372, 614]        --
│    │    └─Conv2d: 3-15                                [1, 264, 372, 614]        69,960
│    │    └─GELU: 3-16                                  [1, 264, 372, 614]        --
│    │    └─LargeKernelAttention: 3-17                  [1, 264, 372, 614]        90,024
│    │    └─Conv2d: 3-18                                [1, 264, 372, 614]        69,960
│    └─Conv2d: 2-12                                     [1, 11, 372, 614]         2,915
=========================================================================================================
Total params: 15,720,614
Trainable params: 15,720,614
Non-trainable params: 0
Total mult-adds (Units.GIGABYTES): 295.05
=========================================================================================================
Input size (MB): 241.20
Forward/backward pass size (MB): 15973.47
Params size (MB): 18.45
Estimated Total Size (MB): 16233.12
=========================================================================================================
```