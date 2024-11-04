# ocf-iam4vp
Application of the Implicit Stacked Autoregressive Model for Video Prediction (IAM4VP) to cloudcasting


## Model summary

```
Settings:

- batch-size: 1
- history-steps: 12
- forecast-steps: 6
- latent-space-channels: 64
- num-convolutions-space: 4
- num-convolutions-time: 6

====================================================================================================
Layer (type:depth-idx)                             Output Shape              Param #
====================================================================================================
IAM4VP                                             [1, 11, 372, 614]         11,116,992
├─Encoder: 1-1                                     [12, 64, 93, 154]         --
│    └─Sequential: 2-1                             --                        --
│    │    └─ConvSC: 3-1                            [12, 64, 372, 614]        6,528
│    │    └─ConvSC: 3-2                            [12, 64, 186, 307]        37,056
│    │    └─ConvSC: 3-3                            [12, 64, 186, 307]        37,056
│    │    └─ConvSC: 3-4                            [12, 64, 93, 154]         37,056
├─TimeMLP: 1-2                                     [1, 64]                   --
│    └─SinusoidalPosEmb: 2-2                       [1, 64]                   --
│    └─Linear: 2-3                                 [1, 256]                  16,640
│    └─GELU: 2-4                                   [1, 256]                  --
│    └─Linear: 2-5                                 [1, 64]                   16,448
├─Predictor: 1-3                                   [1, 12, 64, 93, 154]      --
│    └─Sequential: 2-6                             --                        --
│    │    └─ConvNeXt_bottle: 3-5                   [1, 768, 93, 154]         6,031,104
│    │    └─ConvNeXt_block: 3-6                    [1, 768, 93, 154]         5,423,616
│    │    └─ConvNeXt_block: 3-7                    [1, 768, 93, 154]         5,423,616
│    │    └─ConvNeXt_block: 3-8                    [1, 768, 93, 154]         5,423,616
│    │    └─ConvNeXt_block: 3-9                    [1, 768, 93, 154]         5,423,616
│    │    └─ConvNeXt_block: 3-10                   [1, 768, 93, 154]         5,423,616
├─Decoder: 1-4                                     [12, 11, 372, 614]        --
│    └─Sequential: 2-7                             --                        --
│    │    └─ConvSC: 3-11                           [12, 64, 186, 308]        147,840
│    │    └─ConvSC: 3-12                           [12, 64, 186, 308]        37,056
│    │    └─ConvSC: 3-13                           [12, 64, 372, 616]        147,840
│    │    └─ConvSC: 3-14                           [12, 64, 372, 614]        73,920
│    └─Conv2d: 2-8                                 [12, 11, 372, 614]        715
├─SpatioTemporalRefinement: 1-5                    [1, 11, 372, 614]         --
│    └─Attention: 2-9                              [1, 132, 372, 614]        --
│    │    └─Conv2d: 3-15                           [1, 132, 372, 614]        17,556
│    │    └─GELU: 3-16                             [1, 132, 372, 614]        --
│    │    └─LKA: 3-17                              [1, 132, 372, 614]        27,588
│    │    └─Conv2d: 3-18                           [1, 132, 372, 614]        17,556
│    └─Conv2d: 2-10                                [1, 11, 372, 614]         1,463
====================================================================================================
Total params: 44,888,494
Trainable params: 44,888,494
Non-trainable params: 0
Total mult-adds (Units.GIGABYTES): 510.18
====================================================================================================
Input size (MB): 120.60
Forward/backward pass size (MB): 17547.45
Params size (MB): 135.07
Estimated Total Size (MB): 17803.11
====================================================================================================
```