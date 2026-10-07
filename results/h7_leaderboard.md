# H7 leaderboard (Nantes test partition of each split; published ResNet 0.663/0.701/0.371, ResNet-LSTM 0.685/0.696/0.559, ResNet-3D 0.705/0.735/0.659 for p/p_v/p_t)

| experiment                                                          |   seed | split                  |   params_M |   best_epoch |     p |   p_v |     r |   p_t |   n_test_videos |
|:--------------------------------------------------------------------|-------:|:-----------------------|-----------:|-------------:|------:|------:|------:|------:|----------------:|
| efficientnet_v2_l_none_imagesplit                                   |      0 | split0                 |    117.300 |           10 | 0.930 | 0.936 | 0.999 | 0.886 |             652 |
| resnet18_none_imagesplit                                            |      0 | split0                 |     11.200 |           10 | 0.876 | 0.889 | 0.998 | 0.792 |             652 |
| resnet18_transformer_L16_crossfocal7_evalfix_split4                 |      0 | split4                 |     17.200 |           10 | 0.760 | 0.764 | 0.989 | 0.783 |              67 |
| resnet18_transformer_L16_crossfocal7_evalfix_split3                 |      0 | split3                 |     17.200 |            9 | 0.743 | 0.756 | 0.992 | 0.762 |              68 |
| resnet18_transformer_L16_crossfocal7_evalfix_split2                 |      0 | split2                 |     17.200 |            8 | 0.746 | 0.745 | 0.989 | 0.745 |              65 |
| resnet50_lstm_L8_split0                                             |      1 | split0                 |     73.900 |            5 | 0.706 | 0.719 | 0.985 | 0.733 |              67 |
| resnet18_transformer_L16_crossfocal7_evalfix_split0                 |      0 | split0                 |     17.200 |           10 | 0.713 | 0.722 | 0.989 | 0.733 |              67 |
| r2plus1d_L8_split3                                                  |      0 | split3                 |     31.300 |            7 | 0.699 | 0.734 | 0.989 | 0.732 |              68 |
| resnet18_transformer_L16_crossfocal7_evalfix_split0                 |      2 | split0                 |     17.200 |           10 | 0.706 | 0.718 | 0.986 | 0.731 |              67 |
| resnet18_transformer_L16_multifocal3_evalfix_split0                 |      0 | split0                 |     16.200 |            9 | 0.714 | 0.718 | 0.989 | 0.730 |              67 |
| resnet18_lstm_L4_split4                                             |      0 | split4                 |     49.000 |            4 | 0.734 | 0.739 | 0.986 | 0.726 |              67 |
| resnet18_lstm_L4_multifocal3_split2                                 |      0 | split2                 |     49.000 |           10 | 0.740 | 0.744 | 0.990 | 0.726 |              65 |
| resnet18_lstm_L4_split2                                             |      0 | split2                 |     49.000 |            5 | 0.730 | 0.743 | 0.991 | 0.726 |              65 |
| resnet18_transformer_L16_multifocal3_photo_evalfix_split0           |      0 | split0                 |     16.200 |            9 | 0.712 | 0.721 | 0.989 | 0.726 |              67 |
| resnet50_lstm_L8_multifocal3_split0                                 |      0 | split0                 |     73.900 |            6 | 0.700 | 0.718 | 0.980 | 0.722 |              67 |
| resnet18_transformer_L16_crossfocal7_evalfix_split0                 |      1 | split0                 |     17.200 |           10 | 0.707 | 0.717 | 0.989 | 0.720 |              67 |
| resnet50_lstm_L8_split0                                             |      2 | split0                 |     73.900 |            7 | 0.709 | 0.721 | 0.982 | 0.720 |              67 |
| resnet18_lstm_L4_multifocal3_split4                                 |      0 | split4                 |     49.000 |            6 | 0.718 | 0.726 | 0.985 | 0.720 |              67 |
| resnet18_transformer_L16_evalfix_split0                             |      0 | split0                 |     16.200 |            9 | 0.704 | 0.711 | 0.988 | 0.719 |              67 |
| resnet18_transformer_L16_multifocal3_photo_cellcount_evalfix_split0 |      1 | split0                 |     16.200 |            9 | 0.699 | 0.706 | 0.988 | 0.719 |              67 |
| resnet50_lstm_L8_photo_cellcount_split0                             |      0 | split0                 |     73.900 |            6 | 0.709 | 0.717 | 0.979 | 0.718 |              67 |
| resnet18_transformer_L16_evalfix_split0                             |      1 | split0                 |     16.200 |           10 | 0.701 | 0.710 | 0.977 | 0.716 |              67 |
| resnet50_lstm_L8_photo_cellcount_split0                             |      1 | split0                 |     73.900 |            7 | 0.689 | 0.699 | 0.982 | 0.713 |              67 |
| resnet18_lstm_L8_hires448_split0                                    |      0 | split0                 |     49.000 |            7 | 0.701 | 0.710 | 0.984 | 0.712 |              67 |
| resnet18_lstm_L4_multifocal3_split0                                 |      2 | split0                 |     49.000 |            4 | 0.691 | 0.701 | 0.984 | 0.711 |              67 |
| resnet18_lstm_L4_multifocal3_split0                                 |      1 | split0                 |     49.000 |            9 | 0.710 | 0.716 | 0.983 | 0.710 |              67 |
| r2plus1d_L8_split4                                                  |      0 | split4                 |     31.300 |            2 | 0.721 | 0.746 | 0.989 | 0.709 |              67 |
| resnet18_transformer_L16_evalfix_split0                             |      2 | split0                 |     16.200 |            9 | 0.706 | 0.716 | 0.989 | 0.707 |              67 |
| resnet18_transformer_L16_multifocal3_cellcount_evalfix_split0       |      0 | split0                 |     16.200 |           10 | 0.703 | 0.712 | 0.987 | 0.707 |              67 |
| r2plus1d_L8_split0                                                  |      2 | split0                 |     31.300 |            9 | 0.676 | 0.709 | 0.988 | 0.706 |              67 |
| resnet18_transformer_L16_multifocal3_photo_cellcount_evalfix_split0 |      2 | split0                 |     16.200 |           10 | 0.695 | 0.707 | 0.988 | 0.705 |              67 |
| resnet50_lstm_L8_split0                                             |      0 | split0                 |     73.900 |            5 | 0.707 | 0.715 | 0.984 | 0.704 |              67 |
| resnet50_transformer_L16_evalfix_split0                             |      0 | split0                 |     29.300 |            8 | 0.699 | 0.710 | 0.982 | 0.704 |              67 |
| resnet18_lstm_L4_multifocal7_e20_split0                             |      0 | split0                 |     49.000 |           10 | 0.707 | 0.713 | 0.984 | 0.703 |              67 |
| resnet18_lstm_L4_multifocal3_split1                                 |      0 | split1                 |     49.000 |            7 | 0.712 | 0.725 | 0.984 | 0.703 |              61 |
| resnet18_transformer_L16_crossfocal7_evalfix_split1                 |      0 | split1                 |     17.200 |           10 | 0.702 | 0.715 | 0.985 | 0.703 |              61 |
| resnet50_lstm_L8_embryodiff_split                                   |      0 | nantes_random_73_seed0 |     73.900 |            7 | 0.700 | 0.707 | 0.985 | 0.700 |             196 |
| resnet50_lstm_L8_cellcount_split0                                   |      0 | split0                 |     73.900 |            4 | 0.706 | 0.715 | 0.978 | 0.700 |              67 |
| r2plus1d_L8_split0                                                  |      1 | split0                 |     31.300 |           10 | 0.679 | 0.693 | 0.986 | 0.698 |              67 |
| resnet18_lstm_L4_split1                                             |      0 | split1                 |     49.000 |            5 | 0.706 | 0.722 | 0.984 | 0.698 |              61 |
| resnet18_none_split4                                                |      0 | split4                 |     11.200 |            3 | 0.704 | 0.741 | 0.990 | 0.698 |              67 |
| resnet18_lstm_L4_split3                                             |      0 | split3                 |     49.000 |            9 | 0.708 | 0.718 | 0.989 | 0.697 |              68 |
| resnet18_transformer_L16_multifocal3_photo_cellcount_evalfix_split0 |      0 | split0                 |     16.200 |           10 | 0.708 | 0.719 | 0.988 | 0.696 |              67 |
| resnet18_tcn_L16_split0                                             |      0 | split0                 |     15.600 |           10 | 0.700 | 0.705 | 0.964 | 0.696 |              67 |
| resnet18_none_split3                                                |      0 | split3                 |     11.200 |           10 | 0.683 | 0.703 | 0.990 | 0.696 |              68 |
| resnet18_lstm_L4_multifocal3_split0                                 |      0 | split0                 |     49.000 |           10 | 0.715 | 0.718 | 0.984 | 0.696 |              67 |
| r2plus1d_L8_split0                                                  |      0 | split0                 |     31.300 |            5 | 0.689 | 0.708 | 0.988 | 0.695 |              67 |
| r2plus1d_L8_split2                                                  |      0 | split2                 |     31.300 |            5 | 0.696 | 0.720 | 0.985 | 0.694 |              65 |
| efficientnet_v2_l_none_split0                                       |      0 | split0                 |    117.300 |            8 | 0.686 | 0.722 | 0.986 | 0.694 |              67 |
| resnet18_lstm_L4_embryodiff_split                                   |      0 | nantes_random_73_seed0 |     49.000 |            6 | 0.696 | 0.709 | 0.984 | 0.691 |             196 |
| resnet18_lstm_L4_split0                                             |      0 | split0                 |     49.000 |            9 | 0.687 | 0.704 | 0.984 | 0.687 |              67 |
| resnet18_lstm_L4_multifocal3_split3                                 |      0 | split3                 |     49.000 |            9 | 0.703 | 0.717 | 0.989 | 0.687 |              68 |
| resnet18_lstm_L4_maeinit_split0                                     |      0 | split0                 |     49.000 |            7 | 0.689 | 0.686 | 0.981 | 0.685 |              67 |
| resnet18_mamba_L16_multifocal3_split0                               |      0 | split0                 |     14.600 |            7 | 0.693 | 0.698 | 0.985 | 0.682 |              67 |
| resnet50_lstm_L8_photo_split0                                       |      0 | split0                 |     73.900 |            9 | 0.691 | 0.697 | 0.982 | 0.682 |              67 |
| resnet18_none_split0                                                |      2 | split0                 |     11.200 |            7 | 0.666 | 0.694 | 0.980 | 0.680 |              67 |
| resnet50_lstm_L8_photo_cellcount_split0                             |      2 | split0                 |     73.900 |            5 | 0.676 | 0.699 | 0.985 | 0.679 |              67 |
| resnet50_lstm_L8_hires448_split0                                    |      0 | split0                 |     73.900 |            9 | 0.702 | 0.713 | 0.984 | 0.678 |              67 |
| r2plus1d_L8_split1                                                  |      0 | split1                 |     31.300 |            3 | 0.684 | 0.710 | 0.987 | 0.678 |              61 |
| resnet18_lstm_L8_crossfocal7_split0                                 |      0 | split0                 |     50.000 |            8 | 0.685 | 0.690 | 0.981 | 0.677 |              67 |
| resnet18_lstm_L4_split0                                             |      2 | split0                 |     49.000 |            5 | 0.679 | 0.694 | 0.983 | 0.676 |              67 |
| resnet18_lstm_L4_boundaryloss_split0                                |      1 | split0                 |     49.000 |            4 | 0.683 | 0.688 | 0.981 | 0.674 |              67 |
| resnet18_tcn_L16_split0                                             |      1 | split0                 |     15.600 |            8 | 0.696 | 0.697 | 0.981 | 0.673 |              67 |
| resnet18_none_split2                                                |      0 | split2                 |     11.200 |            3 | 0.700 | 0.729 | 0.990 | 0.673 |              65 |
| resnet18_lstm_L4_boundaryloss_split0                                |      0 | split0                 |     49.000 |            9 | 0.682 | 0.686 | 0.983 | 0.673 |              67 |
| resnet18_tcn_L16_split0                                             |      2 | split0                 |     15.600 |            7 | 0.690 | 0.694 | 0.987 | 0.670 |              67 |
| resnet18_transformer_L32_evalfix_split0                             |      0 | split0                 |     16.200 |            9 | 0.695 | 0.704 | 0.988 | 0.670 |              67 |
| resnet18_none_split0                                                |      1 | split0                 |     11.200 |            3 | 0.652 | 0.682 | 0.982 | 0.668 |              67 |
| resnet18_lstm_L4_ordinal_split0                                     |      0 | split0                 |     49.000 |            8 | 0.671 | 0.681 | 0.982 | 0.665 |              67 |
| resnet18_lstm_L4_grouped_v1                                         |      0 | nantes_grouped_v1      |     49.000 |            8 | 0.705 | 0.714 | 0.986 | 0.664 |              96 |
| resnet18_gru_L8_split0                                              |      0 | split0                 |     39.500 |            4 | 0.679 | 0.696 | 0.986 | 0.663 |              67 |
| resnet18_transformer_L64_evalfix_split0                             |      0 | split0                 |     16.200 |           10 | 0.681 | 0.689 | 0.984 | 0.659 |              67 |
| resnet18_lstm_L4_split0                                             |      1 | split0                 |     49.000 |            4 | 0.683 | 0.693 | 0.976 | 0.657 |              67 |
| resnet18_mamba_L16_split0                                           |      0 | split0                 |     14.600 |            8 | 0.688 | 0.688 | 0.987 | 0.651 |              67 |
| resnet18_mamba_L16_split0                                           |      2 | split0                 |     14.600 |            8 | 0.689 | 0.699 | 0.988 | 0.649 |              67 |
| resnet18_lstm_L4_boundaryloss_split0                                |      2 | split0                 |     49.000 |            5 | 0.664 | 0.675 | 0.980 | 0.647 |              67 |
| convnext_tiny_lstm_L8_split0                                        |      0 | split0                 |     67.700 |            9 | 0.691 | 0.689 | 0.982 | 0.645 |              67 |
| resnet18_mamba_L16_maeinit_split0                                   |      0 | split0                 |     14.600 |           10 | 0.678 | 0.688 | 0.983 | 0.643 |              67 |
| resnet50_mamba_L16_split0                                           |      0 | split0                 |     27.700 |           10 | 0.637 | 0.671 | 0.978 | 0.640 |              67 |
| resnet18_none_split1                                                |      0 | split1                 |     11.200 |            5 | 0.677 | 0.704 | 0.980 | 0.639 |              61 |
| resnet18_transformer_relpos_L16_split0                              |      2 | split0                 |     15.700 |            9 | 0.675 | 0.692 | 0.988 | 0.638 |              67 |
| resnet18_lstm_L4_embryodiff_split                                   |      1 | nantes_random_73_seed0 |     49.000 |            2 | 0.686 | 0.699 | 0.985 | 0.634 |             196 |
| resnet18_none_split0                                                |      0 | split0                 |     11.200 |            3 | 0.660 | 0.683 | 0.985 | 0.632 |              67 |
| resnet18_transformer_relpos_L16_split0                              |      0 | split0                 |     15.700 |            8 | 0.688 | 0.693 | 0.986 | 0.630 |              67 |
| resnet18_lstm_L4_ceweighted_split0                                  |      0 | split0                 |     49.000 |            3 | 0.673 | 0.686 | 0.983 | 0.627 |              67 |
| resnet18_lstm_L16_split0                                            |      0 | split0                 |     49.000 |            6 | 0.690 | 0.694 | 0.985 | 0.626 |              67 |
| resnet18_mamba_L16_split0                                           |      1 | split0                 |     14.600 |            5 | 0.667 | 0.681 | 0.986 | 0.621 |              67 |
| efficientnet_b0_lstm_L8_split0                                      |      0 | split0                 |     48.100 |            4 | 0.658 | 0.676 | 0.982 | 0.616 |              67 |
| resnet18_transformer_relpos_L16_split0                              |      1 | split0                 |     15.700 |            9 | 0.657 | 0.676 | 0.988 | 0.586 |              67 |
| resnet18_node_L16_split0                                            |      0 | split0                 |     11.400 |           10 | 0.645 | 0.677 | 0.986 | 0.575 |              67 |
| swin_t_lstm_L8_split0                                               |      0 | split0                 |     67.400 |            3 | 0.649 | 0.654 | 0.976 | 0.551 |              67 |
| resnet18_node_L16_split0                                            |      1 | split0                 |     11.400 |           10 | 0.567 | 0.561 | 0.960 | 0.480 |              67 |
| resnet18_diffactlite_L16_split0                                     |      0 | split0                 |     14.900 |            1 | 0.565 | 0.589 | 0.976 | 0.470 |              67 |
| resnet50_transformer_relpos_L16_split0                              |      0 | split0                 |     28.800 |            7 | 0.558 | 0.581 | 0.979 | 0.429 |              67 |
| resnet50_transformer_L16_crossfocal7_evalfix_split0                 |      0 | split0                 |     46.100 |            6 | 0.511 | 0.526 | 0.971 | 0.410 |              67 |
| resnet18_diffactlite_L16_split0                                     |      2 | split0                 |     14.900 |            9 | 0.487 | 0.521 | 0.968 | 0.376 |              67 |
| resnet18_diffactlite_L16_split0                                     |      1 | split0                 |     14.900 |            4 | 0.528 | 0.531 | 0.948 | 0.371 |              67 |
| kinetic_cnn_lstm_L4_split0                                          |      0 | split0                 |     38.100 |            6 | 0.583 | 0.580 | 0.978 | 0.369 |              67 |
| resnet18_transformer_L16_split0                                     |      0 | split0                 |     16.200 |           10 | 0.512 | 0.492 | 0.962 | 0.369 |              67 |
| resnet18_transformer_L16_adamw_split0                               |      0 | split0                 |     16.200 |            6 | 0.510 | 0.494 | 0.964 | 0.360 |              67 |
| resnet18_node_L16_split0                                            |      2 | split0                 |     11.400 |           10 | 0.412 | 0.421 | 0.925 | 0.190 |              67 |

## Paired differences vs `resnet18_lstm_L4_split0` (same test videos; mean and 95 % bootstrap CI over videos)

| experiment | Δ acc_viterbi | Δ temporal acc |
|---|---|---|
| convnext_tiny_lstm_L8_split0 (seed 0, n=67) | -0.006 [-0.033, +0.019] | -0.041 [-0.078, -0.003] |
| efficientnet_b0_lstm_L8_split0 (seed 0, n=67) | -0.019 [-0.048, +0.008] | -0.070 [-0.106, -0.033] |
| efficientnet_v2_l_none_imagesplit (seed 0, n=67) | +0.215 [+0.180, +0.251] | +0.180 [+0.132, +0.229] |
| efficientnet_v2_l_none_split0 (seed 0, n=67) | +0.020 [-0.004, +0.045] | +0.006 [-0.027, +0.037] |
| kinetic_cnn_lstm_L4_split0 (seed 0, n=67) | -0.137 [-0.168, -0.107] | -0.315 [-0.361, -0.267] |
| r2plus1d_L8_split0 (seed 0, n=67) | +0.014 [-0.011, +0.039] | +0.005 [-0.025, +0.037] |
| r2plus1d_L8_split0 (seed 1, n=67) | +0.000 [-0.022, +0.021] | +0.011 [-0.026, +0.049] |
| r2plus1d_L8_split0 (seed 2, n=67) | +0.015 [-0.006, +0.035] | +0.019 [-0.010, +0.047] |
| resnet18_diffactlite_L16_split0 (seed 0, n=67) | -0.115 [-0.150, -0.081] | -0.217 [-0.265, -0.168] |
| resnet18_diffactlite_L16_split0 (seed 1, n=67) | -0.175 [-0.212, -0.140] | -0.316 [-0.366, -0.266] |
| resnet18_diffactlite_L16_split0 (seed 2, n=67) | -0.185 [-0.222, -0.148] | -0.312 [-0.358, -0.259] |
| resnet18_gru_L8_split0 (seed 0, n=67) | -0.004 [-0.023, +0.015] | -0.023 [-0.055, +0.011] |
| resnet18_lstm_L16_split0 (seed 0, n=67) | -0.004 [-0.027, +0.018] | -0.065 [-0.103, -0.027] |
| resnet18_lstm_L4_boundaryloss_split0 (seed 0, n=67) | -0.013 [-0.028, +0.003] | -0.015 [-0.042, +0.013] |
| resnet18_lstm_L4_boundaryloss_split0 (seed 1, n=67) | -0.011 [-0.032, +0.009] | -0.013 [-0.046, +0.017] |
| resnet18_lstm_L4_boundaryloss_split0 (seed 2, n=67) | -0.023 [-0.048, +0.000] | -0.041 [-0.079, -0.004] |
| resnet18_lstm_L4_ceweighted_split0 (seed 0, n=67) | -0.014 [-0.033, +0.004] | -0.060 [-0.091, -0.027] |
| resnet18_lstm_L4_embryodiff_split (seed 0, n=16) | -0.035 [-0.080, +0.012] | -0.005 [-0.073, +0.063] |
| resnet18_lstm_L4_embryodiff_split (seed 1, n=16) | +0.010 [-0.030, +0.050] | -0.021 [-0.103, +0.075] |
| resnet18_lstm_L4_grouped_v1 (seed 0, n=14) | +0.002 [-0.046, +0.053] | -0.019 [-0.095, +0.057] |
| resnet18_lstm_L4_maeinit_split0 (seed 0, n=67) | -0.009 [-0.035, +0.017] | -0.002 [-0.038, +0.033] |
| resnet18_lstm_L4_multifocal3_split0 (seed 0, n=67) | +0.019 [-0.000, +0.038] | +0.008 [-0.025, +0.038] |
| resnet18_lstm_L4_multifocal3_split0 (seed 1, n=67) | +0.016 [-0.007, +0.039] | +0.023 [-0.014, +0.061] |
| resnet18_lstm_L4_multifocal3_split0 (seed 2, n=67) | +0.007 [-0.018, +0.031] | +0.023 [-0.012, +0.057] |
| resnet18_lstm_L4_multifocal7_e20_split0 (seed 0, n=67) | +0.015 [-0.009, +0.038] | +0.013 [-0.023, +0.047] |
| resnet18_lstm_L4_ordinal_split0 (seed 0, n=67) | -0.021 [-0.041, -0.001] | -0.021 [-0.053, +0.014] |
| resnet18_lstm_L4_split0 (seed 1, n=67) | -0.003 [-0.024, +0.019] | -0.031 [-0.069, +0.002] |
| resnet18_lstm_L4_split0 (seed 2, n=67) | -0.001 [-0.024, +0.022] | -0.014 [-0.046, +0.021] |
| resnet18_lstm_L8_crossfocal7_split0 (seed 0, n=67) | -0.006 [-0.029, +0.017] | -0.010 [-0.040, +0.020] |
| resnet18_lstm_L8_hires448_split0 (seed 0, n=67) | +0.015 [-0.007, +0.037] | +0.025 [-0.010, +0.062] |
| resnet18_mamba_L16_maeinit_split0 (seed 0, n=67) | -0.013 [-0.033, +0.008] | -0.044 [-0.080, -0.006] |
| resnet18_mamba_L16_multifocal3_split0 (seed 0, n=67) | +0.003 [-0.020, +0.026] | -0.005 [-0.039, +0.029] |
| resnet18_mamba_L16_split0 (seed 0, n=67) | -0.008 [-0.032, +0.015] | -0.036 [-0.075, +0.003] |
| resnet18_mamba_L16_split0 (seed 1, n=67) | -0.015 [-0.035, +0.004] | -0.066 [-0.104, -0.028] |
| resnet18_mamba_L16_split0 (seed 2, n=67) | +0.001 [-0.020, +0.022] | -0.039 [-0.072, -0.006] |
| resnet18_node_L16_split0 (seed 0, n=67) | -0.027 [-0.049, -0.004] | -0.113 [-0.155, -0.069] |
| resnet18_node_L16_split0 (seed 1, n=67) | -0.136 [-0.187, -0.091] | -0.207 [-0.262, -0.154] |
| resnet18_node_L16_split0 (seed 2, n=67) | -0.298 [-0.339, -0.256] | -0.497 [-0.545, -0.446] |
| resnet18_none_imagesplit (seed 0, n=67) | +0.174 [+0.140, +0.209] | +0.115 [+0.068, +0.164] |
| resnet18_none_split0 (seed 0, n=67) | -0.019 [-0.043, +0.004] | -0.056 [-0.096, -0.015] |
| resnet18_none_split0 (seed 1, n=67) | -0.021 [-0.041, -0.002] | -0.020 [-0.053, +0.016] |
| resnet18_none_split0 (seed 2, n=67) | -0.012 [-0.033, +0.010] | -0.007 [-0.039, +0.029] |
| resnet18_tcn_L16_split0 (seed 0, n=67) | +0.008 [-0.010, +0.026] | +0.009 [-0.021, +0.036] |
| resnet18_tcn_L16_split0 (seed 1, n=67) | +0.002 [-0.022, +0.026] | -0.020 [-0.062, +0.019] |
| resnet18_tcn_L16_split0 (seed 2, n=67) | -0.004 [-0.024, +0.017] | -0.018 [-0.050, +0.015] |
| resnet18_transformer_L16_adamw_split0 (seed 0, n=67) | -0.223 [-0.258, -0.186] | -0.321 [-0.366, -0.275] |
| resnet18_transformer_L16_crossfocal7_evalfix_split0 (seed 0, n=67) | +0.024 [+0.001, +0.046] | +0.046 [+0.015, +0.074] |
| resnet18_transformer_L16_crossfocal7_evalfix_split0 (seed 1, n=67) | +0.023 [-0.001, +0.047] | +0.033 [+0.003, +0.062] |
| resnet18_transformer_L16_crossfocal7_evalfix_split0 (seed 2, n=67) | +0.018 [-0.005, +0.041] | +0.044 [+0.013, +0.075] |
| resnet18_transformer_L16_evalfix_split0 (seed 0, n=67) | +0.013 [-0.005, +0.032] | +0.032 [+0.003, +0.065] |
| resnet18_transformer_L16_evalfix_split0 (seed 1, n=67) | +0.012 [-0.009, +0.033] | +0.028 [-0.004, +0.061] |
| resnet18_transformer_L16_evalfix_split0 (seed 2, n=67) | +0.017 [-0.003, +0.036] | +0.020 [-0.010, +0.050] |
| resnet18_transformer_L16_multifocal3_cellcount_evalfix_split0 (seed 0, n=67) | +0.013 [-0.008, +0.033] | +0.020 [-0.011, +0.049] |
| resnet18_transformer_L16_multifocal3_evalfix_split0 (seed 0, n=67) | +0.023 [-0.000, +0.045] | +0.042 [+0.009, +0.076] |
| resnet18_transformer_L16_multifocal3_photo_cellcount_evalfix_split0 (seed 0, n=67) | +0.017 [-0.005, +0.040] | +0.009 [-0.027, +0.043] |
| resnet18_transformer_L16_multifocal3_photo_cellcount_evalfix_split0 (seed 1, n=67) | +0.011 [-0.013, +0.033] | +0.032 [-0.000, +0.063] |
| resnet18_transformer_L16_multifocal3_photo_cellcount_evalfix_split0 (seed 2, n=67) | +0.010 [-0.015, +0.034] | +0.018 [-0.018, +0.051] |
| resnet18_transformer_L16_multifocal3_photo_evalfix_split0 (seed 0, n=67) | +0.026 [+0.005, +0.046] | +0.038 [+0.003, +0.070] |
| resnet18_transformer_L16_split0 (seed 0, n=67) | -0.217 [-0.253, -0.180] | -0.312 [-0.358, -0.260] |
| resnet18_transformer_L32_evalfix_split0 (seed 0, n=67) | +0.006 [-0.015, +0.026] | -0.018 [-0.047, +0.011] |
| resnet18_transformer_L64_evalfix_split0 (seed 0, n=67) | -0.008 [-0.041, +0.023] | -0.029 [-0.070, +0.009] |
| resnet18_transformer_relpos_L16_split0 (seed 0, n=67) | -0.005 [-0.023, +0.013] | -0.057 [-0.085, -0.031] |
| resnet18_transformer_relpos_L16_split0 (seed 1, n=67) | -0.027 [-0.051, -0.002] | -0.101 [-0.141, -0.062] |
| resnet18_transformer_relpos_L16_split0 (seed 2, n=67) | -0.007 [-0.029, +0.015] | -0.049 [-0.081, -0.017] |
| resnet50_lstm_L8_cellcount_split0 (seed 0, n=67) | +0.016 [-0.003, +0.035] | +0.012 [-0.022, +0.046] |
| resnet50_lstm_L8_embryodiff_split (seed 0, n=16) | -0.003 [-0.033, +0.025] | +0.030 [-0.034, +0.109] |
| resnet50_lstm_L8_hires448_split0 (seed 0, n=67) | +0.012 [-0.010, +0.034] | -0.009 [-0.036, +0.016] |
| resnet50_lstm_L8_multifocal3_split0 (seed 0, n=67) | +0.020 [-0.000, +0.041] | +0.035 [-0.003, +0.074] |
| resnet50_lstm_L8_photo_cellcount_split0 (seed 0, n=67) | +0.018 [-0.001, +0.036] | +0.030 [+0.002, +0.060] |
| resnet50_lstm_L8_photo_cellcount_split0 (seed 1, n=67) | -0.002 [-0.024, +0.019] | +0.025 [-0.004, +0.053] |
| resnet50_lstm_L8_photo_cellcount_split0 (seed 2, n=67) | -0.001 [-0.025, +0.022] | -0.009 [-0.043, +0.028] |
| resnet50_lstm_L8_photo_split0 (seed 0, n=67) | -0.002 [-0.025, +0.020] | -0.005 [-0.042, +0.032] |
| resnet50_lstm_L8_split0 (seed 0, n=67) | +0.014 [-0.006, +0.033] | +0.017 [-0.013, +0.045] |
| resnet50_lstm_L8_split0 (seed 1, n=67) | +0.020 [-0.001, +0.041] | +0.046 [+0.014, +0.079] |
| resnet50_lstm_L8_split0 (seed 2, n=67) | +0.026 [+0.009, +0.042] | +0.034 [+0.005, +0.063] |
| resnet50_mamba_L16_split0 (seed 0, n=67) | -0.026 [-0.057, +0.004] | -0.048 [-0.088, -0.011] |
| resnet50_transformer_L16_crossfocal7_evalfix_split0 (seed 0, n=67) | -0.183 [-0.219, -0.149] | -0.277 [-0.325, -0.231] |
| resnet50_transformer_L16_evalfix_split0 (seed 0, n=67) | +0.012 [-0.007, +0.030] | +0.017 [-0.018, +0.053] |
| resnet50_transformer_relpos_L16_split0 (seed 0, n=67) | -0.129 [-0.162, -0.097] | -0.258 [-0.313, -0.204] |
| swin_t_lstm_L8_split0 (seed 0, n=67) | -0.049 [-0.076, -0.023] | -0.138 [-0.180, -0.092] |
