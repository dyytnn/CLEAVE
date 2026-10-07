# TEMPO diagnostics: cross-run comparison

| run | p | p_v | p_t | Edit | F1@50 | MAE h | err@boundary(0-2) | err@interior(>20) | adj. share | under-pred share | probe bb | probe hd | sil bb | sil hd | ECE | H_bnd/H_int |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| resnet18_transformer_L16_crossfocal7_evalfix_split0_seed0 | 0.713 | 0.722 | 0.733 | 79.7 | 63.7 | 2.79 | 0.438 | 0.211 | 0.67 | 0.49 | 0.643 | 0.678 | 0.085 | 0.132 | 0.047 | 0.82/0.62 |
| resnet18_lstm_L8_hires448_split0_seed0 | 0.701 | 0.710 | 0.712 | 81.8 | 60.6 | 2.96 | 0.467 | 0.230 | 0.69 | 0.52 | 0.603 | 0.692 | 0.045 | 0.145 | 0.116 | 0.69/0.45 |

## Timing bias per event (h, pred - gt, median signed)

| run | tPB2 | tPNa | tPNf | t2 | t3 | t4 | t5 | t6 | t7 | t8 | t9+ | tM | tSB | tB | tEB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| resnet18_transformer_L16_crossfocal7_evalfix_split0_seed0 | +0.00 | +0.20 | +0.00 | +0.00 | +0.00 | -0.30 | +0.20 | -0.30 | +0.00 | -0.50 | -0.80 | -0.05 | -0.40 | +0.50 | -0.80 |
| resnet18_lstm_L8_hires448_split0_seed0 | +0.00 | +0.00 | -0.30 | +0.30 | +0.20 | +0.00 | +0.50 | -0.50 | -0.30 | +0.00 | -0.40 | -0.30 | +0.20 | -1.00 | -0.80 |

## within-θ as-is → if per-event median bias were removed

| run | tPB2 | tPNa | tPNf | t2 | t3 | t4 | t5 | t6 | t7 | t8 | t9+ | tM | tSB | tB | tEB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| resnet18_transformer_L16_crossfocal7_evalfix_split0_seed0 | --→-- | 0.58→0.56 | 0.94→0.94 | 0.95→0.95 | 0.84→0.84 | 0.84→0.83 | 0.71→0.71 | 0.61→0.57 | 0.73→0.73 | 0.67→0.67 | 0.68→0.64 | 0.76→0.76 | 0.86→0.86 | 0.65→0.65 | 0.70→0.76 |
| resnet18_lstm_L8_hires448_split0_seed0 | --→-- | 0.70→0.70 | 0.78→0.84 | 0.89→0.95 | 0.89→0.89 | 0.78→0.78 | 0.54→0.60 | 0.71→0.69 | 0.73→0.73 | 0.74→0.74 | 0.64→0.66 | 0.67→0.69 | 0.73→0.73 | 0.75→0.80 | 0.79→0.77 |
