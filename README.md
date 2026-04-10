# Addition Lab

**Task:** The encoder reads a string such as “15+08” and the decoder outputs the sum “23”. This task adds difficulty because the sequence length may vary, and the decoder must learn when to stop. 

**Procedure**
- Generate addition st rings such as 03+05, 11+07, 15+08
- Train the model to output the correct sum
- Test on unseen examples
- Observe if the model learns both arithmetic structure and stopping condition

## Usage
- Set up: `pipenv install`, `pipenv shell`
- To train,
  ```
  python main.py 

  args:
  --hidden_size, --epochs, --train_size, --val_size, --test_size, --patience, --lr
  --model_path, --history_path, --hidden_input
  --norm_padding 
  ```

- To evaluate manually, 
  ```
  python test.py --test

  args:
  --test # To test in REPL
  --test_all # To test all possible 0+0/00+00 combinations
  --plots # Generate plots
  --model_path, history_path, hidden_input
  --norm_padding
  ```
## Training
- Notes
  - The base code accounts for inputs in this kind of format `1+1`. To test all inputs in this format `01+01`, use the `--norm_padding` flag
  - Early stopping is employed
  - All trainings recorded in this repo follow the params:
    ```
    HIDDEN_SIZE = 128
    EPOCHS      = 100
    PATIENCE    = 12 
    LR          = 0.01 
    TRAIN_SIZE  = 8000 
    VAL_SIZE    = 1000  
    TEST_SIZE   = 1000
    ```
- 1+1 Mode
  ```
  Epoch 001 | loss=4.5152 | val_exact=0.0550
  Epoch 002 | loss=3.7500 | val_exact=0.0510
  Epoch 003 | loss=3.3540 | val_exact=0.0980
  Epoch 004 | loss=2.7707 | val_exact=0.2490
  Epoch 005 | loss=2.3764 | val_exact=0.2610
  Epoch 006 | loss=2.1538 | val_exact=0.3960
  Epoch 007 | loss=1.9516 | val_exact=0.4280
  Epoch 008 | loss=1.8083 | val_exact=0.3780
  Epoch 009 | loss=1.6358 | val_exact=0.3960
  Epoch 010 | loss=1.4082 | val_exact=0.5030
  Epoch 011 | loss=1.3619 | val_exact=0.4740
  Epoch 012 | loss=1.2972 | val_exact=0.4570
  Epoch 013 | loss=1.1996 | val_exact=0.6190
  Epoch 014 | loss=1.0568 | val_exact=0.5370
  Epoch 015 | loss=0.9800 | val_exact=0.6280
  Epoch 016 | loss=0.9167 | val_exact=0.6920
  Epoch 017 | loss=0.8340 | val_exact=0.7270
  Epoch 018 | loss=0.7309 | val_exact=0.6910
  Epoch 019 | loss=0.6740 | val_exact=0.8080
  Epoch 020 | loss=0.5830 | val_exact=0.7800
  Epoch 021 | loss=0.4602 | val_exact=0.8230
  Epoch 022 | loss=0.3838 | val_exact=0.8390
  Epoch 023 | loss=0.3324 | val_exact=0.8660
  Epoch 024 | loss=0.3030 | val_exact=0.8520
  Epoch 025 | loss=0.2449 | val_exact=0.8750
  Epoch 026 | loss=0.2978 | val_exact=0.8520
  Epoch 027 | loss=0.3193 | val_exact=0.8880
  Epoch 028 | loss=0.2574 | val_exact=0.8960
  Epoch 029 | loss=0.2528 | val_exact=0.8850
  Epoch 030 | loss=0.1816 | val_exact=0.9000
  Epoch 031 | loss=0.1646 | val_exact=0.9170
  Epoch 032 | loss=0.1191 | val_exact=0.9390
  Epoch 033 | loss=0.0841 | val_exact=0.9520
  Epoch 034 | loss=0.0537 | val_exact=0.9530
  Epoch 035 | loss=0.0373 | val_exact=0.9530
  Epoch 036 | loss=0.0302 | val_exact=0.9570
  Epoch 037 | loss=0.0199 | val_exact=0.9600
  Epoch 038 | loss=0.0145 | val_exact=0.9620
  Epoch 039 | loss=0.0109 | val_exact=0.9620
  Epoch 040 | loss=0.0092 | val_exact=0.9640
  Epoch 041 | loss=0.0084 | val_exact=0.9610
  Epoch 042 | loss=0.0077 | val_exact=0.9640
  Epoch 043 | loss=0.0072 | val_exact=0.9640
  Epoch 044 | loss=0.0067 | val_exact=0.9620
  Epoch 045 | loss=0.0063 | val_exact=0.9610
  Epoch 046 | loss=0.0060 | val_exact=0.9620
  Epoch 047 | loss=0.0057 | val_exact=0.9630
  Epoch 048 | loss=0.0054 | val_exact=0.9650
  Epoch 049 | loss=0.0051 | val_exact=0.9660
  Epoch 050 | loss=0.0049 | val_exact=0.9630
  Epoch 051 | loss=0.0047 | val_exact=0.9650
  Epoch 052 | loss=0.0045 | val_exact=0.9630
  Epoch 053 | loss=0.0043 | val_exact=0.9640
  Epoch 054 | loss=0.0042 | val_exact=0.9640
  Epoch 055 | loss=0.0040 | val_exact=0.9650
  Epoch 056 | loss=0.0039 | val_exact=0.9640
  Epoch 057 | loss=0.0038 | val_exact=0.9660
  Epoch 058 | loss=0.0037 | val_exact=0.9650
  Epoch 059 | loss=0.0035 | val_exact=0.9650
  Epoch 060 | loss=0.0034 | val_exact=0.9650
  Epoch 061 | loss=0.0033 | val_exact=0.9660
  Early stopping triggered.
  
  Best validation accuracy: 0.9660 at epoch 49
  Test accuracy: 0.9760
  Model saved to outputs/secret_accountant_model.npz
  History saved to outputs/training_history.npz
  
  Saved model to: outputs/secret_accountant_model.npz
  Saved history to: outputs/training_history.npz
  
  Validation examples:
  
  Examples:
  IN: 93+6   TARGET: 99   PRED: 100
  IN: 85+4   TARGET: 89   PRED: 89
  IN: 89+68   TARGET: 157   PRED: 157
  IN: 95+59   TARGET: 154   PRED: 154
  IN: 68+90   TARGET: 158   PRED: 158
  IN: 63+45   TARGET: 108   PRED: 108
  IN: 50+76   TARGET: 126   PRED: 126
  IN: 95+38   TARGET: 133   PRED: 133
  IN: 51+52   TARGET: 103   PRED: 103
  IN: 42+94   TARGET: 136   PRED: 136
  IN: 13+96   TARGET: 109   PRED: 109
  IN: 39+35   TARGET: 74   PRED: 74
  
  Test examples:
  
  Examples:
  IN: 27+25   TARGET: 52   PRED: 52
  IN: 72+69   TARGET: 141   PRED: 141
  IN: 16+46   TARGET: 62   PRED: 62
  IN: 97+74   TARGET: 171   PRED: 171
  IN: 71+10   TARGET: 81   PRED: 81
  IN: 14+80   TARGET: 94   PRED: 94
  IN: 92+54   TARGET: 146   PRED: 146
  IN: 47+8   TARGET: 55   PRED: 55
  IN: 38+46   TARGET: 84   PRED: 84
  IN: 11+21   TARGET: 32   PRED: 32
  IN: 5+88   TARGET: 93   PRED: 93
  IN: 88+64   TARGET: 152   PRED: 152
  Saved loss plot to outputs/loss_vs_iteration.png
  Saved hidden state heatmap to outputs/hidden_state_heatmap.png
  ```

- 01+01 Mode
  ```
  Epoch 001 | loss=4.5025 | val_exact=0.0400
  Epoch 002 | loss=3.6413 | val_exact=0.1170
  Epoch 003 | loss=2.7652 | val_exact=0.1980
  Epoch 004 | loss=2.3249 | val_exact=0.2940
  Epoch 005 | loss=2.0092 | val_exact=0.3580
  Epoch 006 | loss=1.6665 | val_exact=0.3670
  Epoch 007 | loss=1.3939 | val_exact=0.5180
  Epoch 008 | loss=1.2410 | val_exact=0.6430
  Epoch 009 | loss=1.0112 | val_exact=0.4660
  Epoch 010 | loss=0.9008 | val_exact=0.6120
  Epoch 011 | loss=0.7193 | val_exact=0.7550
  Epoch 012 | loss=0.5674 | val_exact=0.8300
  Epoch 013 | loss=0.6275 | val_exact=0.7880
  Epoch 014 | loss=0.5910 | val_exact=0.7060
  Epoch 015 | loss=0.5077 | val_exact=0.8510
  Epoch 016 | loss=0.2310 | val_exact=0.9470
  Epoch 017 | loss=0.0997 | val_exact=0.9750
  Epoch 018 | loss=0.0491 | val_exact=0.9930
  Epoch 019 | loss=0.0212 | val_exact=0.9970
  Epoch 020 | loss=0.0154 | val_exact=0.9990
  Epoch 021 | loss=0.0128 | val_exact=0.9980
  Epoch 022 | loss=0.0108 | val_exact=0.9990
  Epoch 023 | loss=0.0095 | val_exact=0.9960
  Epoch 024 | loss=0.0087 | val_exact=0.9980
  Epoch 025 | loss=0.0078 | val_exact=0.9970
  Epoch 026 | loss=0.0072 | val_exact=0.9970
  Epoch 027 | loss=0.0063 | val_exact=0.9980
  Epoch 028 | loss=0.0060 | val_exact=0.9970
  Epoch 029 | loss=0.0058 | val_exact=0.9980
  Epoch 030 | loss=0.0051 | val_exact=0.9990
  Epoch 031 | loss=0.0050 | val_exact=0.9980
  Epoch 032 | loss=0.0045 | val_exact=0.9990
  Early stopping triggered.
  
  Best validation accuracy: 0.9990 at epoch 20
  Test accuracy: 0.9940
  Model saved to norm-padding-outputs/secret_accountant_model.npz
  History saved to norm-padding-outputs/training_history.npz
  
  Saved model to: norm-padding-outputs/secret_accountant_model.npz
  Saved history to: norm-padding-outputs/training_history.npz
  
  Validation examples:
  
  Examples:
  IN: 93+06   TARGET: 99   PRED: 99
  IN: 85+04   TARGET: 89   PRED: 89
  IN: 89+68   TARGET: 157   PRED: 157
  IN: 95+59   TARGET: 154   PRED: 154
  IN: 68+90   TARGET: 158   PRED: 158
  IN: 63+45   TARGET: 108   PRED: 108
  IN: 50+76   TARGET: 126   PRED: 126
  IN: 95+38   TARGET: 133   PRED: 133
  IN: 51+52   TARGET: 103   PRED: 103
  IN: 42+94   TARGET: 136   PRED: 136
  IN: 13+96   TARGET: 109   PRED: 109
  IN: 39+35   TARGET: 74   PRED: 74
  
  Test examples:
  
  Examples:
  IN: 27+25   TARGET: 52   PRED: 52
  IN: 72+69   TARGET: 141   PRED: 141
  IN: 16+46   TARGET: 62   PRED: 62
  IN: 97+74   TARGET: 171   PRED: 171
  IN: 71+10   TARGET: 81   PRED: 81
  IN: 14+80   TARGET: 94   PRED: 94
  IN: 92+54   TARGET: 146   PRED: 146
  IN: 47+08   TARGET: 55   PRED: 55
  IN: 38+46   TARGET: 84   PRED: 84
  IN: 11+21   TARGET: 32   PRED: 32
  IN: 05+88   TARGET: 93   PRED: 93
  IN: 88+64   TARGET: 152   PRED: 152
  Saved loss plot to norm-padding-outputs/loss_vs_iteration.png
  Saved hidden state heatmap to norm-padding-outputs/hidden_state_heatmap.png
  ```

## Evaluation 
- Below are `--test_all` results

- 1+1 Mode
  ```
  Model loaded from outputs/secret_accountant_model.npz

  Full 0+0 to 99+99 evaluation
  Correct: 9942/10000
  Accuracy: 0.9942
  
  Sample mistakes:
    IN: 0+6   TARGET: 6   PRED: 5
    IN: 0+9   TARGET: 9   PRED: 0
    IN: 1+0   TARGET: 1   PRED: 3
    IN: 1+6   TARGET: 7   PRED: 6
    IN: 2+0   TARGET: 2   PRED: 3
    IN: 2+7   TARGET: 9   PRED: 8
    IN: 3+17   TARGET: 20   PRED: 80
    IN: 3+18   TARGET: 21   PRED: 81
    IN: 3+19   TARGET: 22   PRED: 12
    IN: 4+0   TARGET: 4   PRED: 5
    IN: 4+6   TARGET: 10   PRED: 9
    IN: 5+2   TARGET: 7   PRED: 6
    IN: 6+1   TARGET: 7   PRED: 6
    IN: 6+2   TARGET: 8   PRED: 9
    IN: 6+5   TARGET: 11   PRED: 10
    IN: 6+6   TARGET: 12   PRED: 13
    IN: 6+9   TARGET: 15   PRED: 14
    IN: 7+13   TARGET: 20   PRED: 29
    IN: 8+3   TARGET: 11   PRED: 12
    IN: 9+7   TARGET: 16   PRED: 17
    IN: 9+20   TARGET: 29   PRED: 39
    IN: 11+8   TARGET: 19   PRED: 29
    IN: 12+8   TARGET: 20   PRED: 10
    IN: 13+7   TARGET: 20   PRED: 10
    IN: 16+4   TARGET: 20   PRED: 10
  
  Accuracy by target length:
    Length 1: 45/55 = 0.8182
    Length 2: 4954/4995 = 0.9918
    Length 3: 4943/4950 = 0.9986
  
  Stopping condition: 10000/10000 predictions emitted EOS naturally
  Hit max_len (no EOS): 0/10000
  
  Stopping condition: 10000/10000 emitted EOS
  Hit max_len (no EOS): 0/10000
  
  === Examples: NO EOS (hit max_len) ===
  
  === Examples: EARLY EOS (too short) ===
    IN: 4+6 | TARGET: 10 | PRED: 9
    IN: 70+30 | TARGET: 100 | PRED: 90
    IN: 72+28 | TARGET: 100 | PRED: 90
    IN: 95+5 | TARGET: 100 | PRED: 91

  ```

- 01+01 Mode
  ```
  Model loaded from norm-padding-outputs/secret_accountant_model.npz

  Full 00+00 to 99+99 evaluation
  Correct: 9988/10000
  Accuracy: 0.9988
  
  Sample mistakes:
    IN: 00+00   TARGET: 0   PRED: 1
    IN: 00+01   TARGET: 1   PRED: 2
    IN: 00+02   TARGET: 2   PRED: 3
    IN: 00+04   TARGET: 4   PRED: 5
    IN: 00+06   TARGET: 6   PRED: 7
    IN: 00+09   TARGET: 9   PRED: 8
    IN: 01+01   TARGET: 2   PRED: 3
    IN: 01+06   TARGET: 7   PRED: 6
    IN: 02+00   TARGET: 2   PRED: 3
    IN: 04+00   TARGET: 4   PRED: 5
    IN: 06+01   TARGET: 7   PRED: 6
    IN: 25+35   TARGET: 60   PRED: 50
  
  Accuracy by target length:
    Length 1: 44/55 = 0.8000
    Length 2: 4994/4995 = 0.9998
    Length 3: 4950/4950 = 1.0000
  
  Stopping condition: 10000/10000 predictions emitted EOS naturally
  Hit max_len (no EOS): 0/10000
  
  Stopping condition: 10000/10000 emitted EOS
  Hit max_len (no EOS): 0/10000
  
  === Examples: NO EOS (hit max_len) ===
  
  === Examples: EARLY EOS (too short) ===

  ```

## Analysis Questions

**1. Did the model produce the correct output on test examples?**

Yes, for the majority of cases.
- 1+1 Mode: Achieved 97.60% accuracy on the initial test set and 99.42% on the full evaluation 
- 01+01 Mode: Achieved 99.40% accuracy on the initial test set and 99.88% on the full evaluation
- Higher accuracy for the 01+01 mode may be attributed to spatial alignment. In 01+01 mode, the `+` sign is always reliably at index 2. Still, it is significant the difference between them is not very large

However, it still struggled with certain sequences
- 1+1 Mode: Accuracy dropped to 81.82% for single-digit targets. It also triggered EoS errors. For example, it predicted 9 instead of 10 for 4+6
- 01+01 Mode: Accuracy dropped to 80% for single-digit targets as well
- This may be because single-digit targets are generally rarer. It suggests the model relies on pattern matching instead of reasoning
- It is of note many of these errors seem to be off by one 

**2. How fast did the loss decrease?**

Most significant learning seemed to happen within the first 10 epochs.
- 1+1 Mode: Loss started at 4.5 then dropped to 1.4 by Epoch 10
- 01+01 Mode: Loss started at 4.5 then dropped to 0.9 by Epoch 10
- After this drop, loss reduction slowed down
  
In terms of convergence,
- 1+1 was slower, around Epoch 35
- 01+01 was faster, started around Epoch 18
- This may be because 01+01 data was already normalized and had less spatial differences
  
**3. What patterns appeared in the hidden-state heatmap?**
- The heatmaps reveal the model uses the `+` token as a sort of transition marker, suggested by the color shift
- The further right we go, the more high-contrast the patterns become

**4. What does the result suggest about the strengths and limits of Encoder-Decoder RNNs?**
- Results show they are effective even in mapping variable-length sequences, allowing them to learn patterns and stopping conditions. Their limitation is how they rely on statistical frequency instead of actual logical reasoning hence the one-off mistakes we encountered

