# Simplified Self-Supervised Fault-Pattern Discovery Report

## 1. Purpose

This report describes the architecture currently implemented in `anomaly_detection_custom_data.ipynb` and the additional components required to discover and group previously unseen electrical fault patterns.

The goal is not to immediately assign a physical fault name. The first goal is to identify repeatable groups of abnormal electrical behavior that can later be investigated and confirmed.

## 2. Original Architecture

### 2.1 Data input

The notebook reads two columns from each CSV file:

- `V_v`: voltage
- `I_a`: current

Each file is divided into non-overlapping sequences of 40 samples. The folders `with 3 lamps` and `without 3 lamps` are treated as normal data. The `test` folder is treated as anomalous data.

### 2.2 Preprocessing

1. Invalid numeric values are replaced with zero.
2. The normal sequences are split randomly into 80% training and 20% validation data.
3. A `StandardScaler` is fitted on the training data.
4. The training, validation, and test sequences are standardized using that scaler.

### 2.3 LSTM autoencoder

The model contains:

```text
Input: 40 time steps x 2 channels
        ↓
Two-layer LSTM encoder
Hidden size: 64
        ↓
Final hidden state: 64-dimensional representation
        ↓
Repeat representation across 40 time steps
        ↓
Two-layer LSTM decoder
        ↓
Reconstructed voltage/current sequence
```

The model is trained with mean squared error:

```text
Loss = mean((reconstructed_sequence - original_sequence)^2)
```

The original input is also the training target. Therefore, the training objective is already **self-supervised reconstruction**, although the notebook does not yet use self-supervised representation learning specifically for fault-pattern clustering.

### 2.4 Original output

For every sequence, the notebook calculates one reconstruction error:

```text
MSE = mean error across all 40 time steps and both channels
```

Interpretation:

- Low MSE: the sequence resembles the normal training data.
- High MSE: the sequence differs from the normal training data.

The notebook then compares validation errors with test-folder errors and calculates ROC-AUC using the assumption that validation data are normal and test data are anomalous.

### 2.5 What the original architecture can and cannot do

It can:

- Learn common voltage/current behavior from presumed-normal data.
- Detect sequences that reconstruct poorly.
- Produce a useful 64-dimensional encoder representation.

It cannot reliably:

- Identify which type of fault occurred.
- Discover multiple fault groups from unlabeled abnormal data.
- Distinguish a genuine fault from a new but healthy operating mode.
- Explain whether an anomaly comes from voltage, current, phase, harmonics, or a sensor problem because all information is reduced to one MSE value.

## 3. Required New Architecture

The new system should keep the original autoencoder as the baseline detector, but add a separate representation and clustering pipeline.

```text
Voltage/current sequence
        ↓
Quality checks and physical windowing
        ↓
Self-supervised representation learning
        ↓
Normal-reference anomaly scoring
        ↓
Merge nearby anomalous windows into events
        ↓
Extract event embeddings and electrical features
        ↓
HDBSCAN clustering
        ↓
Candidate abnormal-signature groups
```

### 3.1 Improve the data split and preprocessing

Before changing the model:

- Split by recording, session, or device before creating windows.
- Confirm whether `V_v` and `I_a` are instantaneous waveforms or RMS/slow measurements.
- Confirm the sampling rate so that 40 samples have a meaningful physical duration.
- Preserve timestamps, missing-value information, and sensor-quality information.
- Do not automatically replace missing samples with zero without recording that this occurred.

This prevents the model from learning file, sensor, or operating-mode differences as if they were faults.

### 3.2 Preserve richer anomaly information

The current scalar MSE should be supplemented with:

- Voltage reconstruction error.
- Current reconstruction error.
- Error at each time step.
- Peak error and error duration.
- The 64-dimensional encoder embedding.
- Electrical features appropriate to the measurement type.

If synchronized instantaneous waveforms are available, possible features include RMS voltage/current, active power, apparent power, power factor, crest factor, harmonic content, and transient duration. If the data are slow telemetry, use level, trend, duty cycle, transition size, and operating-state duration instead.

### 3.3 Add a masked-reconstruction objective

The first recommended self-supervised extension is masked reconstruction:

1. Hide short contiguous portions of the voltage/current sequence.
2. Provide a mask indicator to the model.
3. Ask the model to reconstruct the original hidden values.
4. Use the resulting encoder representation for anomaly-event clustering.

```text
Original V/I sequence
        ↓
Randomly mask short sections
        ↓
Masked LSTM encoder
        ↓
Decoder predicts the missing values
        ↓
Learned representation of temporal and electrical context
```

This encourages the model to learn temporal structure rather than simply reproduce every input value. The original reconstruction loss can be retained as a comparison baseline.

The masking must preserve the physical relationship between voltage and current. Independent transformations of the two channels can create unrealistic electrical signals.

### 3.4 Optional contrastive representation learning

As a second representation-learning comparison, use TS2Vec or a similar temporal contrastive method. The model receives two physically valid views of the same sequence and learns to keep their representations close while separating unrelated sequences.

Suitable transformations must be tested carefully. Large amplitude changes, arbitrary time permutations, or independent voltage/current shifts may remove or corrupt fault information.

The recommended comparison is:

```text
Existing AE embedding
vs.
Masked-LSTM embedding
vs.
TS2Vec embedding
```

All three should be evaluated on the same recordings, event definitions, and clustering procedure.

### 3.5 Convert anomalous windows into events

Adjacent anomalous windows should be merged into one event. Otherwise, one long incident can be counted as hundreds of separate samples and create misleadingly strong clusters.

For each event, retain:

- Start and end time.
- Maximum and average anomaly score.
- Voltage and current residual summaries.
- Pooled encoder embedding.
- Relevant electrical features.
- Source device and recording metadata.

### 3.6 Cluster abnormal events

Use the event-level features as input to HDBSCAN:

```text
Event embeddings + residual features + electrical features
        ↓
Standardization/PCA if needed
        ↓
HDBSCAN
        ↓
Group A, Group B, Group C, and unassigned/noise events
```

HDBSCAN is a suitable first method because:

- The number of abnormal groups is not known in advance.
- It can identify groups with different densities.
- It can leave unusual events unassigned.

Clusters should initially be named `abnormal_group_01`, `abnormal_group_02`, and so on. A cluster is a group of similar representations, not proof of a specific internal component failure.

### 3.7 Validate the discovered groups

Check whether clusters are stable across:

- Random seeds.
- Recording sessions.
- Devices and appliance models.
- Reasonable HDBSCAN parameters.
- Different representation types.

Also check whether a cluster is actually explained by:

- Lamp count.
- Appliance identity.
- Normal operating mode.
- Sensor or recording setup.
- Supply-voltage differences.

Physical fault labels should only be assigned after maintenance records, controlled experiments, or qualified engineering review confirm them.

## 4. Simplified Comparison

| Aspect | Original notebook | Required new system |
|---|---|---|
| Main objective | Reconstruct normal V/I sequences | Learn useful representations and group abnormal events |
| Representation | Final LSTM hidden state exists but is unused for clustering | Encoder embeddings are explicitly extracted and evaluated |
| Anomaly score | One combined MSE | Channel-specific, temporal, and combined scores |
| Learning objective | Full-sequence reconstruction | Masked reconstruction and optionally contrastive learning |
| Unit of analysis | Independent 40-sample windows | Merged temporal events |
| Fault discovery | None | HDBSCAN or another clustering method |
| Number of fault groups | Not modeled | Discovered from the data, with an unassigned category |
| Interpretation | “This looks unusual” | “These unusual events resemble one another” |
| Fault diagnosis | Not supported | Possible only after independent physical validation |

## 5. Research Papers Inspiring the Approach

### 5.1 LSTM reconstruction anomaly detection

**Malhotra et al. (2016), “LSTM-based Encoder-Decoder for Multi-sensor Anomaly Detection.”**

Source: https://arxiv.org/abs/1607.00148

This paper inspired the use of an LSTM encoder-decoder trained to reconstruct normal multivariate time series. Reconstruction errors are used as anomaly scores. It supports the original notebook architecture, but does not provide a method for naming unknown fault mechanisms.

### 5.2 Self-supervised time-series representations

**Yue et al. (2022), “TS2Vec: Towards Universal Representation of Time Series.”**

Source: https://arxiv.org/abs/2106.10466

TS2Vec uses hierarchical contrastive learning to produce timestamp-level and subsequence-level representations. It motivates comparing the existing autoencoder embedding with a representation trained to capture temporal context at multiple scales.

**Eldele et al. (2021), “Time-Series Representation Learning via Temporal and Contextual Contrasting.”**

Source: https://arxiv.org/abs/2106.14112

TS-TCC combines temporal prediction and contextual contrastive learning. It motivates learning representations that remain useful when only a small number of examples can later be reviewed or labeled. Electrical validity must guide the augmentations.

**Tonekaboni, Eytan, and Goldenberg (2021), “Unsupervised Representation Learning for Time Series with Temporal Neighborhood Coding.”**

Source: https://arxiv.org/abs/2106.00750

Temporal Neighborhood Coding learns from nearby temporal contexts. It is relevant when appliances have persistent operating states, but temporal proximity must not be assumed to mean the same state around a fault onset.

**Nie et al. (2023), “A Time Series is Worth 64 Words: Long-term Forecasting with Transformers.”**

Source: https://arxiv.org/abs/2211.14730

This paper motivates masked temporal-segment reconstruction through PatchTST. The proposed project should first borrow the masked-learning idea in the existing LSTM rather than immediately replacing the model with a Transformer.

### 5.3 Electrical load and V-I representations

**Hart (1992), “Nonintrusive Appliance Load Monitoring.”**

Source: https://www.georgehart.com/research/nalm.html

Hart's work motivates modeling appliance operating states and transitions. A new operating mode should not automatically be interpreted as a fault.

**De Baets et al. (2018), “Appliance classification using VI trajectories and convolutional neural networks.”**

Source: https://doi.org/10.1016/j.enbuild.2017.09.087

This paper motivates V-I trajectory representations for appliance signatures. It is an appliance-classification paper, not an internal-fault diagnosis paper. It is relevant only when synchronized instantaneous voltage and current are available.

**Medico et al. (2020), “A voltage and current measurement dataset for plug load appliance identification in households.”**

Source: https://doi.org/10.1038/s41597-020-0389-7

The PLAID dataset paper motivates testing V-I representations across appliances, models, locations, and acquisition conditions. It is useful for representation development, but it is not a controlled internal-fault benchmark.

### 5.4 Clustering and one-class detection

**Xie, Girshick, and Farhadi (2016), “Unsupervised Deep Embedding for Clustering Analysis.”**

Source: https://proceedings.mlr.press/v48/xieb16.html

DEC motivates jointly refining an embedding and cluster assignments. It should be considered only after a stable frozen-embedding clustering baseline because DEC can reinforce incorrect initial clusters and requires a cluster-count assumption.

**McInnes and Healy (2017), “Accelerated Hierarchical Density Clustering.”**

Source: https://arxiv.org/abs/1705.07321

This paper motivates HDBSCAN-style density clustering for cases where the number of groups is unknown and some events should remain unassigned. Cluster stability and parameter sensitivity still need to be evaluated.

**Ruff et al. (2018), “Deep One-Class Classification.”**

Source: https://proceedings.mlr.press/v80/ruff18a.html

Deep SVDD motivates an alternative normality objective that maps healthy data into a compact representation region. It improves anomaly scoring rather than discovering fault categories, so it should be evaluated as a separate detector.

## 6. Recommended Minimum Implementation

The smallest useful change to the existing notebook is:

1. Keep the current LSTM autoencoder and reconstruction score.
2. Extract the final encoder hidden state for every window.
3. Calculate separate voltage and current residual features.
4. Merge adjacent anomalous windows into events.
5. Combine event embeddings with residual and electrical features.
6. Standardize the combined event representation.
7. Run HDBSCAN and retain noise/unassigned events.
8. Inspect cluster stability and operating-mode confounding.
9. Add masked-LSTM reconstruction as the first new self-supervised training experiment.
10. Compare with TS2Vec only after the baseline clustering pipeline is working.

This sequence preserves the existing detector, limits architectural risk, and directly tests whether better representations improve discovery of abnormal electrical behavior.

## 7. Final Interpretation

The original architecture answers:

> Does this voltage/current sequence resemble the presumed-normal training data?

The proposed architecture should answer:

> Which abnormal events resemble one another, and are those groups stable enough to investigate?

Only independent electrical or maintenance evidence can answer the final question:

> Which physical fault caused this abnormal signature?
