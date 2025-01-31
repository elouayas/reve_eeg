---
library_name: transformers
tags: []
---

# Model Card for REVE Position Bank

Wrapper to provide electrode positions to use the [REVE EEG Foundation Model](https://brain-bzh.github.io/reve/).


## Model Details

### Model Description

<!-- Provide a longer summary of what this model is. -->

- **Developed by:** the [BRAIN team](https://www.imt-atlantique.fr/en/research-innovation/teams/brain) and [UdeM](https://www.umontreal.ca/en/)
- **Funded by :** AI@IMT, ANR JCJC ENDIVE, Jean Zay (with project numbers), Alliance Canada and Region Bretagne.


REVE (Representation for EEG with Versatile Embeddings) is a pretrained model explicitly designed to generalize across diverse EEG signals. 
REVE introduces a novel 4D positional encoding scheme that enables it to process signals of arbitrary length and electrode arrangement. 

This [position bank repository](https://huggingface.co/brain-bzh/position_bank) can be used to fetch electrode positions by name, in order to perform inference with the REVE modeL. 



### Model Sources

<!-- Provide the basic links for the model. -->

- **Repository:** [github](https://brain-bzh.github.io/reve/)
- **Paper :** [arxiv](https://arxiv.org/abs/2510.21585)

## Uses

Example script to fetch electrode positions and extract embeddings with REVE.

```python
from transformers import AutoModel

pos_bank = AutoModel.from_pretrained("brain-bzh/reve-positions", trust_remote_code=True)


eeg_data = ...  # EEG data (batch_size, channels, time_points), must be sampled at 200 Hz
electrode_names = [...]  # List of electrode names corresponding to the channels in eeg_data

positions = pos_bank(electrode_names) # Get positions (channels, 3)

model = AutoModel.from_pretrained("brain-bzh/reve-base", trust_remote_code=True)

## Expand the positions vector to match the batch size 
positions = positions.expand(eeg_data.size(0), -1, -1)  # (batch_size, channels, 3)

output = model(eeg_data, positions)
```

Available electrodes names can be printed using the method
`pos_bank.get_all_positions()`, and can be visualized [here](https://brain-bzh.github.io/reve/#Electrode%20positions).

Most common electrode setups are available (10-20, 10-10, 10-05, EGI 256). For Biosemi-128, use the prefix `biosemi128_` before the electrode names (e.g., `biosemi128_C13`).