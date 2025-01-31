---
library_name: transformers
tags:
  - eeg
  - neuroscience
  - foundation-model
  - pytorch
license: other
extra_gated_prompt: >-
  ## EEG FOUNDATION MODEL RESPONSIBLE USE AGREEMENT

  This model is available for general use (research, commercial, and personal), provided strictly that you adhere to the following privacy and safety standards. By requesting access, you agree to be bound by the following ethical principles and the regulatory guidance outlined in **EDPB Opinion 28/2024**.
  
  1. **No Privacy Intrusion or Reconstruction**
  You acknowledge that AI models trained on personal data may not be fully anonymous and can be vulnerable to attacks. You expressly agree **NOT** to:
    - Attempt to extract, infer, or reconstruct subject-level EEG data or personal information from the model weights or outputs.
    - Perform "Model Inversion" or "Membership Inference" attacks to extract statistical data related to specific individuals.
    - Attempt to re-identify individuals from the model's embeddings.
  
  2. **No Harm, Surveillance, or Discrimination**
  In line with protecting fundamental rights, you will not use this model for:
    - **Biometric Identification:** Continuous monitoring, behavioral profiling, or identification of natural persons.
    - **Discrimination:** Any purpose that leads to unfair treatment of individuals or groups, or exploits vulnerabilities (e.g., age, disability).
    - **Manipulation:** Coercing or exploiting users, particularly vulnerable populations, or infringing on human autonomy.
    
  3. **Fair Use, Security, and Data Minimisation**
  If you deploy this model, you accept accountability for the processing. You must:
    - **Minimize Data:** Ensure any additional data used with the model is limited, pseudonymised where possible, and securely handled.
    - **Be Transparent:** Any research or deployment must clearly state the purpose, limitations, and safeguards implemented to protect rights.
    - **Secure the Deployment:** Implement measures to prevent unauthorized access or adversarial attacks on the model.
  
  4. **Redistribution and Access Revocation**
    - **No Redistribution:** You will not share, host, or distribute the model weights or derivatives to users without permission; they must access the model via this repository to agree to these terms.
    - **Dataset Withdrawal:** If any underlying dataset becomes closed or restricted, access to this model may be revoked or replaced by a retrained version.

extra_gated_fields:
  Full Name: text
  Organization / Entity: text
  I want to use this model for:
    type: select
    options: 
      - Research
      - Industry
      - Education
      - label: Other
        value: other
  geo: ip_location
  I agree to the non-identification, no-harm, and privacy terms above: checkbox
  I acknowledge that access may be revoked if underlying datasets are restricted: checkbox
extra_gated_description: >-
  Access is open to verified users who agree to strict privacy, no-harm, and non-identification policies compliant with EDPB guidelines.
extra_gated_button_content: Accept Terms & Request Access

model-index:
- name: Reve-base
  results:
  - task:
      type: feature-extraction
    dataset:
      name: TUAB
      type: TUAB
    metrics:
    - type: Accuracy
      value: 0.8315
      name: Accuracy
  - task:
      type: feature-extraction
    dataset:
      name: TUEV
      type: TUEV
    metrics:
    - type: Accuracy
      value: 0.6759
      name: Accuracy
  - task:
      type: feature-extraction
    dataset:
      name: PhysionetMI
      type: PhysionetMI
    metrics:
    - type: Accuracy
      value: 0.648
      name: Accuracy
  - task:
      type: feature-extraction
    dataset:
      name: BCICIV2a
      type: BCICIV2a
    metrics:
    - type: Accuracy
      value: 0.6396
      name: Accuracy
  - task:
      type: feature-extraction
    dataset:
      name: FACED
      type: FACED
    metrics:
    - type: Accuracy
      value: 0.5646
      name: Accuracy
  - task:
      type: feature-extraction
    dataset:
      name: ISRUC
      type: ISRUC
    metrics:
    - type: Accuracy
      value: 0.7819
      name: Accuracy
  - task:
      type: feature-extraction
    dataset:
      name: Mumtaz
      type: Mumtaz
    metrics:
    - type: Accuracy
      value: 0.9644
      name: Accuracy
  - task:
      type: feature-extraction
    dataset:
      name: MentalArithmetic
      type: MentalArithmetic
    metrics:
    - type: Accuracy
      value: 0.766
      name: Accuracy
  - task:
      type: feature-extraction
    dataset:
      name: BCI2020-3
      type: BCI2020-3
    metrics:
    - type: Accuracy
      value: 0.5635
      name: Accuracy
  - task:
      type: feature-extraction
    dataset:
      name: TUAB-LP
      type: TUAB-LP
    metrics:
    - type: Accuracy
      value: 0.81
      name: Accuracy
  - task:
      type: feature-extraction
    dataset:
      name: TUEV-LP
      type: TUEV-LP
    metrics:
    - type: Accuracy
      value: 0.592
      name: Accuracy
  - task:
      type: feature-extraction
    dataset:
      name: PhysionetMI-LP
      type: PhysionetMI-LP
    metrics:
    - type: Accuracy
      value: 0.537
      name: Accuracy
  - task:
      type: feature-extraction
    dataset:
      name: BCICIV2a-LP
      type: BCICIV2a-LP
    metrics:
    - type: Accuracy
      value: 0.517
      name: Accuracy
  - task:
      type: feature-extraction
    dataset:
      name: ISRUC-LP
      type: ISRUC-LP
    metrics:
    - type: Accuracy
      value: 0.697
      name: Accuracy
  - task:
      type: feature-extraction
    dataset:
      name: Mumtaz-LP
      type: Mumtaz-LP
    metrics:
    - type: Accuracy
      value: 0.962
      name: Accuracy
  - task:
      type: feature-extraction
    dataset:
      name: MentalArithmetic-LP
      type: MentalArithmetic-LP
    metrics:
    - type: Accuracy
      value: 0.74
      name: Accuracy
  - task:
      type: feature-extraction
    dataset:
      name: BCII2020-3-LP
      type: BCII2020-3-LP
    metrics:
    - type: Accuracy
      value: 0.39
      name: Accuracy
---
# Model Card for REVE-base

<!-- Provide a quick summary of what the model is/does. -->

REVE ([project page here](https://brain-bzh.github.io/reve/)) is a transformer-based foundation model for EEG signal processing. It was trained on 60k hours of EEG data from various sources and is designed to be adaptable to any electrode configuration and a wide range of EEG-based tasks.




## Model Details

### Architecture

<!-- Provide a longer summary of what this model is. -->

REVE (Representation for EEG with Versatile Embeddings), a pretrained encoder explicitly designed to generalize across diverse EEG signals. 
REVE introduces a novel 4D positional encoding scheme that enables it to process signals of arbitrary length and electrode arrangement.
Using a masked autoencoding objective, we pretrain REVE on over 60,000 hours of EEG data from 92 datasets spanning 25,000 subjects.

**Developed by** the [BRAIN team](https://www.imt-atlantique.fr/en/research-innovation/teams/brain) and [UdeM](https://www.umontreal.ca/en/)

**Funded by:** This research was supported by the French National Research Agency (ANR) through its AI@IMT program and grant ANR-24-CE23-7365, as well as by a grant from the Brittany region. Further support was provided by a Discovery Grant from the Natural Sciences and Engineering Research Council of Canada (NSERC), by funding from the Canada Research Chairs program and the Fonds de recherche du Québec – Nature et technologies (FRQ-NT). This work was granted access to the HPC resources of IDRIS under the allocation 2024-AD011015237R1 made by GENCI, as well as HPC provided by Digital Alliance Canada. 

### Model Sources

<!-- Provide the basic links for the model. -->

- **Repository:** [github](https://brain-bzh.github.io/reve/)
- **Paper :** [arxiv](https://arxiv.org/abs/2510.21585)

## Uses

Example script to extract embeddings with REVE, using our position bank:
```python
from transformers import AutoModel

pos_bank = AutoModel.from_pretrained("brain-bzh/reve-positions", trust_remote_code=True)
model = AutoModel.from_pretrained("brain-bzh/reve-base", trust_remote_code=True)

eeg_data = ... # EEG data as a torch Tensor (batch_size, channels, time_points), must be sampled at 200 Hz

electrode_names = [...] # List of electrode names corresponding to the channels in eeg_data
positions = pos_bank(electrode_names) # Get positions (channels, 3)
# Expand the positions vector to match the batch size 
positions = positions.expand(eeg_data.size(0), -1, -1)  # (batch_size, channels, 3)

output = model(eeg_data, positions)
```