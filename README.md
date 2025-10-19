# AccuACL
*Official implementation of the ICLR 2025 paper “Active Learning for Continual Learning: Keeping the Past Alive in the Present”*

AccuACL introduces a **Fisher information–based perspective** to Active Continual Learning (ACL), proposing a unified query strategy that **balances the prevention of catastrophic forgetting** and **rapid adaptation to new tasks**.  
This repository provides reproducible implementations of the experiments presented in the ICLR 2025 paper and extends the Avalanche continual-learning framework with active-learning integration.

---

## Overview

Traditional Active Learning (AL) algorithms focus on selecting informative examples to improve new-task performance but overlook **stability across tasks**, which is a critical issue in Continual Learning (CL).  
Conversely, most CL methods focus on preventing forgetting but do not actively query new data.

**AccuACL bridges this gap** by introducing *Accumulated Informativeness*, a measure that evaluates how candidate samples simultaneously contribute to **retaining prior knowledge** and **learning new information**.  
It leverages the **Fisher Information Matrix (FIM)** to quantify parameter importance across tasks, efficiently guiding active query selection under limited labeling budgets.

### Core Ideas from the Paper
- **Fisher-based Query Strategy:** AccuACL estimates Fisher information over both labeled memory and unlabeled pools, approximating parameter importance across tasks.  
- **Accumulated Informativeness:** A unified measure of informativeness that accounts for both *plasticity* (learning new tasks) and *stability* (retaining old ones).  
- **Efficient Approximation:** A scalable FIM-based selection process with significantly reduced time and space complexity compared to prior methods such as BAIT.  
- **Empirical Validation:** AccuACL achieves **state-of-the-art ACL performance** across SplitCIFAR10, SplitCIFAR100, and SplitTinyImageNet—improving average accuracy by up to 23.8% and reducing forgetting by 17.0% on average.
---

## Repository Structure

- `acl/` – Main experiment codes.
- `avalanche/` – Embedded Avalanche fork with extensions for AccuACL.
- `run_experiment.sh` – Example script

## Dataset Configuration

Several benchmark paths in `acl/data.py` reference absolute locations 
You can either:
- Update these to match your local dataset paths, or  
- Modify the data helper to download them dynamically.

Checkpoints are saved under by default; adjust this in `acl/multi_round_baseline.py`.

## Citation

If you use this repository, please cite:

```
@inproceedings{
park2025active,
title={Active Learning for Continual Learning: Keeping the Past Alive in the Present},
author={Jaehyun Park and Dongmin Park and Jae-Gil Lee},
booktitle={The Thirteenth International Conference on Learning Representations},
year={2025},
url={https://openreview.net/forum?id=mnLmmtW7HO}
}
```
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
