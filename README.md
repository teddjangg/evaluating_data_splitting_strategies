## Overview
This project investigates how different data-splitting strategies affect reported model performance in a regression task.

Rather than optimizing model accuracy, the focus is on evaluating how common assumptions—particularly the IID (independent and identically distributed) assumption—influence performance estimates under different validation settings.

---

## Note
This project serves as an evaluation-focused study on model validation rather than model development.

While random train–test splits and K-fold cross-validation assume strong IID conditions, further investigation shows that relaxing this assumption through time-based and group-based splits can substantially degrade performance and increase variance.  
This highlights the risk of overly optimistic evaluation when deployment conditions differ from training data.

---

## Methods
- Fixed regression model with controlled hyperparameters
- Strong IID evaluation:
  - Random train–test split
  - K-fold cross-validation
- Weak IID evaluation:
  - Time-based split (past → future)
  - Group-based split (leave-one-group-out)
- Performance comparison under progressively relaxed IID assumptions

---

## Evaluation
- Metric: Mean Absolute Error (MAE)
- Rationale: MAE is used to reduce sensitivity to extreme values in the heavy-tailed target distribution

---

## Key Findings
- Strong IID splits produce optimistic and stable performance estimates
- Weak IID splits result in higher error and substantially increased variance
- Group-based splits reveal strong sensitivity to group-specific distributions, indicating limited generalization to unseen groups

---

## Data
The dataset is provided by Kaggle.  
Due to licensing restrictions, raw data is not included in this repository.

You can download the dataset from:  
https://www.kaggle.com/datasets/srikaranelakurthy/online-news-popularity

---

## Tech Stack
- Language: Python
- Libraries: pandas, scikit-learn, matplotlib
