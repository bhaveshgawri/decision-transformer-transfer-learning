> some experiments with training and fine-tuning decision transformer

#### Installation

```
conda create -n my_env python=3.10
conda activate my_env
pip install -r requirements.txt
python -m ipykernel install --user --name my_env --display-name "My Env"
```

#### Report
A pdf copy of the project report is [available in this repo here](https://github.com/bhaveshgawri/decision-transformer-transfer-learning/blob/main/report/report.pdf). \
Decision Transformer: https://arxiv.org/pdf/2106.01345.pdf

#### Sidenote
Higher versions (2.3.4 or above) of Mujoco ~are currently~ were not working for Hopper/Walker2d during the timeframe of this project:
https://github.com/deepmind/mujoco/issues/833
