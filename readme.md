
#### Installation

```
conda create -n my_env python=3.10
conda activate my_env
pip install -r requirements.txt
python -m ipykernel install --user --name my_env --display-name "My Env"
```

Higher versions (2.3.4 or above) of Mujoco are currently not working for Hopper/Walker2d:
https://github.com/deepmind/mujoco/issues/833

