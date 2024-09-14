# DRMOE: Towards Better Mixture of Experts via Dual Routing Strategy

This is  the implementation of the paper "DRMOE: Towards Better Mixture of Experts via Dual Routing Strategy".



## Running

You can implement out model according to the following steps:

1. Install the necessary packages. Run the command:

   ```shell
   pip install -r requirements.txt
   ```

2. To train the DRMOE and generate the answer to test, please run the command:

   ```bash
   bash ./experiments/train.sh
   ```

3. Finally, you can get the answer at results/eval_metric.csv.



## Requirements

To ease the configuration of the environment, I list versions of my hardware and software equipments:

- Hardware:
  - GPU: RTX 3090
  - Cuda: 11.3.1
  - Driver version: 535.146.02
- Software:
  - Python: 3.9.5
  - Pytorch: 1.12.0+cu113
  - transformers: 4.28.1
  - deepspeed: 0.9.4

You can also visit this [link](https://drive.google.com/file/d/1pqRZUa8HB02qxcwGU1TzZqUipAYzpFLm/view?usp=sharing) to get the tar.gz file containing the complete virtual environment.



## Ablation experiment supplement

| Method     | CMeIE  | CHIP-CDN | CHIP-CDEE | CHIP-MDCFNPC | CHIP-CTC | KUAKE-QIC | IMCS-V2-MRG | MedDG  | Average |
| ---------- | ------ | -------- | --------- | ------------ | -------- | --------- | ----------- | ------ | ------- |
| DRMOE      | 0.4675 | 0.8247   | 0.5622    | 0.7813       | 0.8927   | 0.8597    | 0.3771      | 0.1126 | 0.6097  |
| w/o TSL&DR | 0.4497 | 0.8229   | 0.5545    | 0.7810       | 0.8645   | 0.8563    | 0.3608      | 0.1093 | 0.5999  |
| w/o TSL    | 0.4784 | 0.8355   | 0.5668    | 0.7729       | 0.8836   | 0.8021    | 0.3630      | 0.1135 | 0.6020  |
| w/o DR     | 0.4655 | 0.8168   | 0.5685    | 0.7736       | 0.8864   | 0.8587    | 0.3805      | 0.1138 | 0.6080  |

These additional ablation experiments further demonstrate the effectiveness of the proposed module in our study.