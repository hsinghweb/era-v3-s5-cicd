# MNIST Classification

![Build Status](https://github.com/hsinghweb/era-v3-s5-cicd/actions/workflows/python-app.yml/badge.svg)

# PyTorch Model Project

A PyTorch-based machine learning project implementing.

## Prerequisites

- Python 3.8+
- PyTorch
- CUDA (optional, for GPU acceleration)

## Installation

1. Clone the repository:
bash
git clone https://github.com/hsinghweb/era-v3-s5-cicd.git
cd era-v3-s5-cicd

2. Create a virtual environment:
bash
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate

3. Install dependencies:
bash
pip install -r requirements.txt

## Project Structure

.
├── src/
│   ├── model.py          # Model architecture definition
│   ├── train.py          # Training script
│   ├── test_model.py     # Testing/inference script
├── data/                 # Dataset directory
├── requirements.txt      # Project dependencies
└── README.md             # This file

## Development

To contribute to this project:

1. Fork the repository
2. Create a new branch (`git checkout -b feature/improvement`)
3. Make your changes
4. Commit your changes (`git commit -am 'Add new feature'`)
5. Push to the branch (`git push origin feature/improvement`)
6. Create a Pull Request

## Training

1. Prepare your data:
   - Place your dataset in the `data/` directory
   - Or modify the data loader to use your custom dataset location

2. Start training:
```bash
python src/train.py
```

## Testing/Inference

1. To evaluate a trained model:
```bash
python src/test_model.py
```

## Contact

Himanshu Singh - himanshu.kumar.singh@gmail.com

Project Link: https://github.com/hsinghweb/era-v3-s5-cicd

