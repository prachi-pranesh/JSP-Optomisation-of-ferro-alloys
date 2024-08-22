# Alloy Quantity Estimation Model

## Overview

This project aims to develop a machine learning model to estimate the optimal quantity of alloys required to achieve a desired yield in metallurgical processes. The model consists of two submodels:

1. **Chemistry Estimation Submodel**: Predicts the resulting chemical composition when a specific combination of alloys is added to the ladle.
2. **Alloy Quantity Optimization Submodel**: Utilizes the first submodel to determine the optimal amount of each alloy required to achieve the desired yield for a given grade while considering cost-effectiveness.

## Features

- **Predictive Modeling**: Estimate the chemical composition based on alloy inputs.
- **Optimization**: Calculate the most cost-effective alloy quantities to achieve the desired yield.
- **Customizable**: Adjust the model parameters and training data to suit specific metallurgical processes.

## Requirements

To run this project, you need to have the following dependencies installed:

- `tensorflow==2.11.0`
- `numpy==1.23.5`
- `pandas==1.4.4`

## Instructions:
1. Clone the repository using:
```
git clone https://github.com/yourusername/alloy-quantity-estimation.git
cd alloy-quantity-estimation
```

2. Train the model using:
```
python main.py
```
