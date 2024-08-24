# Ferroalloy Optimization Model

## Introduction

The ferroalloy optimization model is designed to enhance the efficiency of alloy additions in the metallurgical process. This model leverages data on alloy compositions, costs, and desired properties of the final product to determine the most cost-effective and efficient alloying strategy. It analyzes the current composition of the molten metal and predicts the optimal amounts and types of ferroalloys to be added to achieve the desired chemical and physical properties.

By incorporating real-time data and predictive algorithms, the model can adjust to variations in raw material quality and operational conditions, ensuring consistent product quality. It minimizes waste and reduces costs by accurately calculating the necessary additions, avoiding excess use of expensive alloys. Additionally, the model can simulate different scenarios, helping metallurgists make informed decisions and adapt quickly to changing requirements.

Overall, this ferroalloy optimization model enhances the control and precision of the alloying process, leading to improved product quality, reduced costs, and increased operational efficiency.The Ferroalloy Optimization Model is designed to enhance the efficiency of alloy additions in the metallurgical process. This model leverages data on alloy compositions, costs, and desired properties of the final product to determine the most cost-effective and efficient alloying strategy. It analyzes the current composition of the molten metal and predicts the optimal amounts and types of ferroalloys to be added to achieve the desired chemical and physical properties.

- **Cost and Efficiency Optimization**: Analyzes the current composition of the molten metal and predicts the optimal amounts and types of ferroalloys to be added to achieve the desired chemical and physical properties.
- **Real-Time Data Integration**: Incorporates real-time data and predictive algorithms to adjust for variations in raw material quality and operational conditions, ensuring consistent product quality.
- **Waste Reduction**: Minimizes waste and reduces costs by accurately calculating necessary additions, avoiding excess use of expensive alloys.
- **Scenario Simulation**: Simulates different scenarios to help metallurgists make informed decisions and adapt quickly to changing requirements.

## Website and Lime Calculation

This project also includes a website that features a lime calculation tool, using machine learning models to enhance prediction accuracy. The website was developed with HTML, CSS, and JavaScript, and deployed using Vercel. The project utilizes Git and GitHub for version control and collaboration, along with Excel and Power BI for data analysis and visualization.

## Requirements

- **Python 3.x**
- **Required Python Libraries**:
  - `pandas`
  - `scipy`
  - `openpyxl`


## Usage

### 1. Prepare Input Files

Ensure the following Excel files are prepared and structured correctly:

- **`efficiency_cost.xlsx`**: Contains the efficiencies and costs of each alloy.
  - **Columns**: `Alloy`, `Cost`, `C Efficiency`, `Mn Efficiency`, `S Efficiency`, `P Efficiency`, `Si Efficiency`, `Al Efficiency`, `Nb Efficiency`.
  
- **`grade_chemistry.xlsx`**: Contains the target chemistry for elements (C, Mn, S, P, Si, Al, Nb) in terms of minimum and maximum allowable values.
  - **Columns**: `Element`, `Min Value`, `Max Value`.
  

- **`initial_weights.xlsx`**: Contains the initial weights of the alloys available for addition.
  - **Columns**: `Alloy`, `Weight (kg)`.
  

### 2. Run the Script

Execute the Python script `lrf_optimization.py`:


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


### 3. Output

The script will output the optimal amounts of each alloy to be added to achieve the desired chemical composition at the lowest cost, along with the total cost.


## Project Deployment

The website is deployed using Vercel, and version control is managed via Git and GitHub. The project also includes analysis tools built with Excel and Power BI to further enhance decision-making processes.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

This project was developed to assist in the optimization of alloy additions in the metallurgical process, with a focus on cost-effectiveness, precision, and improving overall operational efficiency.


