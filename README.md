# CAP6610 Assignment 1 – End-to-End ML Project

## Overview

This repository contains my work for Assignment 1 in the CAP6610 Machine Learning course at the University of North Florida, instructed by Dr. Liu. The assignment focuses on experimenting with various regression models using the dataset discussed in Chapter 2 of 
https://www.amazon.com/Hands-Machine-Learning-Scikit-Learn-TensorFlow/dp/1098125975, our class textbook, employing the scikit-learn package.

## Objective

The main objective of this assignment is to explore and evaluate the performance of different regression models, including K-Nearest Neighbors (KNeighborsRegressor) and Artificial Neural Networks (MLPRegressor), and to fine-tune the models using Grid Search (GridSearchCV).

## Steps and Key Findings

1. **K-Nearest Neighbors Model**: Implemented using 5-fold cross-validation, with `n_neighbors=5`. The performance was evaluated using the square root of Mean Squared Error (MSE) scores.
2. **Artificial Neural Networks Model**: Configured with `solver='lbfgs'`, `hidden_layer_sizes=(15,)`, `random_state=42`, `max_iter=5000`, and evaluated using 5-fold cross-validation. The square root MSE scores were printed for performance assessment.
3. **Hyperparameter Tuning with Grid Search**: Conducted a grid search for the K-Nearest Neighbors model to find the best hyperparameters among specified values for 'n_neighbors', 'p', and 'algorithm', using 5-fold cross-validation.
4. **Model Testing**: The best K-Nearest Neighbors model identified from the grid search was finally tested on `X_test_prepared`, and the square root MSE was printed.

## Installation and Setup

To replicate this project, ensure you have Python and Jupyter Notebook installed. Clone this repository, navigate to the project directory, and install the required dependencies. For this project, I used Anaconda as my Python kernel on VSC.

## Usage

Open the `assignment1.ipynb` notebook in VSC, Jupyter Notebook or any compatible IDE to view the project. The notebook contains all the steps mentioned above, along with detailed comments and results. My code for the assignment 
specifically starts under the following markdown cell. The template is referenced from https://github.com/ageron/handson-ml2 : see `02_end_to_end_machine_learning_project.ipynb`.

![image](https://github.com/Windz-GameDev/Assignment_1_ML/assets/97154040/0d0bedf8-f3fc-4d85-800d-10d372ea79de)

## Conclusion

This assignment provided hands-on experience with applying different regression models and tuning them to achieve better performance. The exploration of hyperparameters through Grid Search helped me better understand optimizing model configurations for improved predictions on unseen data.

