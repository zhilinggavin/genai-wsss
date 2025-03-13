#!/bin/bash

# Define the project name
PROJECT_NAME=$PWD

# # Create the main project directory
# mkdir -p $PROJECT_NAME

# Create subdirectories
mkdir -p $PROJECT_NAME/{data/{raw,processed},notebooks,src,configs,experiments/{logs,checkpoints,results,scripts},tests,scripts}

# Create empty essential files
touch $PROJECT_NAME/{README.md,requirements.txt,environment.yml,setup.py,.gitignore,LICENSE}
touch $PROJECT_NAME/configs/config.yaml
touch $PROJECT_NAME/scripts/{setup_env.sh,train.sh,inference.sh}
touch $PROJECT_NAME/src/{data_loader.py,model.py,train.py,evaluate.py,inference.py,utils.py}
touch $PROJECT_NAME/tests/{test_data_loader.py,test_model.py}
touch $PROJECT_NAME/notebooks/{01_data_exploration.ipynb,02_model_training.ipynb,03_evaluation.ipynb}
touch $PROJECT_NAME/experiments/scripts/{experiment_1.py,experiment_2.py}

# Add a basic README.md
echo "# Deep Learning Project" > $PROJECT_NAME/README.md
echo "This repository contains a deep learning project with structured code, data handling, and experiments." >> $PROJECT_NAME/README.md

# Add a basic .gitignore file
echo "__pycache__/" > $PROJECT_NAME/.gitignore
echo ".ipynb_checkpoints/" >> $PROJECT_NAME/.gitignore
echo "data/raw/" >> $PROJECT_NAME/.gitignore
echo "data/processed/" >> $PROJECT_NAME/.gitignore
echo "experiments/checkpoints/" >> $PROJECT_NAME/.gitignore
echo "experiments/logs/" >> $PROJECT_NAME/.gitignore

# Print completion message
echo "Deep learning project structure created successfully in '$PROJECT_NAME/' 🎉"