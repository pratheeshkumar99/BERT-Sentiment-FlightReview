# BERT-Sentiment-FlightReviews




### Introduction

This repository hosts a PyTorch-based machine learning model for sentiment analysis of airline tweets. The model leverages the BERT architecture, fine-tuned to classify tweets into three sentiment categories: positive, neutral, and negative. This project also includes a Streamlit application that serves as an interactive interface for real-time sentiment analysis.

## Features

- **BERT Sentiment Classification**: Utilizes a pre-trained BERT model fine-tuned for sentiment classification of airline tweets.
- **Interactive Web Application**: A Streamlit application that provides an easy-to-use interface for predicting sentiments directly from user input.
- **Comprehensive Data Visualization**: Generating visual insights into the dataset, such as sentiment distribution and tweet length histograms.
- **Detailed Performance Metrics**: Outputs performance metrics such as accuracy, F1-score, precision, and recall to evaluate model effectiveness.


## Project Structure

The repository is organized as follows:

Each component is crucial for the setup and execution of the sentiment analysis model:
- **app.py**: The Streamlit application file for interactive sentiment prediction.
- **model/**: Contains the saved BERT model and tokenizer necessary for running the predictions.
- **requirements.txt**: Lists all the Python dependencies required to run the project.
- **data/Tweets.csv**: The dataset used for training the model, featuring airline tweets labeled by sentiment.
- **Twitter_US_Airline_sentiment.ipynb**: A Jupyter notebook detailing the process of data handling, analysis, and model training.

This structure ensures that users can easily navigate and utilize the components of this sentiment analysis project.

### Dataset

The model is trained on a dataset consisting of airline tweets tagged with sentiments. This dataset allows the model to understand and predict the sentiment of unseen airline-related tweets based on the training it has received.

### Installation

- Prerequisites
    - Python 3.x
    - pip

- Setup

Clone this repository and install the necessary Python packages using the following commands:

```python
git clone https://github.com/pratheeshkumar99/BERT-Sentiment-FlightReview.git
cd BERT-Sentiment-FlightReview
pip install -r requirements.txt
```
