"""
Copyright (C) Microsoft Corporation. All rights reserved.​
 ​
Microsoft Corporation (“Microsoft”) grants you a nonexclusive, perpetual,
royalty-free right to use, copy, and modify the software code provided by us
("Software Code"). You may not sublicense the Software Code or any use of it
(except to your affiliates and to vendors to perform work on your behalf)
through distribution, network access, service agreement, lease, rental, or
otherwise. This license does not purport to express any claim of ownership over
data you may have shared with Microsoft in the creation of the Software Code.
Unless applicable law gives you more rights, Microsoft reserves all other
rights not expressly granted herein, whether by implication, estoppel or
otherwise. ​
 ​
THE SOFTWARE CODE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS
OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
MICROSOFT OR ITS LICENSORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER
IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THE SOFTWARE CODE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
"""

import os
from sklearn.datasets import load_diabetes
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import joblib
import pandas as pd
import mlflow
import mlflow.sklearn
import argparse

# Function to load and prepare the data
def load_data():
    sample_data = load_diabetes()
    df = pd.DataFrame(data=sample_data.data, columns=sample_data.feature_names)
    df['Y'] = sample_data.target
    return df

# Function to split the data into train and test sets
def split_data(df):
    X = df.drop('Y', axis=1).values
    y = df['Y'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    data = {"train": {"X": X_train, "y": y_train},
            "test": {"X": X_test, "y": y_test}}
    return data

# Function to train the model
def train_model(data, alpha):
    reg_model = Ridge(alpha=alpha)
    reg_model.fit(data["train"]["X"], data["train"]["y"])
    return reg_model

# Function to evaluate the model
def evaluate_model(model, data):
    preds = model.predict(data["test"]["X"])
    mse = mean_squared_error(preds, data["test"]["y"])
    return {"mse": mse}

# Function to save the model
def save_model(model, model_name):
    joblib.dump(value=model, filename=model_name)

# Main function to orchestrate the workflow
def main(alpha, model_name):
    # Start MLflow run
    with mlflow.start_run():
        df = load_data()
        data = split_data(df)
        model = train_model(data, alpha)
        metrics = evaluate_model(model, data)
        
        # Log parameters and metrics with MLflow
        mlflow.log_param("alpha", alpha)
        mlflow.log_metrics(metrics)
        
        # Save and log the model with MLflow
        save_model(model, model_name)
        mlflow.sklearn.log_model(model, "model")
        mlflow.log_artifact(model_name)

        print(f"Model saved as {model_name}")
        print(f"Metrics: {metrics}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Ridge Regression model.")
    parser.add_argument("--alpha", type=float, default=0.5, help="Regularization strength")
    parser.add_argument("--model_name", type=str, default="sklearn_regression_model.pkl", help="Name of the output model file")
    args = parser.parse_args()
    
    main(args.alpha, args.model_name)
