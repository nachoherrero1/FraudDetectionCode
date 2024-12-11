using Pkg
Pkg.add(["CSV", "DataFrames", "MLJ", "MLJBase", "MLJModels", "StatsBase", "Plots", "CategoricalArrays", "MLJDecisionTreeInterface"])

# Import required libraries
using CSV
using DataFrames
import DataFrames: select, Not
using MLJ
using MLJBase
using StatsBase
using Plots
using CategoricalArrays
using MLJDecisionTreeInterface

# Load the dataset
dataset = CSV.read("creditcard.csv", DataFrame)

# Display the first few rows (optional)
println(first(dataset, 5))

# Separate features and target variable
X = select(dataset, Not(:Class))  # Features
y = dataset.Class                 # Target variable (0: non-fraud, 1: fraud)

# Ensure the target variable is categorical
y = categorical(y)

# Normalize features (assuming numerical values)
X_normalized = mapcols(zscore, X)

# Split data into training and testing sets
train_idx, test_idx = partition(eachindex(y), 0.7)  # 70% training, 30% testing
X_train, X_test = X_normalized[train_idx, :], X_normalized[test_idx, :]
y_train, y_test = y[train_idx], y[test_idx]

# Load a Random Forest classifier
RandomForest = @load RandomForestClassifier pkg=DecisionTree

# Initialize the model
rf_model = RandomForest(n_trees=60, max_depth=3)

# Wrap the model and data
rf_machine = machine(rf_model, X_train, y_train)

# Train the model
MLJ.fit!(rf_machine)

# Make predictions
y_pred = predict_mode(rf_machine, X_test)

# Evaluate accuracy
accuracy = mean(y_pred .== y_test)
println("Accuracy: ", accuracy)

# Evaluate using Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
println("Confusion Matrix:")
println(cm)