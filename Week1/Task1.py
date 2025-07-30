import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import make_regression
import plotly.express as px
import plotly.graph_objects as go

class LinearRegression:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
        self.cost_history = []
    
    def fit(self, X, y):
        # Initialize parameters
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        self.cost_history = []
        
        # Gradient descent
        for i in range(self.n_iterations):
            # Forward pass - predictions
            y_pred = np.dot(X, self.weights) + self.bias
            
            # Calculate cost (Mean Squared Error)
            cost = (1 / (2 * n_samples)) * np.sum((y_pred - y) ** 2)
            self.cost_history.append(cost)
            
            # Calculate gradients
            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
            db = (1 / n_samples) * np.sum(y_pred - y)
            
            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
    
    def predict(self, X):
        return np.dot(X, self.weights) + self.bias

def main():
    st.set_page_config(page_title="Linear Regression from Scratch", layout="wide")
    
    st.title("🔍 Linear Regression from Scratch")
    st.write("Interactive implementation using NumPy with gradient descent optimization")
    
    # Sidebar for parameters
    st.sidebar.header("📊 Model Parameters")
    
    # Dataset parameters
    st.sidebar.subheader("Dataset Settings")
    n_samples = st.sidebar.slider("Number of samples", 50, 500, 100)
    noise_level = st.sidebar.slider("Noise level", 0, 50, 20)
    random_seed = st.sidebar.number_input("Random seed", 0, 100, 42)
    
    # Model parameters
    st.sidebar.subheader("Training Settings")
    learning_rate = st.sidebar.slider("Learning rate", 0.001, 0.1, 0.01, 0.001)
    n_iterations = st.sidebar.slider("Number of iterations", 100, 2000, 1000, 100)
    
    # Generate or upload data
    data_option = st.sidebar.selectbox("Data source", ["Generate synthetic data", "Upload CSV file"])
    
    if data_option == "Generate synthetic data":
        # Generate sample dataset
        np.random.seed(random_seed)
        X, y = make_regression(n_samples=n_samples, n_features=1, noise=noise_level, random_state=random_seed)
        
        st.sidebar.success(f"Generated {n_samples} samples with noise level {noise_level}")
        
    else:
        uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            st.sidebar.write("Data preview:")
            st.sidebar.write(df.head())
            
            # Let user select columns
            x_column = st.sidebar.selectbox("Select X column", df.columns)
            y_column = st.sidebar.selectbox("Select y column", df.columns)
            
            X = df[x_column].values.reshape(-1, 1)
            y = df[y_column].values
        else:
            st.warning("Please upload a CSV file or use synthetic data")
            return
    
    # Train model button
    if st.sidebar.button("🚀 Train Model", type="primary"):
        with st.spinner("Training model..."):
            # Create and train the model
            model = LinearRegression(learning_rate=learning_rate, n_iterations=n_iterations)
            model.fit(X, y)
            
            # Make predictions
            y_pred = model.predict(X)
            
            # Calculate metrics
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r_squared = 1 - (ss_res / ss_tot)
            residuals = y - y_pred
            mae = np.mean(np.abs(residuals))
            rmse = np.sqrt(np.mean(residuals**2))
            
            # Store results in session state
            st.session_state.model = model
            st.session_state.X = X
            st.session_state.y = y
            st.session_state.y_pred = y_pred
            st.session_state.metrics = {
                'r_squared': r_squared,
                'mae': mae,
                'rmse': rmse,
                'weight': model.weights[0],
                'bias': model.bias
            }
            st.session_state.residuals = residuals
    
    # Display results if model has been trained
    if hasattr(st.session_state, 'model'):
        col1, col2 = st.columns([2, 1])
        
        with col2:
            st.subheader("📈 Model Performance")
            metrics = st.session_state.metrics
            
            st.metric("R-squared", f"{metrics['r_squared']:.4f}")
            st.metric("Mean Absolute Error", f"{metrics['mae']:.4f}")
            st.metric("Root Mean Squared Error", f"{metrics['rmse']:.4f}")
            
            st.subheader("🔧 Model Parameters")
            st.write(f"**Weight (slope):** {metrics['weight']:.4f}")
            st.write(f"**Bias (intercept):** {metrics['bias']:.4f}")
            
            # Model equation
            st.subheader("📝 Linear Equation")
            if metrics['bias'] >= 0:
                equation = f"y = {metrics['weight']:.4f}x + {metrics['bias']:.4f}"
            else:
                equation = f"y = {metrics['weight']:.4f}x - {abs(metrics['bias']):.4f}"
            st.code(equation)
        
        with col1:
            st.subheader("📊 Visualizations")
            
            # Create tabs for different plots
            tab1, tab2, tab3, tab4 = st.tabs(["Regression Line", "Cost Function", "Residuals", "Actual vs Predicted"])
            
            with tab1:
                # Regression line plot
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=st.session_state.X.flatten(), 
                    y=st.session_state.y,
                    mode='markers',
                    name='Data points',
                    marker=dict(color='blue', opacity=0.6)
                ))
                fig.add_trace(go.Scatter(
                    x=st.session_state.X.flatten(),
                    y=st.session_state.y_pred,
                    mode='lines',
                    name='Regression line',
                    line=dict(color='red', width=3)
                ))
                fig.update_layout(
                    title="Linear Regression: Data Points and Fitted Line",
                    xaxis_title="X",
                    yaxis_title="y",
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with tab2:
                # Cost function plot
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=list(range(len(st.session_state.model.cost_history))),
                    y=st.session_state.model.cost_history,
                    mode='lines',
                    name='Cost',
                    line=dict(color='green', width=2)
                ))
                fig.update_layout(
                    title="Cost Function Over Iterations",
                    xaxis_title="Iterations",
                    yaxis_title="Cost (MSE)",
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with tab3:
                # Residuals plot
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=st.session_state.y_pred,
                    y=st.session_state.residuals,
                    mode='markers',
                    name='Residuals',
                    marker=dict(color='purple', opacity=0.6)
                ))
                fig.add_hline(y=0, line_dash="dash", line_color="red")
                fig.update_layout(
                    title="Residuals Plot",
                    xaxis_title="Predicted values",
                    yaxis_title="Residuals",
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with tab4:
                # Actual vs Predicted plot
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=st.session_state.y,
                    y=st.session_state.y_pred,
                    mode='markers',
                    name='Predictions',
                    marker=dict(color='orange', opacity=0.6)
                ))
                # Perfect prediction line
                min_val, max_val = min(st.session_state.y.min(), st.session_state.y_pred.min()), max(st.session_state.y.max(), st.session_state.y_pred.max())
                fig.add_trace(go.Scatter(
                    x=[min_val, max_val],
                    y=[min_val, max_val],
                    mode='lines',
                    name='Perfect prediction',
                    line=dict(color='red', dash='dash', width=2)
                ))
                fig.update_layout(
                    title="Actual vs Predicted Values",
                    xaxis_title="Actual values",
                    yaxis_title="Predicted values",
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # Data table
        st.subheader("📋 Data Table")
        df_results = pd.DataFrame({
            'X': st.session_state.X.flatten(),
            'Actual_y': st.session_state.y,
            'Predicted_y': st.session_state.y_pred,
            'Residuals': st.session_state.residuals
        })
        st.dataframe(df_results, use_container_width=True)
        
        # Download results
        csv = df_results.to_csv(index=False)
        st.download_button(
            label="📥 Download Results as CSV",
            data=csv,
            file_name='linear_regression_results.csv',
            mime='text/csv'
        )
    
    else:
        st.info("👆 Configure parameters in the sidebar and click 'Train Model' to get started!")
        
        # Show sample data preview
        if 'X' in locals():
            st.subheader("📊 Data Preview")
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=X.flatten(),
                y=y,
                mode='markers',
                name='Data points',
                marker=dict(color='blue', opacity=0.6)
            ))
            fig.update_layout(
                title="Current Dataset",
                xaxis_title="X",
                yaxis_title="y",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()