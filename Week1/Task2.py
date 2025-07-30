import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.datasets import fetch_california_housing, load_diabetes, load_wine
import warnings
warnings.filterwarnings('ignore')

def detect_outliers_iqr(df, column):
    """Detect outliers using IQR method"""
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    return outliers, lower_bound, upper_bound

def detect_outliers_zscore(df, column, threshold=3):
    """Detect outliers using Z-score method"""
    z_scores = np.abs((df[column] - df[column].mean()) / df[column].std())
    outliers = df[z_scores > threshold]
    return outliers

def main():
    st.set_page_config(page_title="Data Cleaning Toolkit", layout="wide")
    
    st.title("🧹 Data Cleaning Toolkit")
    st.write("Comprehensive data cleaning with missing value handling, outlier detection, and normalization")
    
    # Sidebar for options
    st.sidebar.header("📊 Data Source")
    
    data_option = st.sidebar.selectbox(
        "Choose data source",
        ["Upload CSV", "Sample Dataset (Boston Housing)", "Sample Dataset (Diabetes)", "Sample Dataset (Wine)"]
    )
    
    # Load data
    df = None
    if data_option == "Upload CSV":
        uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
        else:
            st.info("Please upload a CSV file to get started!")
            return

    elif data_option == "Sample Dataset (Boston Housing)":
        # Use California housing as a replacement for Boston housing
        housing = fetch_california_housing(as_frame=True)
        df = housing.frame
        df['target'] = df['MedHouseVal']
        # Introduce some missing values for demonstration
        np.random.seed(42)
        missing_indices = np.random.choice(df.index, size=int(0.1 * len(df)), replace=False)
        missing_cols = np.random.choice(df.columns[:5], size=len(missing_indices))
        for i, col in zip(missing_indices, missing_cols):
            df.loc[i, col] = np.nan

    elif data_option == "Sample Dataset (Diabetes)":
        diabetes = load_diabetes()
        df = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
        df['target'] = diabetes.target
        # Introduce some missing values
        np.random.seed(42)
        missing_indices = np.random.choice(df.index, size=int(0.08 * len(df)), replace=False)
        missing_cols = np.random.choice(df.columns[:5], size=len(missing_indices))
        for i, col in zip(missing_indices, missing_cols):
            df.loc[i, col] = np.nan

    elif data_option == "Sample Dataset (Wine)":
        wine = load_wine()
        df = pd.DataFrame(wine.data, columns=wine.feature_names)
        df['target'] = wine.target
        # Introduce some missing values
        np.random.seed(42)
        missing_indices = np.random.choice(df.index, size=int(0.05 * len(df)), replace=False)
        missing_cols = np.random.choice(df.columns[:5], size=len(missing_indices))
        for i, col in zip(missing_indices, missing_cols):
            df.loc[i, col] = np.nan

    if df is not None:
        # Create tabs for different cleaning steps
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📋 Data Overview", 
            "🔍 Missing Values", 
            "⚠️ Outlier Detection", 
            "📏 Normalization", 
            "💾 Export Results"
        ])
        
        # Store original data
        original_df = df.copy()
        
        with tab1:
            st.subheader("📊 Dataset Overview")
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.write("**Basic Information:**")
                st.write(f"Shape: {df.shape}")
                st.write(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024:.2f} KB")
                
                # Data types
                st.write("**Data Types:**")
                dtype_df = pd.DataFrame({
                    'Column': df.dtypes.index,
                    'Data Type': df.dtypes.values,
                    'Non-Null Count': df.count().values,
                    'Null Count': df.isnull().sum().values
                })
                st.dataframe(dtype_df, use_container_width=True)
            
            with col2:
                st.write("**Statistical Summary:**")
                st.dataframe(df.describe(), use_container_width=True)
            
            st.write("**Data Preview:**")
            st.dataframe(df.head(10), use_container_width=True)
            
            # Missing values heatmap
            if df.isnull().sum().sum() > 0:
                st.write("**Missing Values Pattern:**")
                fig, ax = plt.subplots(figsize=(12, 6))
                sns.heatmap(df.isnull(), cbar=True, cmap='viridis', ax=ax)
                plt.title("Missing Values Heatmap")
                st.pyplot(fig)
        
        with tab2:
            st.subheader("🔍 Missing Values Analysis & Treatment")
            
            missing_summary = df.isnull().sum()
            missing_percentage = (df.isnull().sum() / len(df)) * 100
            
            if missing_summary.sum() > 0:
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    st.write("**Missing Values Summary:**")
                    missing_df = pd.DataFrame({
                        'Column': missing_summary.index,
                        'Missing Count': missing_summary.values,
                        'Missing %': missing_percentage.values
                    })
                    missing_df = missing_df[missing_df['Missing Count'] > 0]
                    st.dataframe(missing_df, use_container_width=True)
                    
                    # Missing values bar chart
                    if len(missing_df) > 0:
                        fig = px.bar(missing_df, x='Column', y='Missing %', 
                                   title="Missing Values Percentage by Column")
                        st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    st.write("**Treatment Options:**")
                    
                    # Select columns to treat
                    cols_with_missing = missing_df['Column'].tolist()
                    selected_cols = st.multiselect("Select columns to treat:", cols_with_missing, default=cols_with_missing)
                    
                    # Treatment method
                    treatment_method = st.selectbox(
                        "Choose treatment method:",
                        ["Drop rows with missing values", "Mean imputation", "Median imputation", 
                         "Mode imputation", "Forward fill", "Backward fill", "KNN imputation"]
                    )
                    
                    if st.button("Apply Treatment", type="primary"):
                        df_treated = df.copy()
                        
                        if treatment_method == "Drop rows with missing values":
                            df_treated = df_treated.dropna(subset=selected_cols)
                        elif treatment_method == "Mean imputation":
                            imputer = SimpleImputer(strategy='mean')
                            for col in selected_cols:
                                if df_treated[col].dtype in ['int64', 'float64']:
                                    df_treated[col] = imputer.fit_transform(df_treated[[col]]).flatten()
                        elif treatment_method == "Median imputation":
                            imputer = SimpleImputer(strategy='median')
                            for col in selected_cols:
                                if df_treated[col].dtype in ['int64', 'float64']:
                                    df_treated[col] = imputer.fit_transform(df_treated[[col]]).flatten()
                        elif treatment_method == "Mode imputation":
                            imputer = SimpleImputer(strategy='most_frequent')
                            for col in selected_cols:
                                df_treated[col] = imputer.fit_transform(df_treated[[col]]).flatten()
                        elif treatment_method == "Forward fill":
                            df_treated[selected_cols] = df_treated[selected_cols].fillna(method='ffill')
                        elif treatment_method == "Backward fill":
                            df_treated[selected_cols] = df_treated[selected_cols].fillna(method='bfill')
                        elif treatment_method == "KNN imputation":
                            numeric_cols = [col for col in selected_cols if df_treated[col].dtype in ['int64', 'float64']]
                            if numeric_cols:
                                imputer = KNNImputer(n_neighbors=5)
                                df_treated[numeric_cols] = imputer.fit_transform(df_treated[numeric_cols])
                        
                        # Store treated data in session state
                        st.session_state.df_treated = df_treated
                        st.success(f"Treatment applied! Missing values reduced from {missing_summary.sum()} to {df_treated.isnull().sum().sum()}")
                        
                        # Show comparison
                        col_a, col_b = st.columns(2)
                        with col_a:
                            st.write("**Before Treatment:**")
                            st.write(f"Missing values: {df.isnull().sum().sum()}")
                            st.write(f"Shape: {df.shape}")
                        with col_b:
                            st.write("**After Treatment:**")
                            st.write(f"Missing values: {df_treated.isnull().sum().sum()}")
                            st.write(f"Shape: {df_treated.shape}")
            else:
                st.success("✅ No missing values found in the dataset!")
                st.session_state.df_treated = df.copy()
        
        with tab3:
            st.subheader("⚠️ Outlier Detection & Removal")
            
            # Use treated data if available
            working_df = st.session_state.get('df_treated', df).copy()
            numeric_cols = working_df.select_dtypes(include=[np.number]).columns.tolist()
            
            if len(numeric_cols) > 0:
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    st.write("**Outlier Detection Settings:**")
                    
                    selected_col = st.selectbox("Select column for outlier analysis:", numeric_cols)
                    detection_method = st.selectbox("Detection method:", ["IQR Method", "Z-Score Method"])
                    
                    if detection_method == "Z-Score Method":
                        z_threshold = st.slider("Z-score threshold:", 1.5, 4.0, 3.0, 0.1)
                    
                    # Detect outliers
                    if detection_method == "IQR Method":
                        outliers, lower_bound, upper_bound = detect_outliers_iqr(working_df, selected_col)
                        st.write(f"**IQR Bounds:** [{lower_bound:.2f}, {upper_bound:.2f}]")
                    else:
                        outliers = detect_outliers_zscore(working_df, selected_col, z_threshold)
                    
                    st.write(f"**Outliers detected:** {len(outliers)} ({len(outliers)/len(working_df)*100:.1f}%)")
                    
                    # Outlier treatment options
                    if len(outliers) > 0:
                        treatment_option = st.selectbox(
                            "Outlier treatment:",
                            ["Remove outliers", "Cap outliers (Winsorization)", "Keep outliers"]
                        )
                        
                        if st.button("Apply Outlier Treatment", type="primary"):
                            df_outlier_treated = working_df.copy()
                            
                            if treatment_option == "Remove outliers":
                                df_outlier_treated = df_outlier_treated.drop(outliers.index)
                            elif treatment_option == "Cap outliers (Winsorization)":
                                if detection_method == "IQR Method":
                                    df_outlier_treated[selected_col] = df_outlier_treated[selected_col].clip(
                                        lower=lower_bound, upper=upper_bound
                                    )
                                else:
                                    percentile_95 = df_outlier_treated[selected_col].quantile(0.95)
                                    percentile_05 = df_outlier_treated[selected_col].quantile(0.05)
                                    df_outlier_treated[selected_col] = df_outlier_treated[selected_col].clip(
                                        lower=percentile_05, upper=percentile_95
                                    )
                            
                            st.session_state.df_outlier_treated = df_outlier_treated
                            st.success(f"Outlier treatment applied! Dataset shape: {df_outlier_treated.shape}")
                
                with col2:
                    st.write("**Outlier Visualization:**")
                    
                    # Box plot
                    fig = go.Figure()
                    fig.add_trace(go.Box(y=working_df[selected_col], name=selected_col))
                    if len(outliers) > 0:
                        fig.add_trace(go.Scatter(
                            x=[selected_col] * len(outliers),
                            y=outliers[selected_col],
                            mode='markers',
                            name='Outliers',
                            marker=dict(color='red', size=8)
                        ))
                    fig.update_layout(title=f"Box Plot - {selected_col}")
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Histogram
                    fig = px.histogram(working_df, x=selected_col, nbins=30, title=f"Distribution - {selected_col}")
                    if len(outliers) > 0:
                        fig.add_vline(x=outliers[selected_col].min(), line_dash="dash", line_color="red")
                        fig.add_vline(x=outliers[selected_col].max(), line_dash="dash", line_color="red")
                    st.plotly_chart(fig, use_container_width=True)
                
                # Show outliers data
                if len(outliers) > 0:
                    st.write("**Detected Outliers:**")
                    st.dataframe(outliers, use_container_width=True)
            else:
                st.warning("No numeric columns found for outlier detection.")
        
        with tab4:
            st.subheader("📏 Data Normalization & Standardization")
            
            # Use the most processed data available
            working_df = st.session_state.get('df_outlier_treated', 
                        st.session_state.get('df_treated', df)).copy()
            
            numeric_cols = working_df.select_dtypes(include=[np.number]).columns.tolist()
            
            if len(numeric_cols) > 0:
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    st.write("**Normalization Settings:**")
                    
                    selected_cols_norm = st.multiselect(
                        "Select columns to normalize:", 
                        numeric_cols, 
                        default=numeric_cols[:5] if len(numeric_cols) > 5 else numeric_cols
                    )
                    
                    normalization_method = st.selectbox(
                        "Choose normalization method:",
                        ["StandardScaler (Z-score)", "MinMaxScaler (0-1)", "RobustScaler (Median & IQR)"]
                    )
                    
                    if st.button("Apply Normalization", type="primary"):
                        df_normalized = working_df.copy()
                        
                        if normalization_method == "StandardScaler (Z-score)":
                            scaler = StandardScaler()
                        elif normalization_method == "MinMaxScaler (0-1)":
                            scaler = MinMaxScaler()
                        else:
                            scaler = RobustScaler()
                        
                        df_normalized[selected_cols_norm] = scaler.fit_transform(df_normalized[selected_cols_norm])
                        
                        st.session_state.df_normalized = df_normalized
                        st.session_state.scaler = scaler
                        st.success("Normalization applied successfully!")
                        
                        # Show statistics comparison
                        st.write("**Before vs After Normalization:**")
                        comparison_df = pd.DataFrame({
                            'Column': selected_cols_norm,
                            'Original Mean': working_df[selected_cols_norm].mean().values,
                            'Normalized Mean': df_normalized[selected_cols_norm].mean().values,
                            'Original Std': working_df[selected_cols_norm].std().values,
                            'Normalized Std': df_normalized[selected_cols_norm].std().values
                        })
                        st.dataframe(comparison_df, use_container_width=True)
                
                with col2:
                    if 'df_normalized' in st.session_state:
                        st.write("**Normalization Visualization:**")
                        
                        # Select column for visualization
                        viz_col = st.selectbox("Select column for visualization:", selected_cols_norm)
                        
                        # Before/After comparison
                        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
                        
                        ax1.hist(working_df[viz_col], bins=30, alpha=0.7, color='blue')
                        ax1.set_title(f"Before Normalization - {viz_col}")
                        ax1.set_xlabel("Value")
                        ax1.set_ylabel("Frequency")
                        
                        ax2.hist(st.session_state.df_normalized[viz_col], bins=30, alpha=0.7, color='red')
                        ax2.set_title(f"After Normalization - {viz_col}")
                        ax2.set_xlabel("Normalized Value")
                        ax2.set_ylabel("Frequency")
                        
                        plt.tight_layout()
                        st.pyplot(fig)
                        
                        # Distribution comparison using plotly
                        fig = go.Figure()
                        fig.add_trace(go.Histogram(
                            x=working_df[viz_col],
                            name="Original",
                            opacity=0.7,
                            nbinsx=30
                        ))
                        fig.add_trace(go.Histogram(
                            x=st.session_state.df_normalized[viz_col],
                            name="Normalized",
                            opacity=0.7,
                            nbinsx=30
                        ))
                        fig.update_layout(
                            title=f"Distribution Comparison - {viz_col}",
                            barmode='overlay'
                        )
                        st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No numeric columns found for normalization.")
        
        with tab5:
            st.subheader("💾 Export Cleaned Data")
            
            # Get the final processed data
            final_df = st.session_state.get('df_normalized',
                      st.session_state.get('df_outlier_treated',
                      st.session_state.get('df_treated', df)))
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.write("**Cleaning Summary:**")
                
                cleaning_steps = []
                if 'df_treated' in st.session_state:
                    missing_reduced = df.isnull().sum().sum() - st.session_state.df_treated.isnull().sum().sum()
                    cleaning_steps.append(f"✅ Missing values handled: {missing_reduced} values treated")
                
                if 'df_outlier_treated' in st.session_state:
                    outliers_removed = len(st.session_state.get('df_treated', df)) - len(st.session_state.df_outlier_treated)
                    cleaning_steps.append(f"✅ Outliers handled: {outliers_removed} rows processed")
                
                if 'df_normalized' in st.session_state:
                    norm_cols = len([col for col in final_df.columns if final_df[col].dtype in ['int64', 'float64']])
                    cleaning_steps.append(f"✅ Normalization applied: {norm_cols} numeric columns")
                
                if not cleaning_steps:
                    cleaning_steps.append("ℹ️ No cleaning steps applied yet")
                
                for step in cleaning_steps:
                    st.write(step)
                
                st.write(f"**Final dataset shape:** {final_df.shape}")
                st.write(f"**Missing values:** {final_df.isnull().sum().sum()}")
            
            with col2:
                st.write("**Export Options:**")
                
                # File format
                export_format = st.selectbox("Choose export format:", ["CSV", "Excel", "JSON"])
                
                # Generate download
                if export_format == "CSV":
                    csv_data = final_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Cleaned Data (CSV)",
                        data=csv_data,
                        file_name="cleaned_data.csv",
                        mime="text/csv"
                    )
                elif export_format == "Excel":
                    # Create Excel file in memory
                    from io import BytesIO
                    output = BytesIO()
                    with pd.ExcelWriter(output, engine='openpyxl') as writer:
                        final_df.to_excel(writer, sheet_name='Cleaned_Data', index=False)
                        if df.equals(final_df) == False:
                            original_df.to_excel(writer, sheet_name='Original_Data', index=False)
                    
                    st.download_button(
                        label="📥 Download Cleaned Data (Excel)",
                        data=output.getvalue(),
                        file_name="cleaned_data.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
                else:  # JSON
                    json_data = final_df.to_json(orient='records', indent=2)
                    st.download_button(
                        label="📥 Download Cleaned Data (JSON)",
                        data=json_data,
                        file_name="cleaned_data.json",
                        mime="application/json"
                    )
            
            # Final data preview
            st.write("**Final Cleaned Data Preview:**")
            st.dataframe(final_df.head(10), use_container_width=True)
            
            # Data quality report
            st.write("**Data Quality Report:**")
            quality_metrics = {
                'Metric': [
                    'Total Rows',
                    'Total Columns', 
                    'Missing Values',
                    'Duplicate Rows',
                    'Numeric Columns',
                    'Categorical Columns'
                ],
                'Original Data': [
                    len(original_df),
                    len(original_df.columns),
                    original_df.isnull().sum().sum(),
                    original_df.duplicated().sum(),
                    len(original_df.select_dtypes(include=[np.number]).columns),
                    len(original_df.select_dtypes(exclude=[np.number]).columns)
                ],
                'Cleaned Data': [
                    len(final_df),
                    len(final_df.columns),
                    final_df.isnull().sum().sum(),
                    final_df.duplicated().sum(),
                    len(final_df.select_dtypes(include=[np.number]).columns),
                    len(final_df.select_dtypes(exclude=[np.number]).columns)
                ]
            }
            
            quality_df = pd.DataFrame(quality_metrics)
            st.dataframe(quality_df, use_container_width=True)

if __name__ == "__main__":
    main()