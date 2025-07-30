import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
from sklearn.datasets import load_iris, load_wine, load_breast_cancer, load_diabetes, fetch_california_housing
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set style for matplotlib/seaborn
plt.style.use('default')
sns.set_palette("husl")

def generate_univariate_analysis(df, column):
    """Generate comprehensive univariate analysis for a column"""
    analysis = {}
    
    if df[column].dtype in ['int64', 'float64']:
        # Numerical analysis
        analysis['type'] = 'numerical'
        analysis['stats'] = {
            'count': df[column].count(),
            'mean': df[column].mean(),
            'std': df[column].std(),
            'min': df[column].min(),
            'max': df[column].max(),
            'median': df[column].median(),
            'q1': df[column].quantile(0.25),
            'q3': df[column].quantile(0.75),
            'skewness': df[column].skew(),
            'kurtosis': df[column].kurtosis()
        }
        analysis['missing'] = df[column].isnull().sum()
        analysis['zeros'] = (df[column] == 0).sum()
        analysis['unique'] = df[column].nunique()
    else:
        # Categorical analysis
        analysis['type'] = 'categorical'
        analysis['stats'] = {
            'count': df[column].count(),
            'unique': df[column].nunique(),
            'top': df[column].mode().iloc[0] if len(df[column].mode()) > 0 else None,
            'freq': df[column].value_counts().iloc[0] if len(df[column].value_counts()) > 0 else None
        }
        analysis['missing'] = df[column].isnull().sum()
        analysis['value_counts'] = df[column].value_counts()
    
    return analysis

def create_distribution_plots(df, column):
    """Create distribution plots for numerical columns"""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Histogram', 'Box Plot', 'Q-Q Plot', 'Violin Plot'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Histogram
    fig.add_trace(
        go.Histogram(x=df[column], name="Histogram", nbinsx=30),
        row=1, col=1
    )
    
    # Box Plot
    fig.add_trace(
        go.Box(y=df[column], name="Box Plot"),
        row=1, col=2
    )
    
    # Q-Q Plot data
    sorted_data = np.sort(df[column].dropna())
    theoretical_quantiles = stats.norm.ppf(np.linspace(0.01, 0.99, len(sorted_data)))
    
    fig.add_trace(
        go.Scatter(x=theoretical_quantiles, y=sorted_data, mode='markers', name="Q-Q Plot"),
        row=2, col=1
    )
    
    # Add reference line for Q-Q plot
    fig.add_trace(
        go.Scatter(x=[theoretical_quantiles.min(), theoretical_quantiles.max()],
                  y=[sorted_data.min(), sorted_data.max()],
                  mode='lines', name="Reference Line", line=dict(dash='dash')),
        row=2, col=1
    )
    
    # Violin Plot
    fig.add_trace(
        go.Violin(y=df[column], name="Violin Plot"),
        row=2, col=2
    )
    
    fig.update_layout(height=800, title_text=f"Distribution Analysis - {column}")
    return fig

def create_categorical_plots(df, column):
    """Create plots for categorical columns"""
    value_counts = df[column].value_counts()
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Bar Plot', 'Pie Chart'),
        specs=[[{"secondary_y": False}, {"type": "domain"}]]
    )
    
    # Bar Plot
    fig.add_trace(
        go.Bar(x=value_counts.index, y=value_counts.values, name="Frequency"),
        row=1, col=1
    )
    
    # Pie Chart
    fig.add_trace(
        go.Pie(labels=value_counts.index, values=value_counts.values, name="Distribution"),
        row=1, col=2
    )
    
    fig.update_layout(height=400, title_text=f"Categorical Analysis - {column}")
    return fig

def main():
    st.set_page_config(page_title="EDA Toolkit", layout="wide")
    
    st.title("🔍 Exploratory Data Analysis (EDA) Toolkit")
    st.write("Comprehensive data exploration with univariate and multivariate analysis")
    
    # Sidebar for data loading
    st.sidebar.header("📊 Data Source")
    
    data_option = st.sidebar.selectbox(
        "Choose data source",
        ["Upload CSV", "Iris Dataset", "Wine Dataset", "Breast Cancer Dataset", "Diabetes Dataset", "California Housing Dataset"]
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
    
    elif data_option == "Iris Dataset":
        iris = load_iris()
        df = pd.DataFrame(iris.data, columns=iris.feature_names)
        df['target'] = iris.target
        df['species'] = df['target'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})
    
    elif data_option == "Wine Dataset":
        wine = load_wine()
        df = pd.DataFrame(wine.data, columns=wine.feature_names)
        df['target'] = wine.target
        df['wine_class'] = df['target'].map({0: 'Class_0', 1: 'Class_1', 2: 'Class_2'})
    
    elif data_option == "Breast Cancer Dataset":
        cancer = load_breast_cancer()
        df = pd.DataFrame(cancer.data, columns=cancer.feature_names)
        df['target'] = cancer.target
        df['diagnosis'] = df['target'].map({0: 'malignant', 1: 'benign'})
    
    elif data_option == "Diabetes Dataset":
        diabetes = load_diabetes()
        df = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
        df['target'] = diabetes.target

    elif data_option == "California Housing Dataset":
        housing = fetch_california_housing(as_frame=True)
        df = housing.frame
        df['target'] = df['MedHouseVal']
    
    if df is not None:
        # Create tabs for different analyses
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📋 Dataset Overview",
            "📊 Univariate Analysis", 
            "🔗 Bivariate Analysis",
            "📈 Multivariate Analysis",
            "📋 Statistical Summary"
        ])
        
        with tab1:
            st.subheader("📊 Dataset Overview")
            
            col1, col2, col3 = st.columns([1, 1, 1])
            
            with col1:
                st.metric("Number of Rows", df.shape[0])
                st.metric("Number of Columns", df.shape[1])
                st.metric("Memory Usage (KB)", f"{df.memory_usage(deep=True).sum() / 1024:.2f}")
            
            with col2:
                numeric_cols = len(df.select_dtypes(include=[np.number]).columns)
                categorical_cols = len(df.select_dtypes(exclude=[np.number]).columns)
                st.metric("Numeric Columns", numeric_cols)
                st.metric("Categorical Columns", categorical_cols)
                st.metric("Missing Values", df.isnull().sum().sum())
            
            with col3:
                duplicate_rows = df.duplicated().sum()
                st.metric("Duplicate Rows", duplicate_rows)
                st.metric("Unique Rows", df.shape[0] - duplicate_rows)
                st.metric("Completeness %", f"{((df.size - df.isnull().sum().sum()) / df.size * 100):.1f}")
            
            # Data types and missing values
            st.write("**Column Information:**")
            info_df = pd.DataFrame({
                'Column': df.columns,
                'Data Type': df.dtypes.values,
                'Non-Null Count': df.count().values,
                'Null Count': df.isnull().sum().values,
                'Null %': (df.isnull().sum() / len(df) * 100).values,
                'Unique Values': [df[col].nunique() for col in df.columns]
            })
            st.dataframe(info_df, use_container_width=True)
            
            # Data preview
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
            st.subheader("📊 Univariate Analysis")
            
            # Column selection
            analysis_col = st.selectbox("Select column for detailed analysis:", df.columns)
            
            # Generate analysis
            analysis = generate_univariate_analysis(df, analysis_col)
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.write("**Statistical Summary:**")
                
                if analysis['type'] == 'numerical':
                    stats_df = pd.DataFrame({
                        'Statistic': list(analysis['stats'].keys()),
                        'Value': [f"{v:.4f}" if isinstance(v, float) else str(v) for v in analysis['stats'].values()]
                    })
                    st.dataframe(stats_df, use_container_width=True)
                    
                    # Additional insights
                    st.write("**Data Quality:**")
                    st.write(f"Missing Values: {analysis['missing']} ({analysis['missing']/len(df)*100:.1f}%)")
                    st.write(f"Zero Values: {analysis['zeros']} ({analysis['zeros']/len(df)*100:.1f}%)")
                    st.write(f"Unique Values: {analysis['unique']}")
                    
                    # Distribution insights
                    st.write("**Distribution Insights:**")
                    if abs(analysis['stats']['skewness']) < 0.5:
                        st.write("✅ Approximately symmetric distribution")
                    elif analysis['stats']['skewness'] > 0.5:
                        st.write("⚠️ Right-skewed distribution")
                    else:
                        st.write("⚠️ Left-skewed distribution")
                    
                    if analysis['stats']['kurtosis'] > 3:
                        st.write("📈 Heavy-tailed distribution")
                    elif analysis['stats']['kurtosis'] < 3:
                        st.write("📉 Light-tailed distribution")
                    else:
                        st.write("📊 Normal kurtosis")
                
                else:
                    # Categorical summary
                    st.write(f"**Count:** {analysis['stats']['count']}")
                    st.write(f"**Unique Values:** {analysis['stats']['unique']}")
                    st.write(f"**Most Frequent:** {analysis['stats']['top']}")
                    st.write(f"**Frequency:** {analysis['stats']['freq']}")
                    st.write(f"**Missing Values:** {analysis['missing']}")
                    
                    st.write("**Value Counts:**")
                    st.dataframe(analysis['value_counts'], use_container_width=True)
            
            with col2:
                st.write("**Visualizations:**")
                
                if analysis['type'] == 'numerical':
                    # Distribution plots
                    distribution_fig = create_distribution_plots(df, analysis_col)
                    st.plotly_chart(distribution_fig, use_container_width=True)
                    
                    # Additional statistics plot
                    fig = go.Figure()
                    
                    # Add box plot with outliers
                    fig.add_trace(go.Box(
                        y=df[analysis_col],
                        name=analysis_col,
                        boxpoints='outliers'
                    ))
                    
                    fig.update_layout(title=f"Box Plot with Outliers - {analysis_col}")
                    st.plotly_chart(fig, use_container_width=True)
                
                else:
                    # Categorical plots
                    categorical_fig = create_categorical_plots(df, analysis_col)
                    st.plotly_chart(categorical_fig, use_container_width=True)
            
            # Show all numerical distributions at once
            st.write("**All Numerical Distributions:**")
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            
            if len(numeric_cols) > 0:
                n_cols = min(3, len(numeric_cols))
                n_rows = (len(numeric_cols) + n_cols - 1) // n_cols
                
                fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
                if n_rows == 1:
                    axes = [axes] if n_cols == 1 else axes
                else:
                    axes = axes.flatten() if n_rows > 1 else [axes]
                
                for i, col in enumerate(numeric_cols):
                    if i < len(axes):
                        df[col].hist(bins=30, ax=axes[i], alpha=0.7)
                        axes[i].set_title(f'{col}')
                        axes[i].set_xlabel('Value')
                        axes[i].set_ylabel('Frequency')
                
                # Hide empty subplots
                for i in range(len(numeric_cols), len(axes)):
                    axes[i].set_visible(False)
                
                plt.tight_layout()
                st.pyplot(fig)
        
        with tab3:
            st.subheader("🔗 Bivariate Analysis")
            
            # Variable selection
            col1, col2 = st.columns(2)
            with col1:
                x_var = st.selectbox("Select X variable:", df.columns, key="x_var")
            with col2:
                y_var = st.selectbox("Select Y variable:", df.columns, key="y_var")
            
            if x_var != y_var:
                # Determine plot type based on variable types
                x_is_numeric = df[x_var].dtype in ['int64', 'float64']
                y_is_numeric = df[y_var].dtype in ['int64', 'float64']
                
                if x_is_numeric and y_is_numeric:
                    # Numeric vs Numeric
                    st.write("**Scatter Plot Analysis:**")
                    
                    # Calculate correlation
                    correlation = df[x_var].corr(df[y_var])
                    st.write(f"**Pearson Correlation:** {correlation:.4f}")
                    
                    # Scatter plot
                    fig = px.scatter(df, x=x_var, y=y_var, 
                                   title=f"Scatter Plot: {x_var} vs {y_var}",
                                   trendline="ols")
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Joint plot using seaborn
                    st.write("**Joint Distribution:**")
                    fig, ax = plt.subplots(figsize=(10, 8))
                    
                    # Create joint plot manually
                    from matplotlib.gridspec import GridSpec
                    gs = GridSpec(3, 3)
                    
                    fig = plt.figure(figsize=(10, 8))
                    ax_main = fig.add_subplot(gs[1:3, :2])
                    ax_top = fig.add_subplot(gs[0, :2])
                    ax_right = fig.add_subplot(gs[1:3, 2])
                    
                    # Main scatter plot
                    ax_main.scatter(df[x_var], df[y_var], alpha=0.6)
                    ax_main.set_xlabel(x_var)
                    ax_main.set_ylabel(y_var)
                    
                    # Top histogram
                    ax_top.hist(df[x_var], bins=30, alpha=0.7)
                    ax_top.set_xlim(ax_main.get_xlim())
                    
                    # Right histogram
                    ax_right.hist(df[y_var], bins=30, orientation='horizontal', alpha=0.7)
                    ax_right.set_ylim(ax_main.get_ylim())
                    
                    plt.suptitle(f'Joint Distribution: {x_var} vs {y_var}')
                    st.pyplot(fig)
                
                elif x_is_numeric and not y_is_numeric:
                    # Numeric vs Categorical
                    st.write("**Box Plot Analysis:**")
                    
                    fig = px.box(df, x=y_var, y=x_var,
                               title=f"Distribution of {x_var} by {y_var}")
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Violin plot
                    fig = px.violin(df, x=y_var, y=x_var,
                                  title=f"Violin Plot: {x_var} by {y_var}")
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Statistical summary by category
                    st.write("**Statistics by Category:**")
                    stats_by_cat = df.groupby(y_var)[x_var].agg([
                        'count', 'mean', 'std', 'min', 'max', 'median'
                    ]).round(4)
                    st.dataframe(stats_by_cat, use_container_width=True)
                
                elif not x_is_numeric and y_is_numeric:
                    # Categorical vs Numeric (swap for consistency)
                    st.write("**Box Plot Analysis:**")
                    
                    fig = px.box(df, x=x_var, y=y_var,
                               title=f"Distribution of {y_var} by {x_var}")
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Bar plot of means
                    mean_by_cat = df.groupby(x_var)[y_var].mean().sort_values(ascending=False)
                    fig = px.bar(x=mean_by_cat.index, y=mean_by_cat.values,
                               title=f"Mean {y_var} by {x_var}")
                    st.plotly_chart(fig, use_container_width=True)
                
                else:
                    # Categorical vs Categorical
                    st.write("**Cross-tabulation Analysis:**")
                    
                    # Create crosstab
                    crosstab = pd.crosstab(df[x_var], df[y_var])
                    st.dataframe(crosstab, use_container_width=True)
                    
                    # Heatmap
                    fig = px.imshow(crosstab, title=f"Cross-tabulation: {x_var} vs {y_var}")
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Stacked bar chart
                    crosstab_pct = pd.crosstab(df[x_var], df[y_var], normalize='index') * 100
                    
                    fig = go.Figure()
                    for col in crosstab_pct.columns:
                        fig.add_trace(go.Bar(
                            name=str(col),
                            x=crosstab_pct.index,
                            y=crosstab_pct[col]
                        ))
                    
                    fig.update_layout(
                        barmode='stack',
                        title=f"Percentage Distribution: {x_var} vs {y_var}",
                        yaxis_title="Percentage"
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            # Correlation matrix for all numeric variables
            st.write("**Correlation Matrix:**")
            numeric_df = df.select_dtypes(include=[np.number])
            if len(numeric_df.columns) > 1:
                corr_matrix = numeric_df.corr()
                
                fig = px.imshow(corr_matrix, 
                              title="Correlation Matrix",
                              color_continuous_scale='RdBu_r',
                              aspect="auto")
                st.plotly_chart(fig, use_container_width=True)
                
                # Find highest correlations
                st.write("**Highest Correlations:**")
                corr_pairs = []
                for i in range(len(corr_matrix.columns)):
                    for j in range(i+1, len(corr_matrix.columns)):
                        corr_pairs.append({
                            'Variable 1': corr_matrix.columns[i],
                            'Variable 2': corr_matrix.columns[j],
                            'Correlation': corr_matrix.iloc[i, j]
                        })
                
                corr_df = pd.DataFrame(corr_pairs)
                corr_df = corr_df.reindex(corr_df['Correlation'].abs().sort_values(ascending=False).index)
                st.dataframe(corr_df.head(10), use_container_width=True)
        
        with tab4:
            st.subheader("📈 Multivariate Analysis")
            
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            
            if len(numeric_cols) >= 3:
                # Pair plot
                st.write("**Pair Plot:**")
                
                # Select subset of columns for pair plot (max 6 for performance)
                max_cols = 6
                if len(numeric_cols) > max_cols:
                    selected_cols = st.multiselect(
                        f"Select up to {max_cols} numeric columns for pair plot:",
                        numeric_cols,
                        default=numeric_cols[:max_cols]
                    )
                else:
                    selected_cols = numeric_cols
                
                if selected_cols and len(selected_cols) >= 2:
                    # Create pair plot using plotly
                    fig = ff.create_scatterplotmatrix(
                        df[selected_cols], 
                        diag='histogram',
                        height=800,
                        width=800
                    )
                    fig.update_layout(title="Pair Plot Matrix")
                    st.plotly_chart(fig, use_container_width=True)
                
                # 3D Scatter plot
                if len(numeric_cols) >= 3:
                    st.write("**3D Scatter Plot:**")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        x_3d = st.selectbox("X axis:", numeric_cols, key="3d_x")
                    with col2:
                        y_3d = st.selectbox("Y axis:", numeric_cols, key="3d_y", index=1)
                    with col3:
                        z_3d = st.selectbox("Z axis:", numeric_cols, key="3d_z", index=2)
                    with col4:
                        color_by = st.selectbox("Color by:", [None] + df.columns.tolist(), key="3d_color")
                    
                    fig = px.scatter_3d(df, x=x_3d, y=y_3d, z=z_3d, color=color_by,
                                      title=f"3D Scatter: {x_3d} vs {y_3d} vs {z_3d}")
                    st.plotly_chart(fig, use_container_width=True)
                
                # Parallel coordinates
                st.write("**Parallel Coordinates:**")
                
                # Select categorical column for coloring
                cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
                if cat_cols:
                    color_col = st.selectbox("Color by category:", [None] + cat_cols)
                    
                    if color_col:
                        fig = px.parallel_coordinates(
                            df, 
                            dimensions=selected_cols[:6],  # Limit for performance
                            color=color_col,
                            title="Parallel Coordinates Plot"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                
                # Radar chart for comparing categories
                if cat_cols:
                    st.write("**Radar Chart:**")
                    
                    radar_cat = st.selectbox("Select categorical variable for radar chart:", cat_cols)
                    radar_vars = st.multiselect("Select variables for radar chart:", numeric_cols, default=numeric_cols[:5])
                    
                    if radar_vars and len(radar_vars) >= 3:
                        # Calculate means by category
                        radar_data = df.groupby(radar_cat)[radar_vars].mean()
                        
                        fig = go.Figure()
                        
                        for category in radar_data.index:
                            fig.add_trace(go.Scatterpolar(
                                r=radar_data.loc[category].values,
                                theta=radar_vars,
                                fill='toself',
                                name=str(category)
                            ))
                        
                        fig.update_layout(
                            polar=dict(
                                radialaxis=dict(visible=True)
                            ),
                            title=f"Radar Chart by {radar_cat}"
                        )
                        st.plotly_chart(fig, use_container_width=True)
            
            else:
                st.warning("Need at least 3 numeric columns for multivariate analysis.")
        
        with tab5:
            st.subheader("📋 Statistical Summary Report")
            
            # Comprehensive statistical summary
            st.write("**Descriptive Statistics:**")
            desc_stats = df.describe(include='all').T
            desc_stats = desc_stats.round(4)
            st.dataframe(desc_stats, use_container_width=True)
            
            # Data quality assessment
            st.write("**Data Quality Assessment:**")
            
            quality_metrics = []
            for col in df.columns:
                missing_pct = (df[col].isnull().sum() / len(df)) * 100
                unique_pct = (df[col].nunique() / len(df)) * 100
                
                if df[col].dtype in ['int64', 'float64']:
                    zero_pct = ((df[col] == 0).sum() / len(df)) * 100
                    outlier_count = len(df[(np.abs(stats.zscore(df[col].dropna())) > 3)])
                    outlier_pct = (outlier_count / len(df)) * 100
                else:
                    zero_pct = 0
                    outlier_pct = 0
                
                quality_metrics.append({
                    'Column': col,
                    'Data Type': str(df[col].dtype),
                    'Missing %': missing_pct,
                    'Unique %': unique_pct,
                    'Zero %': zero_pct,
                    'Outlier %': outlier_pct
                })
            
            quality_df = pd.DataFrame(quality_metrics)
            st.dataframe(quality_df, use_container_width=True)
            
            # Recommendations
            st.write("**Data Quality Recommendations:**")
            
            recommendations = []
            for _, row in quality_df.iterrows():
                col = row['Column']
                if row['Missing %'] > 20:
                    recommendations.append(f"⚠️ {col}: High missing values ({row['Missing %']:.1f}%) - consider imputation or removal")
                if row['Unique %'] > 95 and df[col].dtype in ['object']:
                    recommendations.append(f"🔍 {col}: Very high uniqueness ({row['Unique %']:.1f}%) - might be an identifier")
                if row['Outlier %'] > 10:
                    recommendations.append(f"📊 {col}: High outlier percentage ({row['Outlier %']:.1f}%) - investigate outliers")
                if row['Zero %'] > 50 and df[col].dtype in ['int64', 'float64']:
                    recommendations.append(f"🔢 {col}: High percentage of zeros ({row['Zero %']:.1f}%) - verify data validity")
            
            if recommendations:
                for rec in recommendations:
                    st.write(rec)
            else:
                st.success("✅ Data quality looks good overall!")
            
            # Export summary report
            st.write("**Export Options:**")
            
            # Create comprehensive report
            report_sections = {
                'Dataset Overview': {
                    'Shape': df.shape,
                    'Memory Usage (KB)': df.memory_usage(deep=True).sum() / 1024,
                    'Missing Values': df.isnull().sum().sum(),
                    'Duplicate Rows': df.duplicated().sum()
                },
                'Column Information': info_df.to_dict('records'),
                'Descriptive Statistics': desc_stats.to_dict(),
                'Data Quality Metrics': quality_df.to_dict('records'),
                'Recommendations': recommendations
            }
            
            # Download buttons
            col1, col2 = st.columns(2)
            
            with col1:
                # Download processed data
                csv_data = df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Data (CSV)",
                    data=csv_data,
                    file_name="eda_data.csv",
                    mime="text/csv"
                )
            
            with col2:
                # Download summary report
                import json
                report_json = json.dumps(report_sections, indent=2, default=str)
                st.download_button(
                    label="📋 Download EDA Report (JSON)",
                    data=report_json,
                    file_name="eda_report.json",
                    mime="application/json"
                )

if __name__ == "__main__":
    main()