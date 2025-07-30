import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
from sklearn.feature_selection import SelectKBest, f_regression, f_classif, chi2
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.datasets import fetch_california_housing, load_diabetes, load_wine, load_breast_cancer
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

def calculate_correlation_matrix(df, method='pearson'):
    """Calculate correlation matrix"""
    numeric_df = df.select_dtypes(include=[np.number])
    return numeric_df.corr(method=method)

def find_highly_correlated_features(corr_matrix, threshold=0.8):
    """Find highly correlated feature pairs"""
    upper_triangle = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )
    
    high_corr_pairs = []
    for col in upper_triangle.columns:
        for idx in upper_triangle.index:
            value = upper_triangle.loc[idx, col]
            if pd.notna(value) and abs(value) >= threshold:
                high_corr_pairs.append({
                    'Feature 1': idx,
                    'Feature 2': col,
                    'Correlation': value
                })
    
    return pd.DataFrame(high_corr_pairs)

def mutual_information_feature_selection(X, y, task_type='regression', k=10):
    """Perform mutual information feature selection"""
    if task_type == 'regression':
        mi_scores = mutual_info_regression(X, y)
    else:
        mi_scores = mutual_info_classif(X, y)
    
    mi_df = pd.DataFrame({
        'Feature': X.columns,
        'MI_Score': mi_scores
    }).sort_values('MI_Score', ascending=False)
    
    return mi_df

def statistical_feature_selection(X, y, task_type='regression', k=10):
    """Perform statistical feature selection"""
    if task_type == 'regression':
        selector = SelectKBest(score_func=f_regression, k=k)
    else:
        selector = SelectKBest(score_func=f_classif, k=k)
    
    selector.fit(X, y)
    
    scores_df = pd.DataFrame({
        'Feature': X.columns,
        'Score': selector.scores_,
        'P_value': selector.pvalues_
    }).sort_values('Score', ascending=False)
    
    return scores_df, selector

def rfe_feature_selection(X, y, task_type='regression', n_features=10):
    """Perform Recursive Feature Elimination"""
    if task_type == 'regression':
        estimator = LinearRegression()
    else:
        estimator = LogisticRegression(max_iter=1000)
    
    rfe = RFE(estimator=estimator, n_features_to_select=n_features)
    rfe.fit(X, y)
    
    rfe_df = pd.DataFrame({
        'Feature': X.columns,
        'Selected': rfe.support_,
        'Ranking': rfe.ranking_
    }).sort_values('Ranking')
    
    return rfe_df, rfe

def main():
    st.set_page_config(page_title="Feature Selection Toolkit", layout="wide")
    
    st.title("🎯 Feature Selection Toolkit")
    st.write("Comprehensive feature selection using correlation analysis, mutual information, and statistical methods")
    
    # Sidebar for options
    st.sidebar.header("📊 Data Configuration")
    
    data_option = st.sidebar.selectbox(
        "Choose data source",
        ["Upload CSV", "California Housing (Regression)", "Diabetes (Regression)", 
         "Wine Classification", "Breast Cancer (Classification)"]
    )
    
    # Load data
    df = None
    target_col = None
    task_type = 'regression'
    
    if data_option == "Upload CSV":
        uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            target_col = st.sidebar.selectbox("Select target column:", df.columns)
            task_type = st.sidebar.selectbox("Task type:", ["regression", "classification"])
        else:
            st.info("Please upload a CSV file to get started!")
            return
    
    elif data_option == "California Housing (Regression)":
        housing = fetch_california_housing(as_frame=True)
        df = housing.frame
        df['target'] = df['MedHouseVal']
        target_col = 'target'
        task_type = 'regression'
    
    elif data_option == "Diabetes (Regression)":
        diabetes = load_diabetes()
        df = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
        df['target'] = diabetes.target
        target_col = 'target'
        task_type = 'regression'
    
    elif data_option == "Wine Classification":
        wine = load_wine()
        df = pd.DataFrame(wine.data, columns=wine.feature_names)
        df['target'] = wine.target
        target_col = 'target'
        task_type = 'classification'
    
    elif data_option == "Breast Cancer (Classification)":
        cancer = load_breast_cancer()
        df = pd.DataFrame(cancer.data, columns=cancer.feature_names)
        df['target'] = cancer.target
        target_col = 'target'
        task_type = 'classification'
    
    if df is not None:
        # Feature selection parameters
        st.sidebar.header("🔧 Selection Parameters")
        correlation_threshold = st.sidebar.slider("Correlation threshold:", 0.5, 0.95, 0.8, 0.05)
        n_features_to_select = st.sidebar.slider("Number of top features:", 5, min(20, len(df.columns)-1), 10)
        
        # Prepare features and target
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        # Handle categorical variables
        categorical_cols = X.select_dtypes(include=['object']).columns
        if len(categorical_cols) > 0:
            st.sidebar.write(f"Encoding {len(categorical_cols)} categorical columns")
            le = LabelEncoder()
            for col in categorical_cols:
                X[col] = le.fit_transform(X[col].astype(str))
        
        # Create tabs for different analyses
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 Data Overview", 
            "🔗 Correlation Analysis", 
            "📈 Mutual Information", 
            "📉 Statistical Selection",
            "🎯 Feature Ranking"
        ])
        
        with tab1:
            st.subheader("📊 Dataset Overview")
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.write("**Dataset Information:**")
                st.write(f"Shape: {df.shape}")
                st.write(f"Features: {len(X.columns)}")
                st.write(f"Target: {target_col}")
                st.write(f"Task Type: {task_type.title()}")
                
                # Feature types
                feature_types = pd.DataFrame({
                    'Type': ['Numeric', 'Categorical', 'Total'],
                    'Count': [
                        len(X.select_dtypes(include=[np.number]).columns),
                        len(categorical_cols),
                        len(X.columns)
                    ]
                })
                st.dataframe(feature_types, use_container_width=True)
            
            with col2:
                st.write("**Target Distribution:**")
                if task_type == 'classification':
                    target_counts = y.value_counts()
                    fig = px.pie(values=target_counts.values, names=target_counts.index,
                               title="Target Class Distribution")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    fig = px.histogram(x=y, nbins=30, title="Target Distribution")
                    st.plotly_chart(fig, use_container_width=True)
            
            # Dataset preview
            st.write("**Dataset Preview:**")
            st.dataframe(df.head(), use_container_width=True)
            
            # Basic statistics
            st.write("**Feature Statistics:**")
            st.dataframe(X.describe(), use_container_width=True)
        
        with tab2:
            st.subheader("🔗 Correlation Analysis")
            
            # Calculate correlation matrix
            corr_methods = ['pearson', 'spearman', 'kendall']
            selected_method = st.selectbox("Select correlation method:", corr_methods)
            
            corr_matrix = calculate_correlation_matrix(df, method=selected_method)
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.write("**Correlation Heatmap:**")
                
                # Interactive heatmap using plotly
                fig = px.imshow(corr_matrix, 
                              title=f"Feature Correlation Matrix ({selected_method.title()})",
                              color_continuous_scale='RdBu_r',
                              aspect='auto')
                fig.update_layout(height=600)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.write("**Highly Correlated Pairs:**")
                high_corr_pairs = find_highly_correlated_features(corr_matrix, correlation_threshold)
                
                if len(high_corr_pairs) > 0:
                    st.dataframe(high_corr_pairs, use_container_width=True)
                    
                    # Recommendations
                    st.write("**Recommendations:**")
                    for _, row in high_corr_pairs.iterrows():
                        st.write(f"⚠️ Consider removing one of: {row['Feature 1']} or {row['Feature 2']} (r={row['Correlation']:.3f})")
                else:
                    st.success("✅ No highly correlated features found!")
            
            # Target correlation
            st.write("**Feature-Target Correlations:**")
            if target_col in corr_matrix.columns:
                target_corr = corr_matrix[target_col].drop(target_col).abs().sort_values(ascending=False)
                
                fig = px.bar(x=target_corr.values, y=target_corr.index, orientation='h',
                           title="Absolute Correlation with Target",
                           labels={'x': 'Correlation', 'y': 'Features'})
                fig.update_layout(height=max(400, len(target_corr) * 20))
                st.plotly_chart(fig, use_container_width=True)
                
                # Top correlated features
                st.write("**Top Features by Target Correlation:**")
                top_corr_df = pd.DataFrame({
                    'Feature': target_corr.index[:n_features_to_select],
                    'Correlation': target_corr.values[:n_features_to_select]
                })
                st.dataframe(top_corr_df, use_container_width=True)
        
        with tab3:
            st.subheader("📈 Mutual Information Analysis")
            
            # Calculate mutual information
            with st.spinner("Calculating mutual information scores..."):
                mi_scores = mutual_information_feature_selection(X, y, task_type, n_features_to_select)
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.write("**Mutual Information Scores:**")
                
                # Bar plot
                fig = px.bar(mi_scores, x='MI_Score', y='Feature', orientation='h',
                           title="Mutual Information Scores",
                           labels={'MI_Score': 'Mutual Information Score', 'Feature': 'Features'})
                fig.update_layout(height=max(400, len(mi_scores) * 20))
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.write("**Top Features by MI:**")
                top_mi_features = mi_scores.head(n_features_to_select)
                st.dataframe(top_mi_features, use_container_width=True)
                
                # MI score distribution
                st.write("**MI Score Distribution:**")
                fig = px.histogram(mi_scores, x='MI_Score', nbins=20,
                                 title="Distribution of MI Scores")
                st.plotly_chart(fig, use_container_width=True)
            
            # Comparison with correlation
            if target_col in corr_matrix.columns:
                st.write("**Mutual Information vs Correlation:**")
                target_corr = corr_matrix[target_col].drop(target_col).abs()
                
                comparison_df = pd.merge(
                    mi_scores, 
                    target_corr.reset_index().rename(columns={'index': 'Feature', target_col: 'Correlation'}),
                    on='Feature'
                )
                
                fig = px.scatter(comparison_df, x='Correlation', y='MI_Score',
                               hover_data=['Feature'], title="MI Score vs Correlation")
                st.plotly_chart(fig, use_container_width=True)
        
        with tab4:
            st.subheader("📉 Statistical Feature Selection")
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.write("**Statistical Test Results:**")
                with st.spinner("Performing statistical tests..."):
                    stat_scores, stat_selector = statistical_feature_selection(X, y, task_type, n_features_to_select)
                
                # Display results
                st.dataframe(stat_scores.head(15), use_container_width=True)
                
                # Selected features
                selected_features = stat_scores.head(n_features_to_select)['Feature'].tolist()
                st.write("**Selected Features:**")
                for i, feature in enumerate(selected_features, 1):
                    st.write(f"{i}. {feature}")
            
            with col2:
                st.write("**Statistical Scores Visualization:**")
                
                # Bar plot of top features
                top_stat_features = stat_scores.head(n_features_to_select)
                fig = px.bar(top_stat_features, x='Score', y='Feature', orientation='h',
                           title="Statistical Test Scores (Top Features)")
                st.plotly_chart(fig, use_container_width=True)
                
                # P-values plot
                fig = px.scatter(stat_scores, x='Score', y='P_value',
                               hover_data=['Feature'], title="Score vs P-value",
                               log_y=True)
                fig.add_hline(y=0.05, line_dash="dash", line_color="red",
                            annotation_text="p=0.05")
                st.plotly_chart(fig, use_container_width=True)
            
            # RFE Analysis
            st.write("**Recursive Feature Elimination (RFE):**")
            with st.spinner("Performing RFE analysis..."):
                rfe_results, rfe_selector = rfe_feature_selection(X, y, task_type, n_features_to_select)
            
            col_a, col_b = st.columns([1, 1])
            
            with col_a:
                st.write("**RFE Rankings:**")
                st.dataframe(rfe_results, use_container_width=True)
            
            with col_b:
                st.write("**Selected vs Rejected Features:**")
                selected_count = rfe_results['Selected'].sum()
                rejected_count = len(rfe_results) - selected_count
                
                fig = px.pie(values=[selected_count, rejected_count],
                           names=['Selected', 'Rejected'],
                           title="RFE Feature Selection")
                st.plotly_chart(fig, use_container_width=True)
        
        with tab5:
            st.subheader("🎯 Comprehensive Feature Ranking")
            
            # Combine all methods
            methods_results = {}
            
            # Correlation ranking
            if target_col in corr_matrix.columns:
                target_corr = corr_matrix[target_col].drop(target_col).abs().sort_values(ascending=False)
                methods_results['Correlation'] = {feature: rank+1 for rank, feature in enumerate(target_corr.index)}
            
            # MI ranking
            methods_results['Mutual_Info'] = {row['Feature']: rank+1 for rank, (_, row) in enumerate(mi_scores.iterrows())}
            
            # Statistical ranking
            methods_results['Statistical'] = {row['Feature']: rank+1 for rank, (_, row) in enumerate(stat_scores.iterrows())}
            
            # RFE ranking
            methods_results['RFE'] = dict(zip(rfe_results['Feature'], rfe_results['Ranking']))
            
            # Create comprehensive ranking dataframe
            all_features = X.columns.tolist()
            ranking_df = pd.DataFrame(index=all_features)
            
            for method, rankings in methods_results.items():
                ranking_df[method] = [rankings.get(feature, len(all_features)) for feature in all_features]
            
            # Calculate average ranking
            ranking_df['Average_Rank'] = ranking_df.mean(axis=1)
            ranking_df['Overall_Score'] = 1 / ranking_df['Average_Rank']  # Higher score = better rank
            ranking_df = ranking_df.sort_values('Average_Rank')
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.write("**Comprehensive Feature Ranking:**")
                display_df = ranking_df.reset_index().rename(columns={'index': 'Feature'})
                st.dataframe(display_df, use_container_width=True)
                
                # Ranking heatmap
                st.write("**Ranking Heatmap:**")
                fig = px.imshow(ranking_df[['Correlation', 'Mutual_Info', 'Statistical', 'RFE']].T,
                              y=['Correlation', 'Mutual Info', 'Statistical', 'RFE'],
                              x=ranking_df.index,
                              title="Feature Rankings Across Methods",
                              color_continuous_scale='RdYlBu_r')
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.write("**Top Features Summary:**")
                top_features = ranking_df.head(n_features_to_select)
                
                # Display top features
                for i, (feature, row) in enumerate(top_features.iterrows(), 1):
                    st.write(f"**{i}. {feature}**")
                    st.write(f"Average Rank: {row['Average_Rank']:.1f}")
                    st.write("---")
                
                # Method agreement analysis
                st.write("**Method Agreement:**")
                top_by_method = {}
                for method in methods_results.keys():
                    method_top = ranking_df.nsmallest(n_features_to_select, method).index.tolist()
                    top_by_method[method] = set(method_top)
                
                # Find consensus features
                all_top_features = set.union(*top_by_method.values())
                consensus_features = set.intersection(*top_by_method.values())
                
                st.write(f"Consensus features (top in all methods): {len(consensus_features)}")
                for feature in consensus_features:
                    st.write(f"✅ {feature}")
                
                if len(consensus_features) < len(all_top_features):
                    st.write(f"Method-specific features: {len(all_top_features) - len(consensus_features)}")
            
            # Export selected features
            st.write("**Export Selected Features:**")
            export_method = st.selectbox("Choose selection method for export:",
                                       ["Average Ranking", "Correlation", "Mutual Information", 
                                        "Statistical", "RFE", "Consensus Only"])
            
            if export_method == "Average Ranking":
                selected_features_export = ranking_df.head(n_features_to_select).index.tolist()
            elif export_method == "Consensus Only":
                selected_features_export = list(consensus_features)
            else:
                method_map = {
                    "Correlation": "Correlation",
                    "Mutual Information": "Mutual_Info", 
                    "Statistical": "Statistical",
                    "RFE": "RFE"
                }
                selected_features_export = ranking_df.nsmallest(n_features_to_select, 
                                                              method_map[export_method]).index.tolist()
            
            # Create export dataframe
            export_df = X[selected_features_export].copy()
            export_df[target_col] = y
            
            # Download button
            csv_data = export_df.to_csv(index=False)
            st.download_button(
                label=f"📥 Download Selected Features ({len(selected_features_export)} features)",
                data=csv_data,
                file_name=f"selected_features_{export_method.lower().replace(' ', '_')}.csv",
                mime="text/csv"
            )
            
            st.write(f"**Selected Features ({len(selected_features_export)}):**")
            st.write(", ".join(selected_features_export))

if __name__ == "__main__":
    main()