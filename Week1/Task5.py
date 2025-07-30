import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.datasets import load_iris, load_wine, load_breast_cancer, load_digits
from sklearn.manifold import TSNE
import warnings
warnings.filterwarnings('ignore')

def perform_pca_analysis(X, n_components=None, scale_data=True):
    """Perform PCA analysis with optional scaling"""
    X_original = X.copy()
    if scale_data:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_for_pca = X_scaled
    else:
        scaler = None
        X_for_pca = X.values
    if n_components is None:
        n_components = min(X.shape[0], X.shape[1])
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_for_pca)
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
    results = {
        'pca': pca,
        'scaler': scaler,
        'X_original': X_original,
        'X_scaled': X_scaled if scale_data else X.values,
        'X_pca': X_pca,
        'explained_variance_ratio': pca.explained_variance_ratio_,
        'cumulative_variance': cumulative_variance,
        'components': pca.components_,
        'feature_names': X.columns.tolist()
    }
    return results

def plot_explained_variance(results):
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Individual Explained Variance', 'Cumulative Explained Variance'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}]]
    )
    n_components = len(results['explained_variance_ratio'])
    component_labels = [f'PC{i+1}' for i in range(n_components)]
    fig.add_trace(
        go.Bar(x=component_labels, y=results['explained_variance_ratio'],
               name="Explained Variance", showlegend=False),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=component_labels, y=results['cumulative_variance'],
                   mode='lines+markers', name="Cumulative Variance", showlegend=False),
        row=1, col=2
    )
    fig.add_hline(y=0.95, line_dash="dash", line_color="red",
                  annotation_text="95% Variance", row=1, col=2)
    fig.update_xaxes(title_text="Principal Components", row=1, col=1)
    fig.update_xaxes(title_text="Principal Components", row=1, col=2)
    fig.update_yaxes(title_text="Explained Variance Ratio", row=1, col=1)
    fig.update_yaxes(title_text="Cumulative Variance Ratio", row=1, col=2)
    fig.update_layout(height=400, title_text="PCA Explained Variance Analysis")
    return fig

def plot_pca_2d(results, y=None, color_by=None):
    if results['X_pca'].shape[1] < 2:
        return None
    df_pca = pd.DataFrame({
        'PC1': results['X_pca'][:, 0],
        'PC2': results['X_pca'][:, 1]
    })
    if y is not None:
        df_pca['target'] = y
        color = 'target'
    elif color_by is not None:
        df_pca['color'] = color_by
        color = 'color'
    else:
        color = None
    fig = px.scatter(df_pca, x='PC1', y='PC2', color=color,
                     title='PCA 2D Visualization',
                     labels={'PC1': f'PC1 ({results["explained_variance_ratio"][0]:.1%})',
                            'PC2': f'PC2 ({results["explained_variance_ratio"][1]:.1%})'})
    return fig

def plot_pca_3d(results, y=None, color_by=None):
    if results['X_pca'].shape[1] < 3:
        return None
    df_pca = pd.DataFrame({
        'PC1': results['X_pca'][:, 0],
        'PC2': results['X_pca'][:, 1],
        'PC3': results['X_pca'][:, 2]
    })
    if y is not None:
        df_pca['target'] = y
        color = 'target'
    elif color_by is not None:
        df_pca['color'] = color_by
        color = 'color'
    else:
        color = None
    fig = px.scatter_3d(df_pca, x='PC1', y='PC2', z='PC3', color=color,
                        title='PCA 3D Visualization',
                        labels={'PC1': f'PC1 ({results["explained_variance_ratio"][0]:.1%})',
                               'PC2': f'PC2 ({results["explained_variance_ratio"][1]:.1%})',
                               'PC3': f'PC3 ({results["explained_variance_ratio"][2]:.1%})'})
    return fig

def plot_component_heatmap(results, n_components=10):
    n_components = min(n_components, results['components'].shape[0])
    components_df = pd.DataFrame(
        results['components'][:n_components],
        columns=results['feature_names'],
        index=[f'PC{i+1}' for i in range(n_components)]
    )
    fig = px.imshow(components_df, 
                    title='PCA Component Loadings',
                    labels=dict(x="Features", y="Principal Components", color="Loading"),
                    color_continuous_scale='RdBu_r')
    return fig

def analyze_feature_importance(results, n_components=5):
    n_components = min(n_components, results['components'].shape[0])
    importance_data = []
    for i in range(n_components):
        loadings = np.abs(results['components'][i])
        for j, feature in enumerate(results['feature_names']):
            importance_data.append({
                'Component': f'PC{i+1}',
                'Feature': feature,
                'Loading': results['components'][i][j],
                'Abs_Loading': loadings[j],
                'Variance_Explained': results['explained_variance_ratio'][i]
            })
    importance_df = pd.DataFrame(importance_data)
    return importance_df

def main():
    st.set_page_config(page_title="PCA Analysis Toolkit", layout="wide")
    st.title("🔬 Principal Component Analysis (PCA) Toolkit")
    st.write("Comprehensive PCA analysis with 2D/3D visualization and dimensionality reduction")
    st.sidebar.header("📊 Data Configuration")
    data_option = st.sidebar.selectbox(
        "Choose data source",
        ["Upload CSV", "Iris Dataset", "Wine Dataset", "Breast Cancer Dataset", "Digits Dataset"]
    )
    df = None
    target = None
    if data_option == "Upload CSV":
        uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            has_target = st.sidebar.checkbox("Dataset has target/label column")
            if has_target:
                target_col = st.sidebar.selectbox("Select target column:", df.columns)
                target = df[target_col]
                df = df.drop(columns=[target_col])
        else:
            st.info("Please upload a CSV file to get started!")
            return
    elif data_option == "Iris Dataset":
        iris = load_iris()
        df = pd.DataFrame(iris.data, columns=iris.feature_names)
        target = iris.target
        target = pd.Series([iris.target_names[i] for i in target])
    elif data_option == "Wine Dataset":
        wine = load_wine()
        df = pd.DataFrame(wine.data, columns=wine.feature_names)
        target = wine.target
        target = pd.Series([f'Class_{i}' for i in target])
    elif data_option == "Breast Cancer Dataset":
        cancer = load_breast_cancer()
        df = pd.DataFrame(cancer.data, columns=cancer.feature_names)
        target = cancer.target
        target = pd.Series([cancer.target_names[i] for i in target])
    elif data_option == "Digits Dataset":
        digits = load_digits()
        df = pd.DataFrame(digits.data, columns=[f'pixel_{i}' for i in range(digits.data.shape[1])])
        target = digits.target
        target = pd.Series([f'Digit_{i}' for i in target])
    if df is not None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) == 0:
            st.error("No numeric columns found in the dataset!")
            return
        df_numeric = df[numeric_cols]
        st.sidebar.header("🔧 PCA Configuration")
        scale_data = st.sidebar.checkbox("Standardize data before PCA", value=True)
        max_components = min(df_numeric.shape[0], df_numeric.shape[1])
        n_components = st.sidebar.slider(
            "Number of components:", 
            2, max_components, 
            min(10, max_components)
        )
        variance_threshold = st.sidebar.slider(
            "Minimum cumulative variance to retain:", 
            0.8, 0.99, 0.95, 0.01
        )
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📋 Dataset Overview",
            "📊 PCA Results", 
            "📈 2D Visualization",
            "🎯 3D Visualization",
            "🔍 Component Analysis"
        ])
        with tab1:
            st.subheader("📊 Dataset Overview")
            col1, col2, col3 = st.columns([1, 1, 1])
            with col1:
                st.metric("Number of Samples", df_numeric.shape[0])
                st.metric("Number of Features", df_numeric.shape[1])
                st.metric("Missing Values", df_numeric.isnull().sum().sum())
            with col2:
                if target is not None:
                    st.metric("Number of Classes", target.nunique())
                    st.write("**Class Distribution:**")
                    class_counts = target.value_counts()
                    st.dataframe(class_counts, use_container_width=True)
                else:
                    st.write("**No target variable available**")
            with col3:
                st.metric("Data Sparsity", f"{(df_numeric == 0).sum().sum() / df_numeric.size:.1%}")
                st.metric("Feature Correlation (avg)", f"{df_numeric.corr().abs().mean().mean():.3f}")
            st.write("**Dataset Preview:**")
            st.dataframe(df_numeric.head(), use_container_width=True)
            st.write("**Feature Statistics:**")
            st.dataframe(df_numeric.describe(), use_container_width=True)
            if len(numeric_cols) <= 20:
                st.write("**Feature Correlation Matrix:**")
                corr_matrix = df_numeric.corr()
                fig = px.imshow(corr_matrix, 
                              title="Feature Correlation Matrix",
                              color_continuous_scale='RdBu_r')
                fig.update_layout(height=600)
                st.plotly_chart(fig, use_container_width=True)
        with tab2:
            st.subheader("📊 PCA Results")
            with st.spinner("Performing PCA analysis..."):
                pca_results = perform_pca_analysis(df_numeric, n_components, scale_data)
            col1, col2 = st.columns([1, 1])
            with col1:
                st.write("**PCA Summary:**")
                components_for_threshold = np.argmax(pca_results['cumulative_variance'] >= variance_threshold) + 1
                summary_data = {
                    'Metric': [
                        'Total Components',
                        f'Components for {variance_threshold:.0%} variance',
                        'First Component Variance',
                        'First Two Components Variance',
                        'Data Scaling Applied'
                    ],
                    'Value': [
                        len(pca_results['explained_variance_ratio']),
                        components_for_threshold,
                        f"{pca_results['explained_variance_ratio'][0]:.1%}",
                        f"{pca_results['cumulative_variance'][1]:.1%}" if len(pca_results['cumulative_variance']) > 1 else "N/A",
                        "Yes" if scale_data else "No"
                    ]
                }
                summary_df = pd.DataFrame(summary_data)
                st.dataframe(summary_df, use_container_width=True)
                st.write("**Explained Variance by Component:**")
                variance_df = pd.DataFrame({
                    'Component': [f'PC{i+1}' for i in range(len(pca_results['explained_variance_ratio']))],
                    'Individual Variance': pca_results['explained_variance_ratio'],
                    'Cumulative Variance': pca_results['cumulative_variance']
                })
                st.dataframe(variance_df, use_container_width=True)
            with col2:
                variance_fig = plot_explained_variance(pca_results)
                st.plotly_chart(variance_fig, use_container_width=True)
                st.write("**Scree Plot:**")
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=list(range(1, len(pca_results['explained_variance_ratio']) + 1)),
                    y=pca_results['explained_variance_ratio'],
                    mode='lines+markers',
                    name='Explained Variance'
                ))
                fig.update_layout(
                    title="Scree Plot",
                    xaxis_title="Principal Component",
                    yaxis_title="Explained Variance Ratio"
                )
                st.plotly_chart(fig, use_container_width=True)
            st.write("**Dimensionality Reduction Recommendations:**")
            recommendations = []
            eigenvalues = pca_results['explained_variance_ratio'] * (len(df_numeric.columns) - 1)
            kaiser_components = np.sum(eigenvalues > 1)
            recommendations.append(f"🔹 Kaiser criterion: Retain {kaiser_components} components (eigenvalue > 1)")
            recommendations.append(f"🔹 {variance_threshold:.0%} variance criterion: Retain {components_for_threshold} components")
            variance_diff = np.diff(pca_results['explained_variance_ratio'])
            elbow_point = np.argmax(variance_diff) + 2
            recommendations.append(f"🔹 Elbow method suggests: ~{elbow_point} components")
            for rec in recommendations:
                st.write(rec)
        with tab3:
            st.subheader("📈 2D PCA Visualization")
            if pca_results['X_pca'].shape[1] >= 2:
                pca_2d_fig = plot_pca_2d(pca_results, target)
                st.plotly_chart(pca_2d_fig, use_container_width=True)
                st.write("**Custom 2D Component Selection:**")
                col1, col2 = st.columns(2)
                with col1:
                    pc_x = st.selectbox("X-axis component:", 
                                      [f'PC{i+1}' for i in range(pca_results['X_pca'].shape[1])],
                                      index=0)
                with col2:
                    pc_y = st.selectbox("Y-axis component:", 
                                      [f'PC{i+1}' for i in range(pca_results['X_pca'].shape[1])],
                                      index=1)
                pc_x_idx = int(pc_x[2:]) - 1
                pc_y_idx = int(pc_y[2:]) - 1
                if pc_x_idx != pc_y_idx:
                    df_custom = pd.DataFrame({
                        'X': pca_results['X_pca'][:, pc_x_idx],
                        'Y': pca_results['X_pca'][:, pc_y_idx]
                    })
                    if target is not None:
                        df_custom['target'] = target
                        color = 'target'
                    else:
                        color = None
                    fig = px.scatter(df_custom, x='X', y='Y', color=color,
                                   title=f'{pc_x} vs {pc_y}',
                                   labels={'X': f'{pc_x} ({pca_results["explained_variance_ratio"][pc_x_idx]:.1%})',
                                          'Y': f'{pc_y} ({pca_results["explained_variance_ratio"][pc_y_idx]:.1%})'})
                    st.plotly_chart(fig, use_container_width=True)
                if len(df_numeric.columns) <= 20:
                    st.write("**PCA Biplot (Features + Samples):**")
                    fig = go.Figure()
                    if target is not None:
                        unique_targets = target.unique()
                        colors = px.colors.qualitative.Set1[:len(unique_targets)]
                        for i, target_class in enumerate(unique_targets):
                            mask = target == target_class
                            fig.add_trace(go.Scatter(
                                x=pca_results['X_pca'][mask, 0],
                                y=pca_results['X_pca'][mask, 1],
                                mode='markers',
                                name=str(target_class),
                                marker=dict(color=colors[i % len(colors)])
                            ))
                    else:
                        fig.add_trace(go.Scatter(
                            x=pca_results['X_pca'][:, 0],
                            y=pca_results['X_pca'][:, 1],
                            mode='markers',
                            name='Samples',
                            marker=dict(opacity=0.6)
                        ))
                    feature_scale = 3
                    for i, feature in enumerate(pca_results['feature_names']):
                        fig.add_trace(go.Scatter(
                            x=[0, pca_results['components'][0, i] * feature_scale],
                            y=[0, pca_results['components'][1, i] * feature_scale],
                            mode='lines+text',
                            name=feature,
                            text=['', feature],
                            textposition='top center',
                            line=dict(color='red', width=2),
                            showlegend=False
                        ))
                    fig.update_layout(
                        title="PCA Biplot",
                        xaxis_title=f'PC1 ({pca_results["explained_variance_ratio"][0]:.1%})',
                        yaxis_title=f'PC2 ({pca_results["explained_variance_ratio"][1]:.1%})'
                    )
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Need at least 2 components for 2D visualization.")
        with tab4:
            st.subheader("🎯 3D PCA Visualization")
            if pca_results['X_pca'].shape[1] >= 3:
                pca_3d_fig = plot_pca_3d(pca_results, target)
                st.plotly_chart(pca_3d_fig, use_container_width=True)
                st.write("**Custom 3D Component Selection:**")
                col1, col2, col3 = st.columns(3)
                with col1:
                    pc_x_3d = st.selectbox("X-axis:", 
                                         [f'PC{i+1}' for i in range(pca_results['X_pca'].shape[1])],
                                         index=0, key="3d_x")
                with col2:
                    pc_y_3d = st.selectbox("Y-axis:", 
                                         [f'PC{i+1}' for i in range(pca_results['X_pca'].shape[1])],
                                         index=1, key="3d_y")
                with col3:
                    pc_z_3d = st.selectbox("Z-axis:", 
                                         [f'PC{i+1}' for i in range(pca_results['X_pca'].shape[1])],
                                         index=2, key="3d_z")
                pc_x_3d_idx = int(pc_x_3d[2:]) - 1
                pc_y_3d_idx = int(pc_y_3d[2:]) - 1
                pc_z_3d_idx = int(pc_z_3d[2:]) - 1
                if len(set([pc_x_3d_idx, pc_y_3d_idx, pc_z_3d_idx])) == 3:
                    df_custom_3d = pd.DataFrame({
                        'X': pca_results['X_pca'][:, pc_x_3d_idx],
                        'Y': pca_results['X_pca'][:, pc_y_3d_idx],
                        'Z': pca_results['X_pca'][:, pc_z_3d_idx]
                    })
                    if target is not None:
                        df_custom_3d['target'] = target
                        color = 'target'
                    else:
                        color = None
                    fig = px.scatter_3d(df_custom_3d, x='X', y='Y', z='Z', color=color,
                                       title=f'{pc_x_3d} vs {pc_y_3d} vs {pc_z_3d}',
                                       labels={'X': f'{pc_x_3d} ({pca_results["explained_variance_ratio"][pc_x_3d_idx]:.1%})',
                                              'Y': f'{pc_y_3d} ({pca_results["explained_variance_ratio"][pc_y_3d_idx]:.1%})',
                                              'Z': f'{pc_z_3d} ({pca_results["explained_variance_ratio"][pc_z_3d_idx]:.1%})'})
                    st.plotly_chart(fig, use_container_width=True)
                st.write("**Comparison with t-SNE:**")
                if st.button("Generate t-SNE visualization"):
                    with st.spinner("Computing t-SNE (this may take a while)..."):
                        n_components_tsne = min(50, pca_results['X_pca'].shape[1])
                        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(df_numeric)//4))
                        X_tsne = tsne.fit_transform(pca_results['X_pca'][:, :n_components_tsne])
                        col_a, col_b = st.columns(2)
                        with col_a:
                            df_pca_2d = pd.DataFrame({
                                'PC1': pca_results['X_pca'][:, 0],
                                'PC2': pca_results['X_pca'][:, 1],
                                'target': target if target is not None else 'No target'
                            })
                            fig_pca = px.scatter(df_pca_2d, x='PC1', y='PC2', 
                                               color='target' if target is not None else None,
                                               title="PCA 2D")
                            st.plotly_chart(fig_pca, use_container_width=True)
                        with col_b:
                            df_tsne_2d = pd.DataFrame({
                                'tSNE1': X_tsne[:, 0],
                                'tSNE2': X_tsne[:, 1],
                                'target': target if target is not None else 'No target'
                            })
                            fig_tsne = px.scatter(df_tsne_2d, x='tSNE1', y='tSNE2',
                                                color='target' if target is not None else None,
                                                title="t-SNE 2D")
                            st.plotly_chart(fig_tsne, use_container_width=True)
            else:
                st.warning("Need at least 3 components for 3D visualization.")
        with tab5:
            st.subheader("🔍 Component Analysis")
            st.write("**Component Loadings Heatmap:**")
            n_components_heatmap = st.slider("Number of components to show:", 
                                            1, min(15, pca_results['components'].shape[0]), 
                                            min(10, pca_results['components'].shape[0]))
            heatmap_fig = plot_component_heatmap(pca_results, n_components_heatmap)
            st.plotly_chart(heatmap_fig, use_container_width=True)
            st.write("**Feature Importance in Principal Components:**")
            importance_df = analyze_feature_importance(pca_results, n_components_heatmap)
            for i in range(min(5, n_components_heatmap)):
                component = f'PC{i+1}'
                component_importance = importance_df[importance_df['Component'] == component].nlargest(5, 'Abs_Loading')
                st.write(f"**{component} - Top Contributing Features:**")
                st.write(f"Explains {pca_results['explained_variance_ratio'][i]:.1%} of variance")
                fig = px.bar(component_importance, x='Abs_Loading', y='Feature', orientation='h',
                           title=f'Top Features in {component}',
                           labels={'Abs_Loading': 'Absolute Loading', 'Feature': 'Features'})
                st.plotly_chart(fig, use_container_width=True)
                display_df = component_importance[['Feature', 'Loading', 'Abs_Loading']].round(4)
                st.dataframe(display_df, use_container_width=True)
            st.write("**Overall Feature Importance (across all components):**")
            overall_importance = importance_df.groupby('Feature').agg({
                'Abs_Loading': 'mean',
                'Loading': lambda x: np.sqrt(np.sum(x**2))
            }).sort_values('Loading', ascending=False)
            overall_importance.columns = ['Mean_Abs_Loading', 'Overall_Importance']
            overall_importance = overall_importance.reset_index()
            fig = px.bar(overall_importance.head(15), x='Overall_Importance', y='Feature', orientation='h',
                        title='Overall Feature Importance Across All Components')
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(overall_importance, use_container_width=True)
            st.write("**Export PCA Results:**")
            col1, col2, col3 = st.columns(3)
            with col1:
                n_components_export = st.selectbox("Components to export:", 
                                                 [2, 3, 5, 10, components_for_threshold, 'All'],
                                                 index=3)
                if n_components_export == 'All':
                    export_components = pca_results['X_pca']
                else:
                    export_components = pca_results['X_pca'][:, :n_components_export]
                export_df = pd.DataFrame(export_components, 
                                       columns=[f'PC{i+1}' for i in range(export_components.shape[1])])
                if target is not None:
                    export_df['target'] = target.values
                csv_data = export_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download PCA Data",
                    data=csv_data,
                    file_name="pca_transformed_data.csv",
                    mime="text/csv"
                )
            with col2:
                loadings_df = pd.DataFrame(
                    pca_results['components'][:n_components_heatmap].T,
                    columns=[f'PC{i+1}' for i in range(n_components_heatmap)],
                    index=pca_results['feature_names']
                )
                loadings_csv = loadings_df.to_csv()
                st.download_button(
                    label="📊 Download Component Loadings",
                    data=loadings_csv,
                    file_name="pca_component_loadings.csv",
                    mime="text/csv"
                )
            with col3:
                variance_export_df = pd.DataFrame({
                    'Component': [f'PC{i+1}' for i in range(len(pca_results['explained_variance_ratio']))],
                    'Explained_Variance_Ratio': pca_results['explained_variance_ratio'],
                    'Cumulative_Variance_Ratio': pca_results['cumulative_variance']
                })
                variance_csv = variance_export_df.to_csv(index=False)
                st.download_button(
                    label="📈 Download Variance Data",
                    data=variance_csv,
                    file_name="pca_variance_explained.csv",
                    mime="text/csv"
                )

if __name__ == "__main__":
    main()