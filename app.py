import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import IncrementalPCA, PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import umap
import warnings
warnings.filterwarnings('ignore')

#import os
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
#os.environ['OPENBLAS_NUM_THREADS'] = '1'

# Page configuration
st.set_page_config(
    page_title="Drone Phenotyping Data Analysis",
    page_icon="🚁",
    layout="wide"
)

# Title and description
st.title("🚁 High-Throughput Drone Phenotyping Data Analysis")
st.markdown("""
This application analyzes drone phenotyping data with RGB, multispectral, and vegetation indices across multiple timepoints.
Upload your Excel file to explore the latent space using various dimensionality reduction techniques.
""")

# Sidebar for file upload and settings
with st.sidebar:
    st.header("📁 Data Upload")
    uploaded_file = st.file_uploader(
        "Choose an Excel file",
        type=['xlsx', 'xls', 'csv'],
        help="Upload a long-format Excel file with phenotyping data"
    )
    
    if uploaded_file is not None:
        st.success("File uploaded successfully!")
        
        # Load data
        @st.cache_data
        def load_data(file):
            if file.name.endswith('.csv'):
                df = pd.read_csv(file)
            else:
                df = pd.read_excel(file)
            return df
        
        df = load_data(uploaded_file)
        
        st.subheader("📊 Data Overview")
        st.write(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")
        
        # Column selection
        st.subheader("🎯 Feature Selection")
        
        # Identify columns by type
        all_columns = df.columns.tolist()
        
        # Let user identify column types
        timepoint_col = st.selectbox(
            "Select Timepoint Column",
            options=['None'] + all_columns,
            help="Column containing timepoint information"
        )
        
        id_columns = st.multiselect(
            "Select ID/Metadata Columns",
            options=all_columns,
            default=[col for col in all_columns if 'id' in col.lower() or 'plot' in col.lower()],
            help="Columns containing plot IDs, genotypes, or other metadata"
        )
        
        # Identify feature columns
        feature_columns = [col for col in all_columns if col not in id_columns and col != timepoint_col]
        
        # Categorize features
        rgb_features = st.multiselect(
            "Select RGB Features",
            options=feature_columns,
            default=[col for col in feature_columns if any(x in col.lower() for x in ['rgb', 'red', 'green', 'blue', 'color'])],
            help="RGB-related features"
        )
        
        multispec_features = st.multiselect(
            "Select Multispectral Features",
            options=[col for col in feature_columns if col not in rgb_features],
            default=[col for col in feature_columns if any(x in col.lower() for x in ['nir', 'rededge', 'multispec', 'band']) and col not in rgb_features],
            help="Multispectral band features"
        )
        
        vi_features = st.multiselect(
            "Select Vegetation Indices",
            options=[col for col in feature_columns if col not in rgb_features and col not in multispec_features],
            default=[col for col in feature_columns if any(x in col.lower() for x in ['ndvi', 'evi', 'savi', 'gndvi', 'rdvi', 'osavi', 'mcari', 'pri']) and col not in rgb_features and col not in multispec_features],
            help="Vegetation index features"
        )
        
        # Combine selected features
        selected_features = rgb_features + multispec_features + vi_features
        
        st.write(f"Total features selected: {len(selected_features)}")

# Main content area
if uploaded_file is not None:
    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Data Preview", "🔬 Model Selection", "🌐 Latent Space Visualization", "📊 Feature Analysis"])
    
    with tab1:
        st.header("Data Preview")
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("First 10 rows")
            st.dataframe(df.head(10))
        
        with col2:
            st.subheader("Data Statistics")
            st.dataframe(df[selected_features].describe() if selected_features else df.describe())
        
        # Missing data visualization
        st.subheader("Missing Data Analysis")
        missing_data = df[selected_features].isnull().sum() if selected_features else df.isnull().sum()
        missing_df = pd.DataFrame({
            'Column': missing_data.index,
            'Missing Count': missing_data.values,
            'Missing %': (missing_data.values / len(df)) * 100
        })
        missing_df = missing_df[missing_df['Missing Count'] > 0].sort_values('Missing %', ascending=False)
        
        if len(missing_df) > 0:
            fig = px.bar(missing_df, x='Column', y='Missing %', 
                        title="Missing Data Percentage by Column",
                        color='Missing %', color_continuous_scale='Reds')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No missing data found in selected features!")
    
    # Replace the entire "tab2" section with:
    with tab2:
        st.header("LSTM-VAE Model Selection")
        
        col1, col2 = st.columns(2)
        
        with col1:
            latent_dims = st.selectbox(
                "Latent Dimensions",
                [2, 4, 8, 16],
                help="Select the number of latent dimensions for LSTM-VAE"
            )
            
            feature_combo = st.selectbox(
                "Feature Combination",
                ["RGB", "MS", "VI", "RGBMS", "RGBMSVI"],
                help="Select which feature combinations to use"
            )
        
        with col2:
            year_input = st.text_input(
                "Year",
                placeholder="2024",
                help="Enter the specific year (e.g., 2024)"
            )
            
            season = st.selectbox(
                "Season",
                ["WS", "DS"],
                help="WS: Wet Season, DS: Dry Season"
            )
        
        # Model directory path
        st.subheader("Model Directory")
        model_dir = st.text_input(
            "Model Directory Path",
            placeholder="/path/to/your/models/",
            help="Path to directory containing pre-trained LSTM-VAE models"
        )
        
        # Generate model filename based on selection
        if year_input:
            model_name = f"lstm_vae_{feature_combo}_{year_input}_{season}_dim{latent_dims}.keras"
        else:
            st.warning("Please enter a year to generate model filename")
            model_name = ""
        
        if model_name:
            st.info(f"Selected model: {model_name}")
        
        # Load model button
        if st.button("Load LSTM-VAE Model"):
            if model_dir and year_input:
                import os
                model_path = os.path.join(model_dir, model_name)
                if os.path.exists(model_path):
                    st.success(f"Model found: {model_path}")
                    # Store in session state for use in other tabs
                    st.session_state['model_path'] = model_path
                    st.session_state['latent_dims'] = latent_dims
                    st.session_state['feature_combo'] = feature_combo
                    st.session_state['year'] = year_input
                    st.session_state['season'] = season
                else:
                    st.error(f"Model not found: {model_path}")
            else:
                if not model_dir:
                    st.warning("Please specify the model directory path")
                if not year_input:
                    st.warning("Please enter a year")
        
        # Preprocessing options
        st.subheader("Preprocessing")
        scale_data = st.checkbox("Standardize Features", value=True)
        handle_missing = st.selectbox(
            "Handle Missing Values",
            ["Drop rows", "Fill with mean", "Fill with median", "Fill with zero"]
        )
    
    with tab3:
        st.header("Latent Space Visualization")
        
        if selected_features and 'model_path' in st.session_state:
            # Prepare data
            X = df[selected_features].copy()
            
            # Handle missing values
            if handle_missing == "Drop rows":
                X = X.dropna()
                df_clean = df.loc[X.index]
            elif handle_missing == "Fill with mean":
                X = X.fillna(X.mean())
                df_clean = df.copy()
            elif handle_missing == "Fill with median":
                X = X.fillna(X.median())
                df_clean = df.copy()
            else:  # Fill with zero
                X = X.fillna(0)
                df_clean = df.copy()
            
            # Scale data if requested
            if scale_data:
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
            else:
                X_scaled = X.values
            
            # Load and apply LSTM-VAE model
            with st.spinner(f"Loading LSTM-VAE model and generating latent space..."):
                try:
                    import tensorflow as tf
                    
                    # Load the model
                    model = tf.keras.models.load_model(st.session_state['model_path'])
                    
                    # Get the encoder part (assumes encoder outputs latent representation)
                    # You may need to adjust this based on your model architecture
                    encoder = model.encoder if hasattr(model, 'encoder') else model
                    
                    # Generate latent representation
                    # Reshape data for LSTM if needed (assuming shape: samples, timesteps, features)
                    # Adjust this based on your model's expected input shape
                    X_reshaped = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))
                    X_reduced = encoder.predict(X_reshaped)
                    
                    # Get number of dimensions
                    n_components = X_reduced.shape[1]
                    
                    st.success(f"Latent space generated with {n_components} dimensions")
                    
                except Exception as e:
                    st.error(f"Error loading model or generating latent space: {str(e)}")
                    st.stop()
            
            # Create visualization dataframe
            viz_df = pd.DataFrame(
                X_reduced[:, :min(3, n_components)],
                columns=[f'Latent Dim {i+1}' for i in range(min(3, n_components))],
                index=df_clean.index
            )
            
            # Add metadata for coloring
            color_by_options = ["None"] + id_columns
            if timepoint_col != 'None':
                color_by_options.append(timepoint_col)
            
            color_by = st.selectbox("Color points by:", color_by_options)
            
            if color_by != "None" and color_by in df_clean.columns:
                viz_df[color_by] = df_clean[color_by].values
            
            # Create plot
            if n_components >= 3:
                if color_by != "None":
                    fig = px.scatter_3d(viz_df, x='Latent Dim 1', y='Latent Dim 2', z='Latent Dim 3',
                                    color=color_by, title=f"LSTM-VAE Latent Space (3D)",
                                    hover_data=viz_df.columns)
                else:
                    fig = px.scatter_3d(viz_df, x='Latent Dim 1', y='Latent Dim 2', z='Latent Dim 3',
                                    title=f"LSTM-VAE Latent Space (3D)")
                fig.update_traces(marker=dict(size=5))
            else:
                if color_by != "None":
                    fig = px.scatter(viz_df, x='Latent Dim 1', y='Latent Dim 2',
                                color=color_by, title=f"LSTM-VAE Latent Space (2D)",
                                hover_data=viz_df.columns)
                else:
                    fig = px.scatter(viz_df, x='Latent Dim 1', y='Latent Dim 2',
                                title=f"LSTM-VAE Latent Space (2D)")
            
            fig.update_layout(height=600)
            st.plotly_chart(fig, use_container_width=True)
            
            # Download transformed data
            st.subheader("Download Results")
            result_df = pd.concat([df_clean[id_columns], viz_df], axis=1)
            csv = result_df.to_csv(index=False)
            st.download_button(
                label="Download Latent Space Coordinates as CSV",
                data=csv,
                file_name=f"lstm_vae_latent_space.csv",
                mime="text/csv"
            )
        elif selected_features:
            st.warning("Please load an LSTM-VAE model in the Model Selection tab first.")
        else:
            st.warning("Please select features in the sidebar to visualize the latent space.")
    
    with tab4:
        st.header("Feature Analysis")
        
        if selected_features:
            # Feature correlations
            st.subheader("Feature Correlations")
            corr_matrix = df[selected_features].corr()
            
            fig = go.Figure(data=go.Heatmap(
                z=corr_matrix,
                x=corr_matrix.columns,
                y=corr_matrix.columns,
                colorscale='RdBu',
                zmid=0
            ))
            fig.update_layout(title="Feature Correlation Matrix", height=600)
            st.plotly_chart(fig, use_container_width=True)
            
            # Distribution plots
            st.subheader("Feature Distributions")
            feature_to_plot = st.selectbox("Select feature to visualize:", selected_features)
            
            col1, col2 = st.columns(2)
            with col1:
                fig = px.histogram(df, x=feature_to_plot, title=f"Distribution of {feature_to_plot}")
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = px.box(df, y=feature_to_plot, title=f"Box plot of {feature_to_plot}")
                st.plotly_chart(fig, use_container_width=True)
            
            # If LSTM-VAE model is loaded, show latent dimension statistics
            if 'model_path' in st.session_state:
                st.subheader("Latent Space Statistics")
                st.info(f"Model: {st.session_state.get('feature_combo', 'N/A')} | "
                    f"Year: {st.session_state.get('year', 'N/A')} | "
                    f"Season: {st.session_state.get('season', 'N/A')} | "
                    f"Latent Dimensions: {st.session_state.get('latent_dims', 'N/A')}")
        else:
            st.warning("Please select features in the sidebar for analysis.")

else:
    # Instructions when no file is uploaded
    st.info("""
    ### 📋 Instructions:
    1. **Upload your Excel file** using the sidebar
    2. **Select columns** for timepoints, IDs, and features (RGB, Multispectral, Vegetation Indices)
    3. **Choose a model** for dimensionality reduction (PCA, t-SNE, UMAP, or Incremental PCA)
    4. **Explore the latent space** visualization and feature analysis
    
    ### 📊 Expected Data Format:
    - **Long format**: Each row represents one observation
    - **Columns should include**:
        - Timepoint identifier
        - Plot/Sample IDs or metadata
        - RGB features (e.g., mean_red, mean_green, mean_blue)
        - Multispectral bands (e.g., NIR, RedEdge)
        - Vegetation indices (e.g., NDVI, EVI, SAVI)
    """)
    
    # Sample data structure
    st.subheader("Example Data Structure:")
    sample_data = pd.DataFrame({
        'Plot_ID': ['A1', 'A1', 'A2', 'A2', 'B1', 'B1'],
        'Timepoint': [1, 2, 1, 2, 1, 2],
        'Genotype': ['WT', 'WT', 'Mutant', 'Mutant', 'WT', 'WT'],
        'RGB_Red': [125.3, 132.1, 118.7, 124.5, 129.8, 135.2],
        'RGB_Green': [145.2, 152.3, 138.4, 143.7, 148.9, 154.1],
        'RGB_Blue': [98.5, 102.3, 95.2, 99.1, 101.2, 105.8],
        'NIR': [0.82, 0.85, 0.78, 0.81, 0.83, 0.86],
        'RedEdge': [0.65, 0.68, 0.62, 0.64, 0.66, 0.69],
        'NDVI': [0.72, 0.75, 0.68, 0.71, 0.73, 0.76],
        'EVI': [0.68, 0.71, 0.64, 0.67, 0.69, 0.72]
    })
    st.dataframe(sample_data)