#!/usr/bin/env python3
"""
PLS Analysis for EFNY Brain-Behavior Association with Real Data

This script performs PLS analysis using the actual EFNY dataset files:
- Brain data: EFNY_Schaefer100_FC_matrix.npy
- Behavioral data: EFNY_behavioral_data.csv
- Subject list: sublist.txt

It follows the same pipeline as the demo but uses real data.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.cross_decomposition import PLSCanonical
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from scipy.stats import pearsonr
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class EFNYPLSAnalyzer:
    """PLS analyzer specifically for EFNY dataset."""
    
    def __init__(self, n_components=5):
        """
        Initialize the PLS analyzer.
        
        Args:
            n_components: Number of PLS components to extract
        """
        self.n_components = n_components
        self.pls_model = None
        self.X_resid = None
        self.Y_resid = None
        self.results = None
        
    def load_data(self, brain_data_path, behavioral_data_path, sublist_path):
        """
        Load EFNY data from files.
        
        Args:
            brain_data_path: Path to EFNY_Schaefer100_FC_matrix.npy
            behavioral_data_path: Path to EFNY_behavioral_data.csv
            sublist_path: Path to sublist.txt
            
        Returns:
            X_brain: Brain connectivity data
            Y_behavior: Behavioral data with selected metrics
            covariates: Sex and meanFD data
        """
        logger.info("Loading EFNY data from files")
        
        # Load subject list to ensure proper ordering
        with open(sublist_path, 'r') as f:
            subjects = [line.strip() for line in f if line.strip()]
        logger.info(f"Loaded {len(subjects)} subjects from sublist")
        
        # Load brain data
        X_brain = np.load(brain_data_path)
        logger.info(f"Loaded brain data with shape: {X_brain.shape}")
        
        # Load behavioral data
        behavioral_df = pd.read_csv(behavioral_data_path)
        logger.info(f"Loaded behavioral data with shape: {behavioral_df.shape}")
        
        # Define behavioral metrics of interest
        behavioral_metrics = [
            'SST_SSRT', 'CPT_d_prime', 'CPT_ACC', 'CPT_Reaction_Time',
            'FLANKER_Contrast_ACC', 'FLANKER_Contrast_RT', 'ColorStroop_Contrast_ACC',
            'ColorStroop_Contrast_RT', 'EmotionStroop_Contrast_ACC', 'EmotionStroop_Contrast_RT',
            'Number1Back_ACC', 'Number1Back_Reaction_Time', 'Number2Back_ACC',
            'Number2Back_Reaction_Time', 'Spatial1Back_ACC', 'Spatial1Back_Reaction_Time',
            'Spatial2Back_ACC', 'Spatial2Back_Reaction_Time', 'Emotion1Back_ACC',
            'Emotion1Back_Reaction_Time', 'Emotion2Back_ACC', 'Emotion2Back_Reaction_Time',
            'KT_ACC', 'DCCS_ACC', 'DCCS_Reaction_Time', 'DCCS_Switch_Cost',
            'EmotionSwitch_ACC', 'EmotionSwitch_Reaction_Time', 'DT_ACC',
            'DT_Reaction_Time', 'DT_Switch_Cost'
        ]
        
        # Extract behavioral metrics and covariates (only sex and meanFD, excluding age)
        Y_behavior = behavioral_df[behavioral_metrics].copy()
        covariates = behavioral_df[['sex', 'meanFD']].copy()  # Removed age as requested
        
        # Store behavioral metrics names for later use
        self.behavioral_metrics = behavioral_metrics
        
        # Ensure data is aligned with subject list
        # Assuming the data is already in the correct order based on sublist
        logger.info(f"Extracted {len(behavioral_metrics)} behavioral metrics")
        logger.info(f"Extracted {len(covariates.columns)} covariates")
        
        # Convert brain data to DataFrame
        X_brain_df = pd.DataFrame(X_brain, index=subjects, columns=[f'FC_{i}' for i in range(X_brain.shape[1])])
        
        return X_brain_df, Y_behavior, covariates
    
    def regress_out_covariates(self, data, confounds):
        """
        Regress out covariates from data using linear regression and standardize residuals.
        
        Args:
            data: DataFrame of shape (n_samples, n_features)
            confounds: DataFrame of shape (n_samples, n_confounds)
            
        Returns:
            residuals: DataFrame containing standardized residuals (μ=0, σ=1)
        """
        logger.info(f"Regressing out covariates from data with shape {data.shape}")
        
        residuals = pd.DataFrame(index=data.index, columns=data.columns, dtype=float)
        
        for feature in data.columns:
            # Fit linear regression: feature ~ confounds
            lr = LinearRegression()
            lr.fit(confounds.values, data[feature].values)
            
            # Predict values based on confounds
            predicted = lr.predict(confounds.values)
            
            # Calculate residuals (actual - predicted)
            residual = data[feature].values - predicted
            residuals[feature] = residual
        
        # Standardize residuals to have μ=0, σ=1
        scaler = StandardScaler()
        residuals_scaled = pd.DataFrame(
            scaler.fit_transform(residuals.values),
            index=residuals.index,
            columns=residuals.columns
        )
        
        logger.info("Covariate regression and standardization completed")
        return residuals_scaled
    
    def apply_covariate_regression(self, test_data, test_covariates, train_data, train_covariates):
        """
        Apply covariate regression coefficients from training data to test data.
        This prevents data leakage by using only training-derived coefficients.
        
        Args:
            test_data: Test data (n_test_samples × n_features)
            test_covariates: Test covariates (n_test_samples × n_covariates)
            train_data: Training data used to derive regression coefficients
            train_covariates: Training covariates
            
        Returns:
            test_residuals: Test data with training-derived covariate effects removed
        """
        logger.info("Applying training-derived covariate regression to test data")
        
        test_residuals = pd.DataFrame(index=test_data.index, columns=test_data.columns, dtype=float)
        
        # For each feature, fit regression on training data and apply to test data
        for feature in test_data.columns:
            # Fit regression model on training data only
            lr = LinearRegression()
            lr.fit(train_covariates.values, train_data[feature].values)
            
            # Apply trained model to test covariates
            predicted_test = lr.predict(test_covariates.values)
            
            # Calculate residuals for test data
            residual_test = test_data[feature].values - predicted_test
            test_residuals[feature] = residual_test
        
        return test_residuals
    
    def standardize_with_training_stats(self, train_data, test_data):
        """
        Standardize data using statistics from training data only.
        This prevents data leakage by using training-derived scaling parameters.
        
        Args:
            train_data: Training data (n_train_samples × n_features)
            test_data: Test data (n_test_samples × n_features)
            
        Returns:
            train_scaled: Standardized training data
            test_scaled: Test data scaled using training statistics
        """
        logger.info("Standardizing using training statistics only")
        
        # Calculate statistics from training data only
        train_mean = train_data.mean(axis=0)
        train_std = train_data.std(axis=0)
        
        # Avoid division by zero
        train_std = train_std.replace(0, 1)
        
        # Standardize training data
        train_scaled = (train_data - train_mean) / train_std
        
        # Apply same transformation to test data
        test_scaled = (test_data - train_mean) / train_std
        
        return train_scaled, test_scaled
    
    def create_cv_comparison_table(self, cv_correlations, in_sample_correlations):
        """
        Create comparison table between in-sample and cross-validation correlations.
        
        Args:
            cv_correlations: Mean prediction correlations from CV
            in_sample_correlations: In-sample correlations from full data fit
            
        Returns:
            comparison_df: DataFrame with comparison results
        """
        comparison_df = pd.DataFrame({
            'Component': range(1, self.n_components + 1),
            'In_Sample_Correlation': in_sample_correlations,
            'CV_Prediction_Correlation': cv_correlations,
            'Generalization_Drop': in_sample_correlations - cv_correlations,
            'Generalization_Ratio': cv_correlations / in_sample_correlations
        })
        
        # Format for better display
        comparison_df['In_Sample_Correlation'] = comparison_df['In_Sample_Correlation'].round(4)
        comparison_df['CV_Prediction_Correlation'] = comparison_df['CV_Prediction_Correlation'].round(4)
        comparison_df['Generalization_Drop'] = comparison_df['Generalization_Drop'].round(4)
        comparison_df['Generalization_Ratio'] = comparison_df['Generalization_Ratio'].round(4)
        
        return comparison_df
    
    def plot_cv_comparison(self, cv_results, save_path=None):
        """
        Plot comparison between in-sample and cross-validation correlations.
        
        Args:
            cv_results: Results from nested_cross_validation()
            save_path: Optional path to save the plot
        """
        plt.figure(figsize=(14, 6))
        
        components = range(1, self.n_components + 1)
        in_sample_corrs = cv_results['in_sample_correlations']
        cv_corrs = cv_results['mean_prediction_correlations']
        
        # Plot 1: Correlation comparison
        plt.subplot(1, 2, 1)
        plt.plot(components, in_sample_corrs, 'bo-', linewidth=2, markersize=8, label='In-Sample Correlation', alpha=0.8)
        plt.plot(components, cv_corrs, 'ro-', linewidth=2, markersize=8, label='CV Prediction Correlation', alpha=0.8)
        plt.xlabel('PLS Component', fontsize=12)
        plt.ylabel('Correlation', fontsize=12)
        plt.title('In-Sample vs Cross-Validation Correlations', fontsize=14, fontweight='bold')
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.ylim(0, max(max(in_sample_corrs), max(cv_corrs)) + 0.05)
        
        # Add correlation values as text
        for i, (in_corr, cv_corr) in enumerate(zip(in_sample_corrs, cv_corrs)):
            plt.annotate(f'{in_corr:.3f}', (components[i], in_sample_corrs[i]), 
                        textcoords="offset points", xytext=(0,10), ha='center', fontsize=9)
            plt.annotate(f'{cv_corr:.3f}', (components[i], cv_corrs[i]), 
                        textcoords="offset points", xytext=(0,10), ha='center', fontsize=9)
        
        # Plot 2: Generalization drop
        plt.subplot(1, 2, 2)
        generalization_drop = np.array(in_sample_corrs) - np.array(cv_corrs)
        plt.bar(components, generalization_drop, color='orange', alpha=0.7, edgecolor='black')
        plt.xlabel('PLS Component', fontsize=12)
        plt.ylabel('Generalization Drop', fontsize=12)
        plt.title('Correlation Drop (Overfitting Indicator)', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        
        # Add values on bars
        for i, drop in enumerate(generalization_drop):
            plt.text(components[i], drop + 0.005, f'{drop:.3f}', ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"CV comparison plot saved to {save_path}")
        
        plt.show()
    
    def fit_pls_model(self, X, Y):
        """
        Fit PLS model to find latent variables maximizing X-Y covariance.
        
        Args:
            X: Brain connectivity data (n_samples × n_brain_features)
            Y: Behavioral data (n_samples × n_behavioral_measures)
            
        Returns:
            pls_model: Fitted PLSCanonical model
        """
        logger.info(f"Fitting PLS model with {self.n_components} components")
        logger.info(f"X shape: {X.shape}, Y shape: {Y.shape}")
        
        # Initialize and fit PLS model
        self.pls_model = PLSCanonical(n_components=self.n_components, scale=True)
        self.pls_model.fit(X.values, Y.values)
        
        logger.info("PLS model fitting completed")
        return self.pls_model
    
    def calculate_canonical_correlations(self, X_scores, Y_scores):
        """
        Calculate canonical correlations between X and Y scores.
        
        Args:
            X_scores: X latent variables (n_samples × n_components)
            Y_scores: Y latent variables (n_samples × n_components)
            
        Returns:
            correlations: Array of canonical correlations for each component
        """
        correlations = []
        
        for i in range(self.n_components):
            corr, p_value = pearsonr(X_scores[:, i], Y_scores[:, i])
            correlations.append(corr)
        
        return np.array(correlations)
    
    def calculate_variance_explained(self, X, Y, X_scores, Y_scores, X_loadings, Y_loadings):
        """
        Calculate variance explained by each PLS component using R² method.
        
        This method uses R-squared to avoid the >100% issue with traditional variance calculation.
        
        Mathematical approach:
        1. For each feature, calculate R² between original and reconstructed values
        2. Average R² across all features
        3. This gives the proportion of variance explained (0-100%)
        
        Args:
            X: Original X data
            Y: Original Y data
            X_scores: X latent variable scores
            Y_scores: Y latent variable scores
            X_loadings: X loadings/coefficients
            Y_loadings: Y loadings/coefficients
            
        Returns:
            var_exp_X: R²-based variance explained in X for each component
            var_exp_Y: R²-based variance explained in Y for each component
        """
        var_exp_X = []
        var_exp_Y = []
        
        for i in range(self.n_components):
            # Reconstruct data using first i+1 components
            X_reconstructed = X_scores[:, :(i+1)] @ X_loadings[:, :(i+1)].T
            Y_reconstructed = Y_scores[:, :(i+1)] @ Y_loadings[:, :(i+1)].T
            
            # Calculate R² for each feature and average
            r2_scores_X = []
            r2_scores_Y = []
            
            for j in range(X.shape[1]):
                # R² = 1 - SS_res / SS_tot
                ss_res_X = np.sum((X.values[:, j] - X_reconstructed[:, j])**2)
                ss_tot_X = np.sum((X.values[:, j] - np.mean(X.values[:, j]))**2)
                r2_X = 1 - (ss_res_X / ss_tot_X) if ss_tot_X > 0 else 0
                r2_scores_X.append(max(0, r2_X))  # Ensure non-negative
            
            for j in range(Y.shape[1]):
                ss_res_Y = np.sum((Y.values[:, j] - Y_reconstructed[:, j])**2)
                ss_tot_Y = np.sum((Y.values[:, j] - np.mean(Y.values[:, j]))**2)
                r2_Y = 1 - (ss_res_Y / ss_tot_Y) if ss_tot_Y > 0 else 0
                r2_scores_Y.append(max(0, r2_Y))  # Ensure non-negative
            
            # Average R² across all features
            var_exp_X.append(np.mean(r2_scores_X) * 100)
            var_exp_Y.append(np.mean(r2_scores_Y) * 100)
        
        return np.array(var_exp_X), np.array(var_exp_Y)
    
    def run_analysis(self, X_brain, Y_behavior, covariates):
        """
        Run the complete PLS analysis pipeline.
        
        Args:
            X_brain: Brain connectivity data
            Y_behavior: Behavioral data
            covariates: Covariates to regress out (sex, meanFD)
            
        Returns:
            results: Dictionary containing all analysis results
        """
        logger.info("Starting PLS analysis pipeline")
        
        # Step 1: Regress out covariates and standardize
        logger.info("Step 1: Regressing out covariates and standardizing residuals")
        self.X_resid = self.regress_out_covariates(X_brain, covariates)
        self.Y_resid = self.regress_out_covariates(Y_behavior, covariates)
        
        # Verify standardization
        logger.info("Verification - residuals should have μ≈0, σ≈1:")
        logger.info(f"X residuals - mean: {self.X_resid.mean().mean():.4f}, std: {self.X_resid.std().mean():.4f}")
        logger.info(f"Y residuals - mean: {self.Y_resid.mean().mean():.4f}, std: {self.Y_resid.std().mean():.4f}")
        
        # Step 2: Fit PLS model
        logger.info("Step 2: Fitting PLS model")
        self.pls_model = self.fit_pls_model(self.X_resid, self.Y_resid)
        
        # Extract PLS results
        X_scores = self.pls_model._x_scores
        Y_scores = self.pls_model._y_scores
        X_loadings = self.pls_model.x_loadings_
        Y_loadings = self.pls_model.y_loadings_
        
        # Step 3: Calculate canonical correlations
        logger.info("Step 3: Calculating canonical correlations")
        canonical_corrs = self.calculate_canonical_correlations(X_scores, Y_scores)
        
        # Step 4: Calculate variance explained
        logger.info("Step 4: Calculating variance explained")
        var_exp_X, var_exp_Y = self.calculate_variance_explained(
            self.X_resid, self.Y_resid, X_scores, Y_scores, X_loadings, Y_loadings
        )
        
        # Compile results
        self.results = {
            'component_id': np.arange(1, self.n_components + 1),
            'canonical_correlation': canonical_corrs,
            'variance_explained_X': var_exp_X,
            'variance_explained_Y': var_exp_Y,
            'X_scores': X_scores,
            'Y_scores': Y_scores,
            'X_loadings': X_loadings,
            'Y_loadings': Y_loadings
        }
        
        logger.info("PLS analysis completed successfully")
        return self.results

    def save_all_results(self, output_dir=".", prefix="EFNY_PLS"):
        """
        Save all analysis results to CSV files.
        
        Args:
            output_dir: Directory to save files
            prefix: Prefix for filenames
            
        Returns:
            saved_files: Dictionary mapping file types to saved paths
        """
        import os
        from datetime import datetime
        
        if self.results is None:
            raise ValueError("Run analysis first before saving results")
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        saved_files = {}
        
        # 1. Save main results table
        results_df = self.create_results_table()
        main_file = os.path.join(output_dir, f"{prefix}_results_{timestamp}.csv")
        results_df.to_csv(main_file, index=False)
        saved_files['main_results'] = main_file
        logger.info(f"Main results saved to: {main_file}")
        
        # 2. Save CV results if available (store as instance variable first)
        if hasattr(self, 'cv_results') and self.cv_results is not None:
            cv_summary_file = os.path.join(output_dir, f"{prefix}_cv_summary_{timestamp}.csv")
            self.cv_results['comparison_table'].to_csv(cv_summary_file, index=False)
            saved_files['cv_summary'] = cv_summary_file
            logger.info(f"CV summary saved to: {cv_summary_file}")
            
            # Save individual fold results
            cv_folds_file = os.path.join(output_dir, f"{prefix}_cv_folds_{timestamp}.csv")
            fold_data = []
            for fold_result in self.cv_results['fold_results']:
                for i, corr in enumerate(fold_result['prediction_correlations']):
                    fold_data.append({
                        'Fold': fold_result['fold'],
                        'Component': i + 1,
                        'Prediction_Correlation': corr,
                        'Train_Size': fold_result['train_size'],
                        'Test_Size': fold_result['test_size']
                    })
            fold_df = pd.DataFrame(fold_data)
            fold_df.to_csv(cv_folds_file, index=False)
            saved_files['cv_folds'] = cv_folds_file
            logger.info(f"CV fold details saved to: {cv_folds_file}")
        
        return saved_files
    
    def create_results_table(self):
        """
        Create a formatted results table.
        
        Returns:
            results_df: DataFrame with PLS results
        """
        if self.results is None:
            raise ValueError("Run analysis first before creating results table")
        
        results_df = pd.DataFrame({
            'Component': self.results['component_id'],
            'Canonical_Correlation': self.results['canonical_correlation'],
            'R²_Brain_Variance_%': self.results['variance_explained_X'],
            'R²_Behavior_Variance_%': self.results['variance_explained_Y']
        })
        
        # Format for better display
        results_df['Canonical_Correlation'] = results_df['Canonical_Correlation'].round(4)
        results_df['R²_Brain_Variance_%'] = results_df['R²_Brain_Variance_%'].round(2)
        results_df['R²_Behavior_Variance_%'] = results_df['R²_Behavior_Variance_%'].round(2)
        
        return results_df
    
    def plot_scree_plot(self, save_path=None):
        """
        Create scree plot showing variance explained in Y.
        
        Args:
            save_path: Optional path to save the plot
        """
        if self.results is None:
            raise ValueError("Run analysis first before plotting")
        
        plt.figure(figsize=(12, 5))
        
        # Plot variance explained in Y
        components = self.results['component_id']
        var_exp_Y = self.results['variance_explained_Y']
        
        plt.subplot(1, 2, 1)
        plt.plot(components, var_exp_Y, 'bo-', linewidth=2, markersize=8)
        plt.xlabel('PLS Component')
        plt.ylabel('Variance Explained in Y (%)')
        plt.title('Scree Plot: Variance Explained in Behavioral Data')
        plt.grid(True, alpha=0.3)
        
        # Plot canonical correlations
        canonical_corrs = self.results['canonical_correlation']
        
        plt.subplot(1, 2, 2)
        plt.plot(components, canonical_corrs, 'ro-', linewidth=2, markersize=8)
        plt.xlabel('PLS Component')
        plt.ylabel('Canonical Correlation')
        plt.title('Canonical Correlations')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Scree plot saved to {save_path}")
        
        plt.show()
    
    def nested_cross_validation(self, X_brain, Y_behavior, covariates, n_splits=5, random_state=42):
        """
        Perform nested 5-fold cross-validation to assess out-of-sample generalization.
        
        Methodology:
        1. Split data into 5 folds randomly
        2. For each fold:
           - Use 4 folds for training, 1 fold for testing
           - Regress out covariates ONLY on training data
           - Apply same regression coefficients to test data
           - Standardize both sets using training statistics
           - Fit PLS model on training data
           - Project test data onto training weights
           - Calculate prediction correlation between X and Y scores
        
        Args:
            X_brain: Brain connectivity data (n_samples × n_features)
            Y_behavior: Behavioral data (n_samples × n_features)
            covariates: Covariates to regress out (sex, meanFD)
            n_splits: Number of CV folds (default: 5)
            random_state: Random seed for reproducibility
            
        Returns:
            cv_results: Dictionary with CV results and in-sample comparison
        """
        logger.info(f"Starting nested {n_splits}-fold cross-validation")
        logger.info(f"Data shapes - X: {X_brain.shape}, Y: {Y_behavior.shape}")
        
        # Initialize KFold cross-validator
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        
        # Store results for each fold
        fold_results = []
        in_sample_correlations = []
        
        # First get in-sample correlations for comparison
        logger.info("Calculating in-sample correlations for comparison...")
        in_sample_results = self.run_analysis(X_brain, Y_behavior, covariates)
        in_sample_correlations = in_sample_results['canonical_correlation'].copy()
        
        # Perform cross-validation
        fold = 1
        for train_idx, test_idx in kf.split(X_brain):
            logger.info(f"Processing fold {fold}/{n_splits}")
            
            # Split data
            X_train, X_test = X_brain.iloc[train_idx], X_brain.iloc[test_idx]
            Y_train, Y_test = Y_behavior.iloc[train_idx], Y_behavior.iloc[test_idx]
            cov_train, cov_test = covariates.iloc[train_idx], covariates.iloc[test_idx]
            
            logger.info(f"Train: {X_train.shape}, Test: {X_test.shape}")
            
            # Step 1: Regress out covariates on TRAINING data only
            logger.info("Regressing out covariates on training data...")
            X_train_resid = self.regress_out_covariates(X_train, cov_train)
            Y_train_resid = self.regress_out_covariates(Y_train, cov_train)
            
            # Step 2: Apply same regression to test data (prevent data leakage)
            logger.info("Applying covariate regression to test data...")
            # We need to manually apply the regression coefficients from training
            X_test_resid = self.apply_covariate_regression(X_test, cov_test, X_train, cov_train)
            Y_test_resid = self.apply_covariate_regression(Y_test, cov_test, Y_train, cov_train)
            
            # Step 3: Standardize using training statistics
            logger.info("Standardizing using training statistics...")
            X_train_scaled, X_test_scaled = self.standardize_with_training_stats(X_train_resid, X_test_resid)
            Y_train_scaled, Y_test_scaled = self.standardize_with_training_stats(Y_train_resid, Y_test_resid)
            
            # Step 4: Fit PLS model on training data
            logger.info("Fitting PLS model on training data...")
            pls_model = PLSCanonical(n_components=self.n_components, scale=True)
            pls_model.fit(X_train_scaled.values, Y_train_scaled.values)
            
            # Step 5: Project test data onto training weights
            logger.info("Projecting test data onto training weights...")
            X_test_scores = X_test_scaled.values @ pls_model.x_rotations_
            Y_test_scores = Y_test_scaled.values @ pls_model.y_rotations_
            
            # Step 6: Calculate prediction correlations for each component
            fold_correlations = []
            for comp in range(self.n_components):
                corr, p_val = pearsonr(X_test_scores[:, comp], Y_test_scores[:, comp])
                fold_correlations.append(corr)
            
            fold_results.append({
                'fold': fold,
                'train_size': len(train_idx),
                'test_size': len(test_idx),
                'prediction_correlations': fold_correlations
            })
            
            fold += 1
        
        # Calculate mean prediction correlations across folds
        mean_prediction_correlations = np.mean([fold['prediction_correlations'] for fold in fold_results], axis=0)
        
        # Compile final results
        cv_results = {
            'fold_results': fold_results,
            'mean_prediction_correlations': mean_prediction_correlations,
            'in_sample_correlations': in_sample_correlations,
            'comparison_table': self.create_cv_comparison_table(mean_prediction_correlations, in_sample_correlations)
        }
        
        logger.info("Cross-validation completed successfully")
        
        # Store CV results as instance variable for saving
        self.cv_results = cv_results
        
        return cv_results
    
    def save_all_results(self, output_dir="d:\\code\\data_driven_EF\\data\\EFNY\\results", prefix="pls_analysis"):
        """
        Save all PLS analysis results to files including weights, scores, and CV results.
        
        Args:
            output_dir: Directory to save results
            prefix: Prefix for output filenames
            
        Returns:
            saved_files: Dictionary with paths to saved files
        """
        import os
        from datetime import datetime
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        saved_files = {}
        
        # 1. Save main PLS results if available
        if self.results is not None:
            # Save summary results as CSV
            summary_file = os.path.join(output_dir, f"{prefix}_summary_{timestamp}.csv")
            summary_df = pd.DataFrame({
                'Component': self.results['component_id'],
                'Canonical_Correlation': self.results['canonical_correlation'],
                'R2_Brain_Variance_Pct': self.results['variance_explained_X'],
                'R2_Behavior_Variance_Pct': self.results['variance_explained_Y']
            })
            summary_df.to_csv(summary_file, index=False)
            saved_files['summary_csv'] = summary_file
            logger.info(f"Summary results saved to: {summary_file}")
            
            # Save weight matrices as NPZ (compressed numpy)
            weights_file = os.path.join(output_dir, f"{prefix}_weights_{timestamp}.npz")
            np.savez_compressed(weights_file,
                              X_loadings=self.results['X_loadings'],
                              Y_loadings=self.results['Y_loadings'],
                              X_scores=self.results['X_scores'],
                              Y_scores=self.results['Y_scores'])
            saved_files['weights_npz'] = weights_file
            logger.info(f"Weight matrices saved to: {weights_file}")
            
            # Save individual weight matrices as CSV for easy inspection
            for i, comp_id in enumerate(self.results['component_id']):
                # X loadings (brain weights)
                x_weights_file = os.path.join(output_dir, f"{prefix}_X_weights_component{comp_id}_{timestamp}.csv")
                x_weights_df = pd.DataFrame({
                    'Feature_Index': range(self.results['X_loadings'].shape[0]),
                    'Weight': self.results['X_loadings'][:, i]
                })
                x_weights_df.to_csv(x_weights_file, index=False)
                saved_files[f'X_weights_comp{comp_id}'] = x_weights_file
                
                # Y loadings (behavior weights)
                y_weights_file = os.path.join(output_dir, f"{prefix}_Y_weights_component{comp_id}_{timestamp}.csv")
                y_weights_df = pd.DataFrame({
                    'Behavioral_Measure': self.behavioral_metrics if hasattr(self, 'behavioral_metrics') else [f'Measure_{j}' for j in range(self.results['Y_loadings'].shape[0])],
                    'Weight': self.results['Y_loadings'][:, i]
                })
                y_weights_df.to_csv(y_weights_file, index=False)
                saved_files[f'Y_weights_comp{comp_id}'] = y_weights_file
                
                logger.info(f"Component {comp_id} weights saved separately")
        
        return saved_files


def main():
    """Main function to run PLS analysis on EFNY data."""
    
    # Define file paths - Schaefer400 version
    brain_data_path = r"d:\code\data_driven_EF\data\EFNY\fc_vector\Schaefer400\EFNY_Schaefer400_FC_matrix.npy"
    behavioral_data_path = r"d:\code\data_driven_EF\data\EFNY\table\demo\EFNY_behavioral_data.csv"
    sublist_path = r"d:\code\data_driven_EF\data\EFNY\table\sublist\sublist.txt"
    
    # Initialize analyzer
    analyzer = EFNYPLSAnalyzer(n_components=5)
    
    # Load data
    logger.info("=== EFNY PLS Brain-Behavior Analysis ===")
    try:
        X_brain, Y_behavior, covariates = analyzer.load_data(
            brain_data_path, behavioral_data_path, sublist_path
        )
        
        # Run PLS analysis
        results = analyzer.run_analysis(X_brain, Y_behavior, covariates)
        
        # Display results
        results_table = analyzer.create_results_table()
        print("\n=== PLS Analysis Results ===")
        print(results_table.to_string(index=False))
        
        # Create visualization
        analyzer.plot_scree_plot()
        
        # Run nested cross-validation for generalization assessment
        print("\n" + "="*80)
        print("RUNNING NESTED 5-FOLD CROSS-VALIDATION FOR GENERALIZATION ASSESSMENT")
        print("="*80)
        
        cv_results = analyzer.nested_cross_validation(X_brain, Y_behavior, covariates)
        
        # Display CV comparison results
        print("\n=== IN-SAMPLE vs CROSS-VALIDATION COMPARISON ===")
        print(cv_results['comparison_table'].to_string(index=False))
        
        # Plot CV comparison
        analyzer.plot_cv_comparison(cv_results)
        
        # Save all results
        print("\n" + "="*60)
        print("SAVING ALL ANALYSIS RESULTS")
        print("="*60)
        
        saved_files = analyzer.save_all_results(prefix="efny_schaefer400_pls")
        
        print("\n=== SAVED FILES ===")
        for file_type, file_path in saved_files.items():
            print(f"{file_type}: {file_path}")
        
        logger.info("Analysis completed successfully")
        
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        logger.info("Please ensure all data files are in the correct locations")
        logger.info("Running with simulated data instead...")
        
        # Fallback to simulated data
        from pls_brain_behavior_analysis import PLSSrainBehaviorAnalyzer
        demo_analyzer = PLSSrainBehaviorAnalyzer(n_components=5)
        X_brain, Y_behavior, covariates = demo_analyzer.generate_simulated_data()
        results = demo_analyzer.run_analysis(X_brain, Y_behavior, covariates)
        
        results_table = demo_analyzer.create_results_table()
        print("\n=== PLS Analysis Results (Simulated Data) ===")
        print(results_table.to_string(index=False))
        demo_analyzer.plot_scree_plot()


if __name__ == "__main__":
    main()