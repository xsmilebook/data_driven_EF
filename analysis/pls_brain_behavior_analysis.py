#!/usr/bin/env python3
"""
Partial Least Squares (PLS) Analysis for Brain-Behavior Association

This script performs PLS analysis to investigate the relationship between
functional connectivity matrices and behavioral measures while controlling
for covariates (sex and meanFD).

Mathematical Background:
- PLS finds latent variables that maximize covariance between X and Y
- Canonical correlation measures the strength of relationship between X-scores and Y-scores
- Variance explained is calculated as the ratio of variance captured by each component
to the total variance in the data
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.cross_decomposition import PLSCanonical
from sklearn.metrics import explained_variance_score
from scipy.stats import pearsonr
import seaborn as sns
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class PLSSrainBehaviorAnalyzer:
    """PLS analyzer for brain-behavior relationships."""
    
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
        
    def generate_simulated_data(self, n_subjects=394, n_brain_features=4950, n_behavioral=30):
        """
        Generate simulated data for testing the pipeline.
        
        Args:
            n_subjects: Number of subjects
            n_brain_features: Number of brain features (FC connections)
            n_behavioral: Number of behavioral measures
            
        Returns:
            X_brain: Simulated brain connectivity data (n_subjects × n_brain_features)
            Y_behavior: Simulated behavioral data (n_subjects × n_behavioral)
            covariates: Simulated covariates (n_subjects × 2)
        """
        logger.info(f"Generating simulated data: {n_subjects} subjects, {n_brain_features} brain features, {n_behavioral} behavioral measures")
        
        # Set random seed for reproducibility
        np.random.seed(42)
        
        # Generate covariates
        sex = np.random.choice([0, 1], size=n_subjects)  # 0=female, 1=male
        meanFD = np.random.normal(0.15, 0.05, size=n_subjects)  # Typical meanFD values
        covariates = pd.DataFrame({'sex': sex, 'meanFD': meanFD})
        
        # Generate brain data with some structure
        # Create latent factors that influence both brain and behavior
        n_latent = 3
        latent_factors = np.random.normal(0, 1, size=(n_subjects, n_latent))
        
        # Brain data influenced by latent factors and covariates
        brain_loadings = np.random.normal(0, 0.3, size=(n_latent, n_brain_features))
        X_brain = latent_factors @ brain_loadings + np.random.normal(0, 0.5, size=(n_subjects, n_brain_features))
        
        # Add covariate effects to brain data
        sex_effect = np.random.normal(0, 0.2, size=n_brain_features)
        meanFD_effect = np.random.normal(0, 2, size=n_brain_features)
        
        X_brain += np.outer(sex, sex_effect) + np.outer(meanFD, meanFD_effect)
        
        # Behavioral data influenced by same latent factors
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
        
        # Select subset of behavioral measures
        selected_metrics = behavioral_metrics[:n_behavioral]
        
        behavioral_loadings = np.random.normal(0, 0.4, size=(n_latent, n_behavioral))
        Y_behavior = latent_factors @ behavioral_loadings + np.random.normal(0, 0.3, size=(n_subjects, n_behavioral))
        
        # Add covariate effects to behavioral data
        sex_effect_beh = np.random.normal(0, 0.1, size=n_behavioral)
        meanFD_effect_beh = np.random.normal(0, 1, size=n_behavioral)
        
        Y_behavior += np.outer(sex, sex_effect_beh) + np.outer(meanFD, meanFD_effect_beh)
        
        X_brain = pd.DataFrame(X_brain, columns=[f'FC_{i}' for i in range(n_brain_features)])
        Y_behavior = pd.DataFrame(Y_behavior, columns=selected_metrics)
        
        logger.info("Simulated data generation completed")
        return X_brain, Y_behavior, covariates
    
    def regress_out_covariates(self, data, confounds):
        """
        Regress out covariates from data using linear regression.
        
        This function removes the effects of confounding variables by:
        1. Fitting a linear model: feature ~ confounds for each feature
        2. Computing residuals (actual - predicted)
        3. Returning the residuals as cleaned data
        
        Args:
            data: DataFrame of shape (n_samples, n_features)
            confounds: DataFrame of shape (n_samples, n_confounds)
            
        Returns:
            residuals: DataFrame of shape (n_samples, n_features) containing residuals
        """
        logger.info(f"Regressing out covariates from data with shape {data.shape}")
        
        residuals = pd.DataFrame(index=data.index, columns=data.columns)
        
        for feature in data.columns:
            # Fit linear regression: feature ~ confounds
            lr = LinearRegression()
            lr.fit(confounds.values, data[feature].values)
            
            # Predict values based on confounds
            predicted = lr.predict(confounds.values)
            
            # Calculate residuals (actual - predicted)
            residual = data[feature].values - predicted
            residuals[feature] = residual
        
        logger.info("Covariate regression completed")
        return residuals
    
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
        Calculate variance explained by each PLS component.
        
        Mathematical approach:
        1. For X: Var_explained = (variance of reconstructed X) / (total variance of X)
        2. For Y: Var_explained = (variance of reconstructed Y) / (total variance of Y)
        3. Reconstruction uses: X_reconstructed = X_scores @ X_loadings.T
        
        Args:
            X: Original X data
            Y: Original Y data
            X_scores: X latent variable scores
            Y_scores: Y latent variable scores
            X_loadings: X loadings/coefficients
            Y_loadings: Y loadings/coefficients
            
        Returns:
            var_exp_X: Variance explained in X for each component
            var_exp_Y: Variance explained in Y for each component
        """
        var_exp_X = []
        var_exp_Y = []
        
        # Calculate total variance in original data
        total_var_X = np.sum(np.var(X.values, axis=0))
        total_var_Y = np.sum(np.var(Y.values, axis=0))
        
        for i in range(self.n_components):
            # Reconstruct data using first i+1 components
            X_reconstructed = X_scores[:, :(i+1)] @ X_loadings[:, :(i+1)].T
            Y_reconstructed = Y_scores[:, :(i+1)] @ Y_loadings[:, :(i+1)].T
            
            # Calculate variance of reconstructed data
            var_reconstructed_X = np.sum(np.var(X_reconstructed, axis=0))
            var_reconstructed_Y = np.sum(np.var(Y_reconstructed, axis=0))
            
            # Calculate percentage variance explained
            var_exp_X.append((var_reconstructed_X / total_var_X) * 100)
            var_exp_Y.append((var_reconstructed_Y / total_var_Y) * 100)
        
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
        
        # Step 1: Regress out covariates
        logger.info("Step 1: Regressing out covariates")
        self.X_resid = self.regress_out_covariates(X_brain, covariates)
        self.Y_resid = self.regress_out_covariates(Y_behavior, covariates)
        
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
    
    def create_results_table(self):
        """
        Create a formatted results table.
        
        Returns:
            results_df: DataFrame with PLS results
        """
        if self.results is None:
            raise ValueError("Run analysis first before creating results table")
        
        results_df = pd.DataFrame({
            'Component_ID': self.results['component_id'],
            'Canonical_Correlation': self.results['canonical_correlation'],
            'Var_Exp_in_X (%)': self.results['variance_explained_X'],
            'Var_Exp_in_Y (%)': self.results['variance_explained_Y']
        })
        
        # Format for better display
        results_df['Canonical_Correlation'] = results_df['Canonical_Correlation'].round(4)
        results_df['Var_Exp_in_X (%)'] = results_df['Var_Exp_in_X (%)'].round(2)
        results_df['Var_Exp_in_Y (%)'] = results_df['Var_Exp_in_Y (%)'].round(2)
        
        return results_df
    
    def plot_scree_plot(self, save_path=None):
        """
        Create scree plot showing variance explained in Y.
        
        Args:
            save_path: Optional path to save the plot
        """
        if self.results is None:
            raise ValueError("Run analysis first before plotting")
        
        plt.figure(figsize=(10, 6))
        
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


def main():
    """Main function to demonstrate PLS analysis."""
    
    # Initialize analyzer
    analyzer = PLSSrainBehaviorAnalyzer(n_components=5)
    
    # Generate simulated data for testing
    logger.info("=== PLS Brain-Behavior Analysis Demo ===")
    X_brain, Y_behavior, covariates = analyzer.generate_simulated_data()
    
    # Run PLS analysis
    results = analyzer.run_analysis(X_brain, Y_behavior, covariates)
    
    # Display results
    results_table = analyzer.create_results_table()
    print("\n=== PLS Analysis Results ===")
    print(results_table.to_string(index=False))
    
    # Create visualization
    analyzer.plot_scree_plot()
    
    logger.info("Demo completed successfully")


if __name__ == "__main__":
    main()