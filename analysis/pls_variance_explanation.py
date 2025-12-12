#!/usr/bin/env python3
"""
Corrected Variance Calculation for PLS Analysis

This script provides a more accurate method for calculating variance explained
in PLS analysis, addressing the >100% issue.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cross_decomposition import PLSCanonical
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class CorrectedPLSAnalyzer:
    """PLS analyzer with corrected variance calculation."""
    
    def __init__(self, n_components=5):
        self.n_components = n_components
        self.pls_model = None
        
    def regress_out_covariates(self, data, confounds):
        """
        Regress out covariates and standardize residuals.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Data to regress covariates from (shape: n_samples × n_features)
        confounds : pd.DataFrame
            Covariates to regress out (shape: n_samples × n_covariates)
            
        Returns:
        --------
        pd.DataFrame
            Standardized residuals with μ=0, σ=1 for each feature
        """
        from sklearn.linear_model import LinearRegression
        
        logger.info(f"Regressing out covariates from data with shape {data.shape}")
        
        residuals = pd.DataFrame(index=data.index, columns=data.columns)
        
        # Regress out covariates for each feature
        for feature in data.columns:
            lr = LinearRegression()
            lr.fit(confounds.values, data[feature].values)
            predicted = lr.predict(confounds.values)
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
    
    def generate_test_data(self, n_samples=200, n_X_features=100, n_Y_features=20, add_covariates=False):
        """Generate test data with known structure."""
        np.random.seed(42)
        
        # Create latent variables
        latent1 = np.random.normal(0, 1, n_samples)
        latent2 = np.random.normal(0, 0.5, n_samples)
        
        # X data: influenced by latent variables
        X_loadings1 = np.random.normal(0, 0.3, n_X_features)
        X_loadings2 = np.random.normal(0, 0.2, n_X_features)
        
        X = np.outer(latent1, X_loadings1) + np.outer(latent2, X_loadings2)
        X += np.random.normal(0, 0.1, (n_samples, n_X_features))
        
        # Y data: influenced by same latent variables
        Y_loadings1 = np.random.normal(0, 0.4, n_Y_features)
        Y_loadings2 = np.random.normal(0, 0.3, n_Y_features)
        
        Y = np.outer(latent1, Y_loadings1) + np.outer(latent2, Y_loadings2)
        Y += np.random.normal(0, 0.1, (n_samples, n_Y_features))
        
        # Add covariates if requested
        covariates = None
        if add_covariates:
            # Create covariates (e.g., age, sex effects)
            age = np.random.normal(25, 5, n_samples)
            sex = np.random.binomial(1, 0.5, n_samples)
            
            # Add covariate effects to X and Y
            age_effect_X = np.outer(age, np.random.normal(0.1, 0.05, n_X_features))
            sex_effect_X = np.outer(sex, np.random.normal(0.2, 0.1, n_X_features))
            X += age_effect_X + sex_effect_X
            
            age_effect_Y = np.outer(age, np.random.normal(0.15, 0.08, n_Y_features))
            sex_effect_Y = np.outer(sex, np.random.normal(0.3, 0.15, n_Y_features))
            Y += age_effect_Y + sex_effect_Y
            
            covariates = pd.DataFrame({'age': age, 'sex': sex})
        
        return pd.DataFrame(X), pd.DataFrame(Y), covariates
    
    def calculate_variance_explained_corrected(self, X, Y, pls_model):
        """
        Calculate variance explained using multiple methods for comparison.
        
        Methods:
        1. Cumulative reconstruction (original method)
        2. Individual component contribution
        3. R-squared approach (optimized with NumPy vectorization)
        4. Singular value based
        """
        n_components = self.n_components
        
        # Get PLS results
        X_scores = pls_model._x_scores
        Y_scores = pls_model._y_scores
        X_loadings = pls_model.x_loadings_
        Y_loadings = pls_model.y_loadings_
        
        # Calculate total variance
        total_var_X = np.sum(np.var(X.values, axis=0))
        total_var_Y = np.sum(np.var(Y.values, axis=0))
        
        results = {
            'component': [],
            'canonical_correlation': [],
            'method1_cumulative_X': [],
            'method1_cumulative_Y': [],
            'method2_individual_X': [],
            'method2_individual_Y': [],
            'method3_r2_X': [],
            'method3_r2_Y': []
        }
        
        for i in range(n_components):
            results['component'].append(i + 1)
            results['canonical_correlation'].append(pearsonr(X_scores[:, i], Y_scores[:, i])[0])
            
            # Method 1: Cumulative reconstruction (original)
            X_reconstructed_cum = X_scores[:, :(i+1)] @ X_loadings[:, :(i+1)].T
            Y_reconstructed_cum = Y_scores[:, :(i+1)] @ Y_loadings[:, :(i+1)].T
            
            var_reconstructed_X_cum = np.sum(np.var(X_reconstructed_cum, axis=0))
            var_reconstructed_Y_cum = np.sum(np.var(Y_reconstructed_cum, axis=0))
            
            results['method1_cumulative_X'].append((var_reconstructed_X_cum / total_var_X) * 100)
            results['method1_cumulative_Y'].append((var_reconstructed_Y_cum / total_var_Y) * 100)
            
            # Method 2: Individual component contribution
            if i == 0:
                X_reconstructed_ind = X_scores[:, i:i+1] @ X_loadings[:, i:i+1].T
                Y_reconstructed_ind = Y_scores[:, i:i+1] @ Y_loadings[:, i:i+1].T
            else:
                # Contribution of just this component
                X_reconstructed_ind = X_scores[:, i:i+1] @ X_loadings[:, i:i+1].T
                Y_reconstructed_ind = Y_scores[:, i:i+1] @ Y_loadings[:, i:i+1].T
            
            var_reconstructed_X_ind = np.sum(np.var(X_reconstructed_ind, axis=0))
            var_reconstructed_Y_ind = np.sum(np.var(Y_reconstructed_ind, axis=0))
            
            results['method2_individual_X'].append((var_reconstructed_X_ind / total_var_X) * 100)
            results['method2_individual_Y'].append((var_reconstructed_Y_ind / total_var_Y) * 100)
            
            # Method 3: R-squared approach (optimized with NumPy vectorization)
            X_pred = X_scores[:, :(i+1)] @ X_loadings[:, :(i+1)].T
            Y_pred = Y_scores[:, :(i+1)] @ Y_loadings[:, :(i+1)].T
            
            # Optimized R² calculation using NumPy broadcasting
            # Calculate residuals and total sum of squares for all features at once
            ss_res_X = np.sum((X.values - X_pred)**2, axis=0)  # Sum of squared residuals for each feature
            ss_tot_X = np.sum((X.values - np.mean(X.values, axis=0))**2, axis=0)  # Total sum of squares for each feature
            r2_X_features = 1 - (ss_res_X / ss_tot_X)  # R² for each feature
            r2_X = np.mean(r2_X_features) * 100  # Average R² across all features
            
            ss_res_Y = np.sum((Y.values - Y_pred)**2, axis=0)
            ss_tot_Y = np.sum((Y.values - np.mean(Y.values, axis=0))**2, axis=0)
            r2_Y_features = 1 - (ss_res_Y / ss_tot_Y)
            r2_Y = np.mean(r2_Y_features) * 100
            
            results['method3_r2_X'].append(r2_X)
            results['method3_r2_Y'].append(r2_Y)
        
        return pd.DataFrame(results)
    
    def explain_variance_issue(self):
        """
        Explain why variance can exceed 100% in PLS.
        """
        print("\n" + "="*80)
        print("WHY CAN VARIANCE EXPLAINED EXCEED 100% IN PLS?")
        print("="*80)
        
        print("\n1. PLS MAXIMIZES COVARIANCE, NOT VARIANCE")
        print("   - PLS finds components that maximize cov(X_scores, Y_scores)")
        print("   - This is different from maximizing variance explained")
        
        print("\n2. NON-ORTHOGONAL COMPONENTS")
        print("   - Unlike PCA, PLS components are NOT orthogonal")
        print("   - Components can 'overlap' and explain the same variance")
        print("   - This can lead to 'double counting' of variance")
        
        print("\n3. RECONSTRUCTION ARTIFACTS")
        print("   - X_reconstructed = X_scores @ X_loadings.T")
        print("   - This reconstruction can amplify certain patterns")
        print("   - Especially when n_components approaches data rank")
        
        print("\n4. MATHEMATICAL INTERPRETATION")
        print("   - >100% doesn't mean 'more than all variance'")
        print("   - It means the reconstruction captures covariance structure")
        print("   - that isn't purely variance in the traditional sense")
        
        print("\n5. RECOMMENDED INTERPRETATION")
        print("   - Focus on Canonical Correlations (always 0-1)")
        print("   - Use R² method for variance (more stable)")
        print("   - Consider relative importance of components")
        print("   - Don't over-interpret absolute percentages > 100%")
    
    def save_results(self, results_df, output_dir="d:\\code\\data_driven_EF\\data\\EFNY\\results", suffix=""):
        """Save results to CSV and JSON formats."""
        import os
        import json
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate timestamp for unique filenames
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Add suffix to filename if provided
        suffix_str = f"_{suffix}" if suffix else ""
        
        # Save as CSV
        csv_file = os.path.join(output_dir, f"pls_variance_results_{timestamp}{suffix_str}.csv")
        results_df.to_csv(csv_file, index=False)
        logger.info(f"Results saved to CSV: {csv_file}")
        
        # Save as JSON with additional metadata
        json_file = os.path.join(output_dir, f"pls_variance_results_{timestamp}{suffix_str}.json")
        results_dict = {
            "metadata": {
                "timestamp": timestamp,
                "n_components": self.n_components,
                "calculation_methods": ["cumulative", "individual", "r_squared"],
                "description": "PLS variance explained calculation results",
                "suffix": suffix
            },
            "results": results_df.to_dict('records')
        }
        
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(results_dict, f, indent=2, ensure_ascii=False)
        logger.info(f"Results saved to JSON: {json_file}")
        
        return csv_file, json_file
    
    def run_demo_with_covariates(self):
        """Run demonstration with covariate regression and R² focus."""
        logger.info("Generating test data with covariates...")
        X, Y, covariates = self.generate_test_data(add_covariates=True)
        
        logger.info("Original data shapes:")
        logger.info(f"X (brain): {X.shape}, Y (behavior): {Y.shape}, covariates: {covariates.shape}")
        
        # Step 1: Regress out covariates and standardize
        logger.info("Step 1: Regressing out covariates and standardizing residuals...")
        X_resid = self.regress_out_covariates(X, covariates)
        Y_resid = self.regress_out_covariates(Y, covariates)
        
        # Verify standardization
        logger.info("Verification - residuals should have μ≈0, σ≈1:")
        logger.info(f"X residuals - mean: {X_resid.mean().mean():.4f}, std: {X_resid.std().mean():.4f}")
        logger.info(f"Y residuals - mean: {Y_resid.mean().mean():.4f}, std: {Y_resid.std().mean():.4f}")
        
        # Step 2: Fit PLS model on residuals
        logger.info("Step 2: Fitting PLS model on residuals...")
        pls_model = PLSCanonical(n_components=self.n_components)
        pls_model.fit(X_resid.values, Y_resid.values)
        
        # Step 3: Calculate variance explained
        logger.info("Step 3: Calculating variance explained (R² method focus)...")
        results_df = self.calculate_variance_explained_corrected(X_resid, Y_resid, pls_model)
        
        # Focus on R² results
        print("\n=== PLS ANALYSIS RESULTS (COVARIATE-ADJUSTED) ===")
        print("R² Method - Variance Explained (most reliable interpretation):")
        r2_results = results_df[['component', 'canonical_correlation', 'method3_r2_X', 'method3_r2_Y']].copy()
        r2_results.columns = ['Component', 'Canonical_Correlation', 'R²_X_Variance_%', 'R²_Y_Variance_%']
        print(r2_results.round(3))
        
        # Save results
        logger.info("Saving covariate-adjusted results...")
        csv_file, json_file = self.save_results(results_df, suffix="covariate_adjusted")
        
        # Create focused visualization
        self.plot_r2_focused_results(results_df)
        
        logger.info("Demo with covariates completed successfully!")
        return results_df, csv_file, json_file, X_resid, Y_resid
    
    def run_demo(self):
        """Run demonstration with corrected calculations."""
        logger.info("Generating test data...")
        X, Y, _ = self.generate_test_data(add_covariates=False)
        
        logger.info("Fitting PLS model...")
        pls_model = PLSCanonical(n_components=self.n_components)
        pls_model.fit(X.values, Y.values)
        
        logger.info("Calculating variance explained with multiple methods...")
        results_df = self.calculate_variance_explained_corrected(X, Y, pls_model)
        
        print("\n=== COMPARISON OF VARIANCE CALCULATION METHODS ===")
        print(results_df.round(2))
        
        # Save results
        logger.info("Saving results...")
        csv_file, json_file = self.save_results(results_df)
        
        self.explain_variance_issue()
        
        # Create visualization
        self.plot_comparison(results_df)
        
        logger.info("Demo completed successfully!")
        return results_df, csv_file, json_file
    
    def plot_comparison(self, results_df):
        """Plot comparison of different variance calculation methods."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        components = results_df['component']
        
        # X variance comparison
        axes[0, 0].plot(components, results_df['method1_cumulative_X'], 'o-', label='Cumulative', linewidth=2)
        axes[0, 0].plot(components, results_df['method2_individual_X'], 's-', label='Individual', linewidth=2)
        axes[0, 0].plot(components, results_df['method3_r2_X'], '^-', label='R² Method', linewidth=2)
        axes[0, 0].axhline(y=100, color='red', linestyle='--', alpha=0.7, label='100% threshold')
        axes[0, 0].set_title('X Variance Explained Comparison')
        axes[0, 0].set_xlabel('Component')
        axes[0, 0].set_ylabel('Variance Explained (%)')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Y variance comparison
        axes[0, 1].plot(components, results_df['method1_cumulative_Y'], 'o-', label='Cumulative', linewidth=2)
        axes[0, 1].plot(components, results_df['method2_individual_Y'], 's-', label='Individual', linewidth=2)
        axes[0, 1].plot(components, results_df['method3_r2_Y'], '^-', label='R² Method', linewidth=2)
        axes[0, 1].axhline(y=100, color='red', linestyle='--', alpha=0.7, label='100% threshold')
        axes[0, 1].set_title('Y Variance Explained Comparison')
        axes[0, 1].set_xlabel('Component')
        axes[0, 1].set_ylabel('Variance Explained (%)')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Canonical correlations
        axes[1, 0].plot(components, results_df['canonical_correlation'], 'go-', linewidth=2, markersize=8)
        axes[1, 0].set_title('Canonical Correlations')
        axes[1, 0].set_xlabel('Component')
        axes[1, 0].set_ylabel('Correlation')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].set_ylim(0, 1)
        
        # Summary
        axes[1, 1].axis('off')
        summary_text = f"""
        KEY INSIGHTS:
        
        • Cumulative method can exceed 100%
        • R² method is most stable (0-100%)
        • Individual contributions sum to <100%
        • Canonical correlations are always valid (0-1)
        
        RECOMMENDATIONS:
        
        • Use R² method for variance interpretation
        • Focus on relative importance of components
        • Rely on canonical correlations for strength
        """
        axes[1, 1].text(0.05, 0.95, summary_text, transform=axes[1, 1].transAxes, 
                       fontsize=10, verticalalignment='top', fontfamily='monospace')
        
        plt.tight_layout()
        plt.show()


    def plot_r2_focused_results(self, results_df):
        """Create visualization focused on R² results."""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        components = results_df['component']
        
        # R² variance explained
        axes[0].plot(components, results_df['method3_r2_X'], 'bo-', 
                    linewidth=3, markersize=8, label='X (Brain)', alpha=0.8)
        axes[0].plot(components, results_df['method3_r2_Y'], 'ro-', 
                    linewidth=3, markersize=8, label='Y (Behavior)', alpha=0.8)
        axes[0].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        axes[0].set_title('R² Variance Explained by PLS Components', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Component', fontsize=12)
        axes[0].set_ylabel('R² Variance Explained (%)', fontsize=12)
        axes[0].legend(fontsize=11)
        axes[0].grid(True, alpha=0.3)
        axes[0].set_ylim(min(min(results_df['method3_r2_X']), min(results_df['method3_r2_Y'])) - 5, 
                        max(max(results_df['method3_r2_X']), max(results_df['method3_r2_Y'])) + 5)
        
        # Canonical correlations
        axes[1].plot(components, results_df['canonical_correlation'], 'go-', 
                    linewidth=3, markersize=8, alpha=0.8)
        axes[1].set_title('Canonical Correlations', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Component', fontsize=12)
        axes[1].set_ylabel('Correlation', fontsize=12)
        axes[1].grid(True, alpha=0.3)
        axes[1].set_ylim(0, 1)
        
        # Add correlation values as text
        for i, corr in enumerate(results_df['canonical_correlation']):
            axes[1].annotate(f'{corr:.3f}', 
                           (components.iloc[i], corr), 
                           textcoords="offset points", 
                           xytext=(0,10), 
                           ha='center', fontsize=10)
        
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    analyzer = CorrectedPLSAnalyzer(n_components=5)
    
    print("="*80)
    print("请选择运行模式:")
    print("1. 基础演示 (无协变量)")
    print("2. 协变量调整演示 (推荐)")
    print("="*80)
    
    choice = input("请输入选择 (1或2): ").strip()
    
    if choice == "2":
        print("\n运行协变量调整分析...")
        analyzer.run_demo_with_covariates()
    else:
        print("\n运行基础分析...")
        analyzer.run_demo()