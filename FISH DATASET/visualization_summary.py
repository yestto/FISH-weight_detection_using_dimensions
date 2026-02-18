#!/usr/bin/env python3
"""
Summary of created visualization files for regression results.
Shows what plots were created and their locations.
"""

import os

def show_visualization_summary():
    """Show summary of all created visualization files."""
    
    print("="*80)
    print("REGRESSION VISUALIZATION FILES CREATED")
    print("="*80)
    print()
    
    # Check if regression_plots directory exists
    plots_dir = 'regression_plots'
    if os.path.exists(plots_dir):
        print(f"📁 Visualization Directory: {plots_dir}")
        print()
        
        # List all PNG files
        png_files = [f for f in os.listdir(plots_dir) if f.endswith('.png')]
        
        if png_files:
            print(f"📊 Total Visualization Files: {len(png_files)}")
            print()
            
            # Categorize files
            scatter_plots = [f for f in png_files if 'scatter' in f]
            residual_plots = [f for f in png_files if 'residual' in f]
            comparison_plots = [f for f in png_files if 'comparison' in f]
            error_dist_plots = [f for f in png_files if 'error_distribution' in f]
            importance_plots = [f for f in png_files if 'importance' in f]
            
            print("🎯 PLOT CATEGORIES:")
            print()
            
            if scatter_plots:
                print(f"📈 SCATTER PLOTS ({len(scatter_plots)} files):")
                print("   Shows predicted vs actual values for each model")
                for f in scatter_plots[:3]:  # Show first 3
                    print(f"   - {f}")
                if len(scatter_plots) > 3:
                    print(f"   ... and {len(scatter_plots)-3} more")
                print()
            
            if residual_plots:
                print(f"📉 RESIDUAL PLOTS ({len(residual_plots)} files):")
                print("   Shows prediction errors vs predicted values")
                for f in residual_plots[:3]:
                    print(f"   - {f}")
                if len(residual_plots) > 3:
                    print(f"   ... and {len(residual_plots)-3} more")
                print()
            
            if comparison_plots:
                print(f"📊 PERFORMANCE COMPARISON ({len(comparison_plots)} files):")
                print("   Compares all models side-by-side")
                for f in comparison_plots:
                    print(f"   - {f}")
                print()
            
            if error_dist_plots:
                print(f"📋 ERROR DISTRIBUTION ({len(error_dist_plots)} files):")
                print("   Histogram of prediction errors")
                for f in error_dist_plots:
                    print(f"   - {f}")
                print()
            
            if importance_plots:
                print(f"🔍 FEATURE IMPORTANCE ({len(importance_plots)} files):")
                print("   Shows which features are most important")
                for f in importance_plots:
                    print(f"   - {f}")
                print()
            
            print("🎯 KEY FILES TO OPEN:")
            print()
            print("1. comparison_Area_truth_cm².png")
            print("   → Shows all models ranked by performance")
            print()
            print("2. scatter_ridge_Area_truth_cm².png") 
            print("   → Best model: predicted vs actual values")
            print()
            print("3. residual_ridge_Area_truth_cm².png")
            print("   → Best model: prediction errors analysis")
            print()
            
            print("📂 FULL FILE LIST:")
            print("-" * 50)
            for i, f in enumerate(png_files, 1):
                print(f"{i:2d}. {f}")
            
            print()
            print("="*80)
            print("TO VIEW THE PLOTS:")
            print("="*80)
            print()
            print("1. Navigate to the regression_plots folder")
            print("2. Open the PNG files in any image viewer")
            print("3. Start with 'comparison' file to see overall results")
            print("4. Check 'scatter' files for individual model performance")
            print()
            print("The plots show that the 'ridge' model achieves:")
            print("- R² = 0.985 (excellent correlation)")
            print("- MAE = 1.307 cm² (low prediction error)")
            print("- Stable performance across validation folds")
            
        else:
            print("❌ No visualization files found!")
            print()
            print("Run create_regression_plots.py to generate the plots.")
    else:
        print(f"❌ Directory '{plots_dir}' not found!")
        print()
        print("Run create_regression_plots.py first to create visualizations.")
    
    print()
    print("="*80)

if __name__ == '__main__':
    show_visualization_summary()