"""
Taylor Diagram Analysis for Foam Prediction Models (Basic Features)

This script creates Taylor diagrams to visualize the performance of different models
in terms of their correlation coefficient, standard deviation, and RMSE.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from matplotlib import rcParams
import matplotlib.patheffects as pe
from matplotlib.projections import PolarAxes
import mpl_toolkits.axisartist.grid_finder as gf
import mpl_toolkits.axisartist.floating_axes as fa

# Configure plot style
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Arial']
rcParams['axes.labelsize'] = 12
rcParams['axes.titlesize'] = 14
rcParams['xtick.labelsize'] = 10
rcParams['ytick.labelsize'] = 10
rcParams['legend.fontsize'] = 10

def taylor_statistics(predicted, reference):
    """
    Calculate statistics for Taylor diagram.
    """
    std_dev_p = np.std(predicted)
    std_dev_r = np.std(reference)
    correlation = np.corrcoef(predicted, reference)[0, 1]
    rmse = np.sqrt(np.mean((predicted - reference) ** 2))
    
    return std_dev_p, std_dev_r, correlation, rmse

class TaylorDiagram(object):
    """
    Taylor diagram implementation in the first quadrant.
    """
    def __init__(self, refstd, fig=None, rect=111, label='_'):
        """
        Set up Taylor diagram axes using mpl_toolkits.axisartist.floating_axes.
        
        Parameters:
        * refstd: reference standard deviation
        * fig: input Figure or None
        * rect: subplot definition
        * label: reference label
        """
        self.refstd = refstd
        
        tr = PolarAxes.PolarTransform()
        
        # Correlation labels
        rlocs = np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1])
        tlocs = np.arccos(rlocs)  # Conversion to polar angles
        gl1 = gf.FixedLocator(tlocs)  # Positions
        tf1 = gf.DictFormatter(dict(zip(tlocs, map(str, rlocs))))
        
        # Standard deviation axis
        self.smin = 0
        self.smax = 1.5 * self.refstd
        ghelper = fa.GridHelperCurveLinear(
            tr,
            extremes=(0, np.pi/2, # Theta min, max
                      self.smin, self.smax), # Radius min, max
            grid_locator1=gl1,
            tick_formatter1=tf1,
        )
        
        if fig is None:
            fig = plt.figure(figsize=(8, 8))
            
        ax = fa.FloatingSubplot(fig, rect, grid_helper=ghelper)
        fig.add_subplot(ax)
        
        # Adjust axes
        ax.axis["top"].set_axis_direction("bottom")
        ax.axis["top"].toggle(ticklabels=True, label=True)
        ax.axis["top"].major_ticklabels.set_axis_direction("top")
        ax.axis["top"].label.set_axis_direction("top")
        ax.axis["top"].label.set_text("Correlation coefficient")
        ax.axis["top"].label.set_fontweight("bold")
        
        ax.axis["left"].set_axis_direction("bottom")
        ax.axis["left"].label.set_text("Standard deviation")
        ax.axis["left"].label.set_fontweight("bold")
        
        ax.axis["right"].set_axis_direction("top")
        ax.axis["right"].toggle(ticklabels=True, label=True)
        ax.axis["right"].major_ticklabels.set_axis_direction("left")
        ax.axis["right"].label.set_text("Standard deviation")
        ax.axis["right"].label.set_fontweight("bold")
        
        ax.axis["bottom"].set_visible(False)
        
        # Contours for RMSE
        self._ax = ax  # Store axes for adding stuff later
        self.ax = ax.get_aux_axes(tr)  # Use auxiliary axes for plotting
        
        # Add reference point and reference std
        self.ax.plot([0], self.refstd, 'k*', markersize=12, label=label)
        
        # Add reference standard deviation contour
        t = np.linspace(0, np.pi/2)
        self.ax.plot(t, np.ones(t.shape) * self.refstd, 'k--', label='_')
        
        # Collect plotted artists
        self.samplePoints = []
        self.stds = []
        self.corrs = []
        self.artists = []
        
    def add_sample(self, stddev, corrcoef, *args, **kwargs):
        """
        Add sample to the diagram.
        """
        self.stds.append(stddev)
        self.corrs.append(corrcoef)
        
        theta = np.arccos(corrcoef)
        marker_effect = [pe.withStroke(linewidth=2, foreground='k')]
        
        # Plot point
        l, = self.ax.plot(theta, stddev, *args, path_effects=marker_effect, **kwargs)
        self.samplePoints.append(l)
        self.artists.append(l)
        return l
        
    def add_contours(self, levels=5, **kwargs):
        """
        Add RMSE contours.
        """
        rs, ts = np.meshgrid(
            np.linspace(self.smin, self.smax, 100),
            np.linspace(0, np.pi/2, 100)
        )
        
        # Compute RMSE
        rms = np.sqrt(self.refstd**2 + rs**2 - 2*self.refstd*rs*np.cos(ts))
        
        if isinstance(levels, int):
            levels = np.linspace(0, self.smax, levels+1)[1:]
            
        # Set default contour kwargs
        contour_defaults = {
            'colors': 'k',
            'linestyles': '--',
            'alpha': 0.5,
            'linewidths': 0.8
        }
        
        # Update with user-provided kwargs
        for k, v in contour_defaults.items():
            if k not in kwargs:
                kwargs[k] = v
                
        # Plot contours
        cs = self.ax.contour(ts, rs, rms, levels, **kwargs)
        return cs

def plot_taylor_diagram(statistics, labels, colors, output_path):
    """
    Create a Taylor diagram using the statistics from each model.
    
    Parameters:
    -----------
    statistics : list of tuples
        Each tuple contains (std_dev, std_dev_r, correlation, rmse) for one model
    labels : list of str
        Model names
    colors : list of str
        Colors for each model
    output_path : str
        Where to save the plot
    """
    # Create figure
    fig = plt.figure(figsize=(10, 8))
    
    # Get reference standard deviation (first item in statistics)
    ref_std = statistics[0][1]
    
    # Create Taylor diagram
    diagram = TaylorDiagram(ref_std, fig=fig, rect=111, label='Reference')
    
    # Add contours for RMSE
    contour_levels = np.arange(0.2, 1.0, 0.2) * ref_std
    contours = diagram.add_contours(levels=contour_levels)
    
    # Add each model to the diagram
    for i, ((std_p, _, corr, _), label, color) in enumerate(zip(statistics[1:], labels[1:], colors[1:])):
        diagram.add_sample(std_p, corr, 'o', color=color, markersize=8, label=label)
    
    # Add legend
    legend = fig.legend(diagram.artists, labels, numpoints=1, 
                      loc='upper right', bbox_to_anchor=(1.05, 1.0), frameon=True)
    
    # Add grid
    plt.grid(True, alpha=0.3, linestyle='-', color='gray', linewidth=0.5)
    
    # Save the figure
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    print(f"Taylor diagram saved to {output_path}")
    plt.close()

# Create results directory if it doesn't exist
results_dir = 'basic_foam_predictions'
data_dir = os.path.join(results_dir, 'data')
images_dir = os.path.join(results_dir, 'visualizations')

if not os.path.exists(results_dir):
    os.makedirs(results_dir)
if not os.path.exists(images_dir):
    os.makedirs(images_dir)

# Initialize lists for statistics
statistics = []
labels = ['Reference']
colors = ['black']

# Get list of prediction files
prediction_files = [f for f in os.listdir(data_dir) if f.endswith('_predictions.csv')]

if not prediction_files:
    print("No prediction files found in the results directory.")
    print(f"Please run Simple_Foam_Predictions.py first to generate model predictions in {data_dir}")
    exit()

# Define model abbreviations and colors to match the example image
model_map = {
    'Ridge': 'Ridge',
    'Lasso': 'Lasso',
    'ElasticNet': 'ENR',
    'SVR': 'SVR',
    'RandomForest': 'RFR',
    'GradientBoosting': 'GBR',
    'XGBoost': 'XGB',
    'ensemble': 'ENS'
}

# Colors that match the example image more closely
model_colors = {
    'Ridge': '#FF624D',  # Reddish
    'Lasso': '#FF0000',  # Red
    'ElasticNet': '#00FF00',  # Green
    'SVR': '#0000FF',  # Blue
    'RandomForest': '#228B22',  # Forest Green
    'GradientBoosting': '#FFA500',  # Orange
    'XGBoost': '#800080',  # Purple
    'ensemble': '#000000'  # Black
}

# Process each model's predictions
model_stats = {}
first_file = True

for file in prediction_files:
    model_name = file.replace('_predictions.csv', '')
    predictions_df = pd.read_csv(os.path.join(data_dir, file))
    
    # Get actual and predicted values
    y_true = predictions_df['Actual'].values
    y_pred = predictions_df['Predicted'].values
    
    # For the first file, add reference statistics
    if first_file:
        statistics.append((np.std(y_true), np.std(y_true), 1.0, 0.0))
        first_file = False
    
    # Calculate statistics for this model
    stats = taylor_statistics(y_pred, y_true)
    statistics.append(stats)
    
    # Use abbreviated model names
    display_name = model_map.get(model_name, model_name)
    labels.append(display_name)
    
    # Use predefined colors
    colors.append(model_colors.get(model_name, '#17becf'))
    
    # Store statistics for report
    model_stats[model_name] = {
        'display_name': display_name,
        'std_dev': stats[0],
        'correlation': stats[2],
        'rmse': stats[3]
    }

# Create Taylor diagram
plot_taylor_diagram(statistics, labels, colors, 
                   os.path.join(images_dir, 'taylor_diagram.png'))

# Create detailed analysis report
with open(os.path.join(data_dir, 'taylor_diagram_analysis.txt'), 'w') as f:
    f.write("TAYLOR DIAGRAM ANALYSIS REPORT (BASIC FEATURES)\n")
    f.write("============================================\n\n")
    f.write(f"Reference Standard Deviation: {statistics[0][1]:.4f}\n\n")
    f.write("Model Statistics:\n")
    f.write("-----------------\n")
    
    # Sort models by correlation coefficient
    sorted_models = sorted(model_stats.items(), 
                         key=lambda x: x[1]['correlation'], 
                         reverse=True)
    
    for model_name, stats in sorted_models:
        f.write(f"\n{model_name} ({stats['display_name']}):\n")
        f.write(f"  Standard Deviation: {stats['std_dev']:.4f}")
        f.write(f"  (Ratio to Reference: {stats['std_dev']/statistics[0][1]:.4f})\n")
        f.write(f"  Correlation Coefficient: {stats['correlation']:.4f}\n")
        f.write(f"  RMSE: {stats['rmse']:.4f}\n")
        
    # Add overall assessment
    f.write("\nOVERALL ASSESSMENT\n")
    f.write("=================\n")
    best_model = sorted_models[0][0]
    best_corr = sorted_models[0][1]['correlation']
    f.write(f"\nBest performing model (using only basic features): {best_model} ({model_stats[best_model]['display_name']})")
    f.write(f"\nBest correlation achieved: {best_corr:.4f}\n\n")
    
    # Add interpretation guidelines
    f.write("Interpretation Guidelines:\n")
    f.write("------------------------\n")
    f.write("1. Correlation Coefficient:\n")
    f.write("   > 0.9: Excellent\n")
    f.write("   0.7-0.9: Good\n")
    f.write("   0.5-0.7: Moderate\n")
    f.write("   < 0.5: Poor\n\n")
    f.write("2. Standard Deviation Ratio:\n")
    f.write("   Close to 1.0 is ideal\n")
    f.write("   < 1.0: Model underpredicts variability\n")
    f.write("   > 1.0: Model overpredicts variability\n\n")
    f.write("3. RMSE:\n")
    f.write("   Smaller values indicate better predictions\n")
    f.write("   Should be interpreted relative to the scale of your data\n")

print(f"Taylor diagram and analysis using basic features have been saved in the {results_dir} directory:")
print(f"- {images_dir}/taylor_diagram.png: Visual representation of model performance")
print(f"- {data_dir}/taylor_diagram_analysis.txt: Detailed statistical analysis") 