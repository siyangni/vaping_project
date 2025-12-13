#!/usr/bin/env python
# coding: utf-8

# In[3]:


import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

def draw_box(ax, text, x, y, width, height, fontsize=12, facecolor='lightgrey', edgecolor='black', fontweight='bold'):
    """Draws a rounded rectangle with text."""
    box = FancyBboxPatch((x, y), width, height,
                         boxstyle="round,pad=0.02", 
                         linewidth=1.5, 
                         edgecolor=edgecolor, 
                         facecolor=facecolor)
    ax.add_patch(box)
    ax.text(x + width/2, y + height/2, text, 
            ha='center', va='center', fontsize=fontsize, fontweight=fontweight)

# Create figure and axis
fig, ax = plt.subplots(figsize=(14, 10))
ax.set_xlim(0, 10)
ax.set_ylim(0, 12)
ax.axis('off')

# Title
ax.text(2, 11.5, 'ML Expert System\n(Stage I)', fontsize=16, fontweight='bold', ha='center')
ax.text(7.5, 11.5, 'Nested Logistic Regression\n(Stage II)', fontsize=16, fontweight='bold', ha='center')

# Stage I boxes
ml_models = ["Lasso", "Random Forest", "Gradient Boost", "Histogram Gradient Boost", "XGBoost", "CatBoost"]
for i, model in enumerate(ml_models):
    draw_box(ax, model, x=1.5, y=10 - i*1.5, width=2, height=0.8)

# Stage II boxes
regression_models = [
    "Model 1 (Base Model)\n(Variables Identified by All 6 Models)",
    "Model 2\n(Variables Identified by at least 5/6 Models)",
    "Model 3\n(Variables Identified by at least 4/6 Models)",
    "Model 4\n(Variables Identified by at least 3/6 Models)",
    "Model 5\n(Variables Identified by at least 2/6 Models)",
    "Model 6 (Full Model)\n(Variables Identified by at least 1/6 Models)"
]

for i, model in enumerate(regression_models):
    draw_box(ax, model, x=6, y=10 - i*1.5, width=3.5, height=0.8)

# Arrows connecting stages
for i in range(6):
    ax.annotate('', xy=(6, 10.4 - i*1.5), xytext=(3.5, 10.4 - i*1.5),
                arrowprops=dict(arrowstyle='->', lw=1.5))

plt.tight_layout()
plt.show()



# In[4]:


import matplotlib.pyplot as plt
import numpy as np

# Sample data - approximating the values from your images
years = np.array([2017, 2018, 2019, 2020, 2021, 2022, 2023])

# Values for each model (estimated from images)
random_forest = np.array([0.2, 0.3, 0.33, 0.27, 0.7, 0.71, 0.72])
gradient_boost = np.array([0.17, 0.28, 0.35, 0.31, 0.71, 0.71, 0.73])
hist_gradient_boost = np.array([0.2, 0.28, 0.34, 0.27, 0.71, 0.71, 0.72])
xgboost = np.array([0.2, 0.28, 0.34, 0.27, 0.71, 0.71, 0.72])
catboost = np.array([0.2, 0.28, 0.34, 0.26, 0.71, 0.71, 0.72])

# Create a figure with a 3x2 grid (leaving one subplot empty)
fig, axes = plt.subplots(2, 3, figsize=(15, 10), constrained_layout=True)
fig.suptitle('Partial Dependence Plots (PDP) for Wave Across Different Models', fontsize=16)

# Flatten the axes array for easier indexing
axes = axes.flatten()

# Plot for each model
models = [
    ('Random Forest', random_forest),
    ('Gradient Boost', gradient_boost),
    ('Histogram Gradient Boost', hist_gradient_boost),
    ('XGBoost', xgboost),
    ('CatBoost', catboost)
]

for i, (model_name, values) in enumerate(models):
    ax = axes[i]
    ax.plot(years, values, 'b-', linewidth=2)
    ax.set_title(f'PDP for wave - {model_name}')
    ax.set_xlabel('wave')
    ax.set_ylabel('Partial dependence')
    ax.set_ylim(0.1, 0.8)
    ax.set_xticks(years)
    ax.grid(True, linestyle='--', alpha=0.7)

# Remove the last subplot if there are only 5 models
if len(models) < 6:
    fig.delaxes(axes[5])

# Add a common colorbar or legend if needed
# plt.colorbar(label='Partial dependence', ax=axes)

plt.savefig('pdp_wave_grid.png', dpi=300, bbox_inches='tight')
plt.show()


# In[6]:





# In[7]:


import matplotlib.pyplot as plt
import numpy as np

# X-axis values (marijuana use frequency/levels)
x_values = np.array([1, 2, 3, 4, 5, 6, 7])

# Y-axis values for each model (approximated from the images)
# Model 1: Random Forest
random_forest = np.array([0.57, 0.508, 0.503, 0.509, 0.502, 0.498, 0.477])

# Model 2: Gradient Boost
gradient_boost = np.array([0.565, 0.522, 0.51, 0.522, 0.51, 0.505, 0.48])

# Model 3: Histogram Gradient Boost
hist_gradient_boost = np.array([0.568, 0.515, 0.515, 0.552, 0.505, 0.505, 0.473])

# Model 4: XGBoost
xgboost = np.array([0.567, 0.508, 0.501, 0.548, 0.498, 0.495, 0.46])

# Model 5: CatBoost
catboost = np.array([0.57, 0.518, 0.51, 0.542, 0.505, 0.51, 0.478])

# Create a figure with a 2x3 grid (6 subplots, one will be empty)
fig, axes = plt.subplots(2, 3, figsize=(15, 10), constrained_layout=True)

# Set the main title for the entire figure
fig.suptitle('Partial Dependence Plots for Marijuana Use (V2116) Across Different Models', 
             fontsize=16, fontweight='bold')

# Flatten the axes array for easier indexing
axes = axes.flatten()

# Store model names and corresponding data
models = [
    ('Random Forest', random_forest),
    ('Gradient Boost', gradient_boost),
    ('Histogram Gradient Boost', hist_gradient_boost),
    ('XGBoost', xgboost),
    ('CatBoost', catboost)
]

# Plot each model's PDP in its own subplot
for i, (model_name, values) in enumerate(models):
    ax = axes[i]
    ax.plot(x_values, values, 'b-', linewidth=2)
    ax.set_title(f'PDP for V2116 - {model_name}', fontsize=12)
    ax.set_xlabel('V2116 (Marijuana Use)')
    ax.set_ylabel('Partial dependence')
    
    # Set consistent y-axis limits for better comparison
    ax.set_ylim(0.45, 0.58)
    
    # Set x-axis ticks to match the original plots
    ax.set_xticks(x_values)
    
    # Add grid for better readability
    ax.grid(True, linestyle='--', alpha=0.7)

# Remove the last subplot (since we only have 5 models)
fig.delaxes(axes[5])

# Add a common legend or explanation at the bottom
plt.figtext(0.5, 0.01, 
           'V2116 represents frequency of marijuana use (1=lowest, 7=highest)', 
           ha='center', fontsize=10, fontstyle='italic')

# Tight layout to optimize spacing
plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust rect to accommodate the text at the bottom

# Save the figure with high resolution
plt.savefig('marijuana_use_pdp_grid.png', dpi=300, bbox_inches='tight')

plt.show()


# In[8]:


import matplotlib.pyplot as plt
import numpy as np

# X-axis values (alcohol use frequency/levels)
x_values = np.array([1, 2, 3, 4, 5, 6, 7])

# Y-axis values for each model (approximated from the images)
# Model 1: Random Forest
random_forest = np.array([0.554, 0.557, 0.553, 0.542, 0.538, 0.535, 0.533])

# Model 2: Gradient Boost
gradient_boost = np.array([0.556, 0.555, 0.555, 0.543, 0.538, 0.534, 0.523])

# Model 3: Histogram Gradient Boost
hist_gradient_boost = np.array([0.558, 0.559, 0.559, 0.54, 0.535, 0.543, 0.495])

# Model 4: XGBoost
xgboost = np.array([0.551, 0.556, 0.556, 0.533, 0.54, 0.533, 0.51])

# Model 5: CatBoost
catboost = np.array([0.555, 0.55, 0.551, 0.539, 0.535, 0.534, 0.52])

# Create a figure with a 2x3 grid (6 subplots, one will be empty)
fig, axes = plt.subplots(2, 3, figsize=(15, 10), constrained_layout=True)

# Set the main title for the entire figure
fig.suptitle('Partial Dependence Plots for Alcohol Use (V2105) Across Different Models', 
             fontsize=16, fontweight='bold')

# Flatten the axes array for easier indexing
axes = axes.flatten()

# Store model names and corresponding data
models = [
    ('Random Forest', random_forest),
    ('Gradient Boost', gradient_boost),
    ('Histogram Gradient Boost', hist_gradient_boost),
    ('XGBoost', xgboost),
    ('CatBoost', catboost)
]

# Plot each model's PDP in its own subplot
for i, (model_name, values) in enumerate(models):
    ax = axes[i]
    ax.plot(x_values, values, 'b-', linewidth=2)
    ax.set_title(f'PDP for V2105 - {model_name}', fontsize=12)
    ax.set_xlabel('V2105 (Alcohol Use)')
    ax.set_ylabel('Partial dependence')
    
    # Set consistent y-axis limits for better comparison
    ax.set_ylim(0.49, 0.57)
    
    # Set x-axis ticks to match the original plots
    ax.set_xticks(x_values)
    
    # Add grid for better readability
    ax.grid(True, linestyle='--', alpha=0.7)

# Remove the last subplot (since we only have 5 models)
fig.delaxes(axes[5])

# Add a common legend or explanation at the bottom
plt.figtext(0.5, 0.01, 
           'V2105 represents frequency of alcohol use (1=lowest, 7=highest)', 
           ha='center', fontsize=10, fontstyle='italic')

# Tight layout to optimize spacing
plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust rect to accommodate the text at the bottom

# Save the figure with high resolution
plt.savefig('alcohol_use_pdp_grid.png', dpi=300, bbox_inches='tight')

plt.show()


# In[9]:


import matplotlib.pyplot as plt
import numpy as np

# X-axis values (cigarette use levels)
x_values = np.array([1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0])

# Y-axis values for each model (approximated from the images)
# Model 1: Random Forest
random_forest = np.array([0.558, 0.53, 0.498, 0.48, 0.48, 0.483, 0.488, 0.489, 0.489])

# Model 2: Gradient Boost
gradient_boost = np.array([0.558, 0.535, 0.49, 0.485, 0.477, 0.488, 0.501, 0.505, 0.51])

# Model 3: Histogram Gradient Boost
hist_gradient_boost = np.array([0.558, 0.52, 0.465, 0.455, 0.45, 0.49, 0.57, 0.585, 0.605])

# Model 4: XGBoost
xgboost = np.array([0.56, 0.525, 0.47, 0.46, 0.455, 0.505, 0.58, 0.59, 0.603])

# Model 5: CatBoost
catboost = np.array([0.558, 0.53, 0.498, 0.485, 0.47, 0.495, 0.545, 0.558, 0.57])

# Create a figure with a 2x3 grid (6 subplots, one will be empty)
fig, axes = plt.subplots(2, 3, figsize=(15, 10), constrained_layout=True)

# Set the main title for the entire figure
fig.suptitle('Partial Dependence Plots for Cigarette Use (V2101) Across Different Models', 
             fontsize=16, fontweight='bold')

# Flatten the axes array for easier indexing
axes = axes.flatten()

# Store model names and corresponding data
models = [
    ('Random Forest', random_forest),
    ('Gradient Boost', gradient_boost),
    ('Histogram Gradient Boost', hist_gradient_boost),
    ('XGBoost', xgboost),
    ('CatBoost', catboost)
]

# Plot each model's PDP in its own subplot
for i, (model_name, values) in enumerate(models):
    ax = axes[i]
    ax.plot(x_values, values, 'b-', linewidth=2)
    ax.set_title(f'PDP for V2101 - {model_name}', fontsize=12)
    ax.set_xlabel('V2101 (Cigarette Use)')
    ax.set_ylabel('Partial dependence')
    
    # Set consistent y-axis limits for better comparison across all plots
    ax.set_ylim(0.44, 0.62)
    
    # Set x-axis ticks to match the original plots
    ax.set_xticks([1.0, 2.0, 3.0, 4.0, 5.0])
    
    # Add grid for better readability
    ax.grid(True, linestyle='--', alpha=0.7)

# Remove the last subplot (since we only have 5 models)
fig.delaxes(axes[5])

# Add a common legend or explanation at the bottom
plt.figtext(0.5, 0.01, 
           'V2101 represents level of cigarette use (1=lowest, 5=highest)', 
           ha='center', fontsize=10, fontstyle='italic')

# Add a comparison note
plt.figtext(0.5, 0.03, 
           'Note: Different models show varying patterns at higher cigarette use levels', 
           ha='center', fontsize=10, fontstyle='italic')

# Tight layout to optimize spacing
plt.tight_layout(rect=[0, 0.05, 1, 0.95])  # Adjust rect to accommodate the text at the bottom

# Save the figure with high resolution
plt.savefig('cigarette_use_pdp_grid.png', dpi=300, bbox_inches='tight')

plt.show()


# In[3]:


import matplotlib.pyplot as plt
import numpy as np

# X-axis values (race categories)
x_values = np.array([1, 2, 3])

# Race category labels
race_labels = ["Black or\nAfrican American", "White\n(Caucasian)", "Hispanic"]

# Y-axis values for each model (approximated from the images)
# Model 1: Random Forest
random_forest = np.array([0.541, 0.5575, 0.5435])

# Model 2: Gradient Boost
gradient_boost = np.array([0.530, 0.561, 0.539])

# Model 3: Histogram Gradient Boost
hist_gradient_boost = np.array([0.531, 0.561, 0.535])

# Model 4: XGBoost
xgboost = np.array([0.535, 0.558, 0.540])

# Model 5: CatBoost
catboost = np.array([0.541, 0.557, 0.542])

# Create a figure with a 2x3 grid (6 subplots, one will be used for legend)
fig, axes = plt.subplots(2, 3, figsize=(15, 10), constrained_layout=True)

# Set the main title for the entire figure
fig.suptitle('Partial Dependence Plots for Race Across Different Models', 
             fontsize=16, fontweight='bold')

# Flatten the axes array for easier indexing
axes = axes.flatten()

# Store model names and corresponding data
models = [
    ('Random Forest', random_forest),
    ('Gradient Boost', gradient_boost),
    ('Histogram Gradient Boost', hist_gradient_boost),
    ('XGBoost', xgboost),
    ('CatBoost', catboost)
]

# Plot each model's PDP in its own subplot
for i, (model_name, values) in enumerate(models):
    ax = axes[i]
    ax.plot(x_values, values, 'b-', linewidth=2)
    ax.set_title(f'PDP for Race - {model_name}', fontsize=12)
    ax.set_xlabel('Race/Ethnicity')
    ax.set_ylabel('Partial dependence')
    
    # Set x-axis ticks with race labels
    ax.set_xticks(x_values)
    ax.set_xticklabels(race_labels)
    
    # Add grid for better readability
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Set consistent y-axis limits for better comparison
    # Using a range that encompasses all models while showing detail
    ax.set_ylim(0.525, 0.565)

# Use the last subplot for a legend explaining the race categories
ax_legend = axes[5]
ax_legend.axis('off')  # Hide axes

# Create a table with race explanations
table_data = [
    ['1', 'Black or African American'],
    ['2', 'White (Caucasian)'],
    ['3', 'Hispanic']
]
column_labels = ["Value", "Race/Ethnicity"]
table = ax_legend.table(
    cellText=table_data,
    colLabels=column_labels,
    loc='center',
    cellLoc='center',
    colWidths=[0.25, 0.6]  # Adjust column widths for better fit
)
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1.2, 1.5)

# Add title for the legend
ax_legend.text(0.5, 0.85, "Race Categories",
         horizontalalignment='center',
         verticalalignment='center',
         transform=ax_legend.transAxes,
         fontsize=12,
         fontweight='bold')

# Add observation about the pattern
plt.figtext(0.5, 0.01, 
           'Note: All models show highest partial dependence for White/Caucasian respondents (category 2),\nwith lower values for Black/African American (1) and Hispanic (3) respondents.',
           ha='center', fontsize=10, fontstyle='italic')

# Tight layout to optimize spacing
plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust rect to accommodate the text at the bottom

# Save the figure with high resolution
plt.savefig('race_pdp_grid_revised.png', dpi=300, bbox_inches='tight')

plt.show()


# In[12]:


import matplotlib.pyplot as plt
import numpy as np

# X-axis values (driving frequency categories)
x_values = np.array([1, 2, 3, 4, 5, 6])

# Y-axis values for each model (approximated from the images)
# Model 1: Random Forest
random_forest = np.array([0.5545, 0.5555, 0.5485, 0.5558, 0.5575, 0.5570])

# Model 2: Gradient Boost
gradient_boost = np.array([0.5572, 0.5582, 0.5475, 0.5545, 0.5570, 0.5575])

# Model 3: Histogram Gradient Boost
hist_gradient_boost = np.array([0.5505, 0.5595, 0.5500, 0.5530, 0.5580, 0.5540])

# Model 4: CatBoost
catboost = np.array([0.5525, 0.5600, 0.5490, 0.5515, 0.5575, 0.5545])

# Create a figure with a 2x2 grid (4 subplots)
fig, axes = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)

# Set the main title for the entire figure
fig.suptitle('Partial Dependence Plots for Driving Frequency (V2196) Across Different Models', 
             fontsize=16, fontweight='bold')

# Flatten the axes array for easier indexing
axes = axes.flatten()

# Store model names and corresponding data
models = [
    ('Random Forest', random_forest),
    ('Gradient Boost', gradient_boost),
    ('Histogram Gradient Boost', hist_gradient_boost),
    ('CatBoost', catboost)
]

# Plot each model's PDP in its own subplot
for i, (model_name, values) in enumerate(models):
    ax = axes[i]
    ax.plot(x_values, values, 'b-', linewidth=2)
    ax.set_title(f'PDP for V2196 - {model_name}', fontsize=12)
    ax.set_xlabel('V2196 (Driving Frequency)')
    ax.set_ylabel('Partial dependence')
    
    # Set x-axis ticks to match the original plots
    ax.set_xticks(x_values)
    
    # Add grid for better readability
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Set consistent y-axis limits for better comparison
    # Using a slightly wider range than needed to show the patterns clearly
    ax.set_ylim(0.547, 0.561)

# Add a common explanation at the bottom
plt.figtext(0.5, 0.01, 
           'Note: V2196 represents driving frequency categories (higher value = more frequent driving)', 
           ha='center', fontsize=10, fontstyle='italic')

# Tight layout to optimize spacing
plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust rect to accommodate the text at the bottom

# Save the figure with high resolution
plt.savefig('driving_frequency_pdp_grid.png', dpi=300, bbox_inches='tight')

plt.show()


# In[14]:


import matplotlib.pyplot as plt
import numpy as np

# X-axis values (political belief categories)
x_values = np.array([1, 2, 3, 4, 5, 6, 7, 8])

# Political belief category labels
category_labels = [
    "Strongly Republican", 
    "Mildly Republican", 
    "Mildly Democrat", 
    "Strongly Democrat", 
    "Independent", 
    "No preference", 
    "Other", 
    "Don't know"
]

# Y-axis values for each model (approximated from the images)
# Model 1: Random Forest
random_forest = np.array([0.5455, 0.5480, 0.5515, 0.5540, 0.5540, 0.5540, 0.5555, 0.5565])

# Model 2: Gradient Boost
gradient_boost = np.array([0.542, 0.547, 0.550, 0.554, 0.553, 0.553, 0.561, 0.562])

# Model 3: Histogram Gradient Boost
hist_gradient_boost = np.array([0.550, 0.547, 0.549, 0.557, 0.554, 0.549, 0.554, 0.563])

# Model 4: XGBoost
xgboost = np.array([0.549, 0.546, 0.545, 0.555, 0.557, 0.546, 0.557, 0.565])

# Model 5: CatBoost
catboost = np.array([0.550, 0.547, 0.546, 0.555, 0.558, 0.547, 0.558, 0.565])

# Create a figure with a 2x3 grid (6 subplots, one will be empty or used for legend)
fig, axes = plt.subplots(2, 3, figsize=(18, 12), constrained_layout=True)

# Set the main title for the entire figure
fig.suptitle('Partial Dependence Plots for Political Belief (V2166) Across Different Models', 
             fontsize=16, fontweight='bold')

# Flatten the axes array for easier indexing
axes = axes.flatten()

# Store model names and corresponding data
models = [
    ('Random Forest', random_forest),
    ('Gradient Boost', gradient_boost),
    ('Histogram Gradient Boost', hist_gradient_boost),
    ('XGBoost', xgboost),
    ('CatBoost', catboost)
]

# Plot each model's PDP in its own subplot
for i, (model_name, values) in enumerate(models):
    ax = axes[i]
    ax.plot(x_values, values, 'b-', linewidth=2)
    ax.set_title(f'PDP for V2166 - {model_name}', fontsize=12)
    ax.set_xlabel('Political Belief')
    ax.set_ylabel('Partial dependence')
    
    # Set x-axis ticks with category labels
    ax.set_xticks(x_values)
    
    # Add grid for better readability
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Adjust y-axis limits to make patterns visible but maintain comparability
    y_min = min(values) - 0.002
    y_max = max(values) + 0.002
    ax.set_ylim(y_min, y_max)

# Use the last subplot for a legend explaining the category labels
ax_legend = axes[5]
ax_legend.axis('off')  # Hide axes

# Create a table with category explanations
table_data = [[f"{i+1}", label] for i, label in enumerate(category_labels)]
column_labels = ["Value", "Political Belief"]
table = ax_legend.table(
    cellText=table_data,
    colLabels=column_labels,
    loc='center',
    cellLoc='center'
)
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 1.5)

# Add explanation about the data
plt.figtext(0.5, 0.01, 
           'Note: Higher partial dependence values may indicate increased likelihood of a particular outcome.',
           ha='center', fontsize=10, fontstyle='italic')

# Tight layout to optimize spacing
plt.tight_layout(rect=[0, 0.03, 1, 0.95])

# Save the figure with high resolution
plt.savefig('political_belief_pdp_grid.png', dpi=300, bbox_inches='tight')

plt.show()


# In[16]:


import matplotlib.pyplot as plt
import numpy as np

# X-axis values (region categories)
x_values = np.array([1, 2, 3, 4])

# Region category labels
region_labels = ["Northeast", "Midwest", "South", "West"]

# Y-axis values for each model (approximated from the images)
# Model 1: Random Forest
random_forest = np.array([0.5505, 0.5520, 0.5525, 0.5610])

# Model 2: Gradient Boost
gradient_boost = np.array([0.547, 0.551, 0.550, 0.571])

# Model 3: Histogram Gradient Boost
hist_gradient_boost = np.array([0.549, 0.549, 0.551, 0.572])

# Model 4: XGBoost
xgboost = np.array([0.5505, 0.5520, 0.5525, 0.5610])

# Model 5: CatBoost
catboost = np.array([0.549, 0.551, 0.550, 0.570])

# Create a figure with a 2x3 grid (6 subplots, one will be empty or used for legend)
fig, axes = plt.subplots(2, 3, figsize=(16, 10), constrained_layout=True)

# Set the main title for the entire figure
fig.suptitle('Partial Dependence Plots for Region (V13) Across Different Models', 
             fontsize=16, fontweight='bold')

# Flatten the axes array for easier indexing
axes = axes.flatten()

# Store model names and corresponding data
models = [
    ('Random Forest', random_forest),
    ('Gradient Boost', gradient_boost),
    ('Histogram Gradient Boost', hist_gradient_boost),
    ('XGBoost', xgboost),
    ('CatBoost', catboost)
]

# Plot each model's PDP in its own subplot
for i, (model_name, values) in enumerate(models):
    ax = axes[i]
    ax.plot(x_values, values, 'b-', linewidth=2)
    ax.set_title(f'PDP for V13 - {model_name}', fontsize=12)
    ax.set_xlabel('Region')
    ax.set_ylabel('Partial dependence')
    
    # Set x-axis ticks with region labels
    ax.set_xticks(x_values)
    ax.set_xticklabels(region_labels)
    
    # Add grid for better readability
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Set consistent y-axis limits for better comparison
    # Using a range that encompasses all models while showing detail
    ax.set_ylim(0.545, 0.575)

# Use the last subplot for a legend explaining the regions
ax_legend = axes[5]
ax_legend.axis('off')  # Hide axes

# Create a table with region explanations - clean format to match the image provided
table_data = [[f"{i+1}", label] for i, label in enumerate(region_labels)]
column_labels = ["Value", "Census Region"]
table = ax_legend.table(
    cellText=table_data,
    colLabels=column_labels,
    loc='center',
    cellLoc='center',
    colWidths=[0.25, 0.55]  # Adjust column widths for better fit
)
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1.2, 1.5)

# Add a title for the table above it
ax_legend.text(0.5, 0.85, "V13: Region of the country",
         horizontalalignment='center',
         verticalalignment='center',
         transform=ax_legend.transAxes,
         fontsize=12,
         fontweight='bold')

# Add the description below the table
ax_legend.text(0.5, 0.1, "Based on Census categories, in which\nrespondent's school is located",
         horizontalalignment='center',
         verticalalignment='center',
         transform=ax_legend.transAxes,
         fontsize=10)

# Add insight about the patterns
plt.figtext(0.5, 0.01, 
           'Note: All models show higher partial dependence for the West region (4),\nindicating a stronger relationship with the target variable.',
           ha='center', fontsize=10, fontstyle='italic')

# Tight layout to optimize spacing
plt.tight_layout(rect=[0, 0.03, 1, 0.95])

# Save the figure with high resolution
plt.savefig('region_pdp_grid.png', dpi=300, bbox_inches='tight')

plt.show()


# In[22]:


import matplotlib.pyplot as plt
import numpy as np

# X-axis values (grade categories)
x_values = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])

# Grade category labels
grade_labels = [
    "D (69 or below)", 
    "C- (70-72)", 
    "C (73-76)", 
    "C+ (77-79)", 
    "B- (80-82)", 
    "B (83-86)", 
    "B+ (87-89)", 
    "A- (90-92)", 
    "A (93-100)"
]

# Reversed grade labels (for display in the legend table, showing 9=A, etc.)
reversed_grade_labels = list(reversed(grade_labels))
reversed_values = list(reversed(x_values))

# Y-axis values for each model (approximated from the images)
# Model 1: Random Forest
random_forest = np.array([0.545, 0.546, 0.549, 0.550, 0.552, 0.551, 0.551, 0.553, 0.555])

# Model 2: Gradient Boost
gradient_boost = np.array([0.518, 0.522, 0.550, 0.555, 0.554, 0.552, 0.551, 0.553, 0.554])

# Model 3: Histogram Gradient Boost
hist_gradient_boost = np.array([0.551, 0.518, 0.557, 0.551, 0.560, 0.55, 0.551, 0.553, 0.555])

# Model 4: XGBoost
xgboost = np.array([0.551, 0.532, 0.553, 0.550, 0.557, 0.549, 0.550, 0.553, 0.554])

# Model 5: CatBoost
catboost = np.array([0.551, 0.527, 0.552, 0.550, 0.556, 0.549, 0.550, 0.553, 0.555])

# Create a figure with a 2x3 grid (6 subplots, one will be used for legend)
fig, axes = plt.subplots(2, 3, figsize=(18, 12), constrained_layout=True)

# Set the main title for the entire figure
fig.suptitle('Partial Dependence Plots for Academic Grades (V2179) Across Different Models', 
             fontsize=16, fontweight='bold')

# Flatten the axes array for easier indexing
axes = axes.flatten()

# Store model names and corresponding data
models = [
    ('Random Forest', random_forest),
    ('Gradient Boost', gradient_boost),
    ('Histogram Gradient Boost', hist_gradient_boost),
    ('XGBoost', xgboost),
    ('CatBoost', catboost)
]

# Plot each model's PDP in its own subplot
for i, (model_name, values) in enumerate(models):
    ax = axes[i]
    ax.plot(x_values, values, 'b-', linewidth=2)
    ax.set_title(f'PDP for V2179 - {model_name}', fontsize=12)
    ax.set_xlabel('Grade Category')
    ax.set_ylabel('Partial dependence')
    
    # Set x-axis ticks at each value
    ax.set_xticks(x_values)
    
    # Add grid for better readability
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Set y-axis limits to better show the patterns
    # Use model-specific limits to highlight pattern
    y_min = min(values) - 0.002
    y_max = max(values) + 0.002
    ax.set_ylim(y_min, y_max)

# Use the last subplot for a legend explaining the grade categories
ax_legend = axes[5]
ax_legend.axis('off')  # Hide axes

# Create a table with grade explanations
table_data = [[f"{val}", label] for val, label in zip(reversed_values, reversed_grade_labels)]
column_labels = ["Value", "Grade Level"]
table = ax_legend.table(
    cellText=table_data,
    colLabels=column_labels,
    loc='center',
    cellLoc='center',
    colWidths=[0.2, 0.6]  # Adjust column widths for better fit
)
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 1.5)

# Add title for the legend
ax_legend.text(0.5, 0.85, "V2179: Average Grades in School Year",
         horizontalalignment='center',
         verticalalignment='center',
         transform=ax_legend.transAxes,
         fontsize=12,
         fontweight='bold')

# Note about interpretation
plt.figtext(0.8, 0.005, 
           'Note: Higher values (9) represent better grades (A), while lower values (1) represent poorer grades (D).',
           ha='center', fontsize=10, fontstyle='italic')

# Save the figure with high resolution
plt.savefig('academic_grades_pdp_grid.png', dpi=300, bbox_inches='tight')

plt.show()


# In[18]:


import matplotlib.pyplot as plt
import numpy as np

# X-axis values (graduate school aspiration levels)
x_values = np.array([1, 2, 3, 4])

# Graduate school aspiration labels
aspiration_labels = [
    "Definitely\nWon't", 
    "Probably\nWon't", 
    "Probably\nWill", 
    "Definitely\nWill"
]

# Y-axis values for each model (approximated from the images)
# Model 1: Random Forest
random_forest = np.array([0.5513, 0.5535, 0.5545, 0.5498])

# Model 2: Histogram Gradient Boost (note: there are 4 models in total, not 5 as in previous examples)
hist_gradient_boost = np.array([0.5505, 0.5560, 0.5555, 0.5450])

# Model 3: XGBoost
xgboost = np.array([0.5490, 0.5560, 0.5550, 0.5440])

# Model 4: CatBoost
catboost = np.array([0.5495, 0.5605, 0.5550, 0.5440])

# Create a figure with a 2x2 grid (4 subplots)
fig, axes = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)

# Set the main title for the entire figure
fig.suptitle('Partial Dependence Plots for Graduate School Aspiration (V2184) Across Different Models', 
             fontsize=16, fontweight='bold')

# Flatten the axes array for easier indexing
axes = axes.flatten()

# Store model names and corresponding data
models = [
    ('Random Forest', random_forest),
    ('Histogram Gradient Boost', hist_gradient_boost),
    ('XGBoost', xgboost),
    ('CatBoost', catboost)
]

# Plot each model's PDP in its own subplot
for i, (model_name, values) in enumerate(models):
    ax = axes[i]
    ax.plot(x_values, values, 'b-', linewidth=2)
    ax.set_title(f'PDP for V2184 - {model_name}', fontsize=12)
    ax.set_xlabel('Graduate School Aspiration')
    ax.set_ylabel('Partial dependence')
    
    # Set x-axis ticks with aspiration labels
    ax.set_xticks(x_values)
    ax.set_xticklabels(aspiration_labels)
    
    # Add grid for better readability
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Set consistent y-axis limits for better comparison
    # Using a range that shows the pattern while keeping comparison fair
    y_min = 0.544
    y_max = 0.562
    ax.set_ylim(y_min, y_max)

# Add explanation about the variable
plt.figtext(0.5, 0.01, 
           'V2184: "Attend graduate or professional school after college"\n1="Definitely Won\'t" 2="Probably Won\'t" 3="Probably Will" 4="Definitely Will"',
           ha='center', fontsize=10, fontstyle='italic')

# Add insight about the pattern
plt.figtext(0.5, 0.05, 
           'Note: All models show an inverted U-shape with peak at "Probably Won\'t" (2),\nindicating a non-linear relationship with the target variable.',
           ha='center', fontsize=10, fontstyle='italic')

# Tight layout to optimize spacing
plt.tight_layout(rect=[0, 0.08, 1, 0.95])

# Save the figure with high resolution
plt.savefig('graduate_school_pdp_grid.png', dpi=300, bbox_inches='tight')

plt.show()


# In[20]:


import matplotlib.pyplot as plt
import numpy as np

# X-axis values (graduate school aspiration categories)
x_values = np.array([1, 2, 3, 4])

# Graduate school aspiration labels
aspiration_labels = [
    "Definitely Won't", 
    "Probably Won't", 
    "Probably Will", 
    "Definitely Will"
]

# Y-axis values for each model (approximated from the images)
# Model 1: Random Forest
random_forest = np.array([0.5520, 0.5540, 0.5550, 0.5500])

# Model 2: Histogram Gradient Boost
hist_gradient_boost = np.array([0.5505, 0.5560, 0.5560, 0.5450])

# Model 3: XGBoost
xgboost = np.array([0.5490, 0.5560, 0.5550, 0.5460])

# Model 4: CatBoost
catboost = np.array([0.5500, 0.5605, 0.5550, 0.5470])

# Model 5: Single PDP from second image
single_pdp = np.array([0.5510, 0.5558, 0.5545, 0.5500])

# Create a figure with a 2x3 grid (6 subplots, one will be used for legend)
fig, axes = plt.subplots(2, 3, figsize=(18, 12), constrained_layout=True)

# Set the main title for the entire figure
fig.suptitle('Partial Dependence Plots for Graduate School Aspiration (V2184) Across Different Models', 
             fontsize=16, fontweight='bold')

# Flatten the axes array for easier indexing
axes = axes.flatten()

# Store model names and corresponding data
models = [
    ('Random Forest', random_forest),
    ('Histogram Gradient Boost', hist_gradient_boost),
    ('XGBoost', xgboost),
    ('CatBoost', catboost),
    ('Partial Dependence of V2184', single_pdp)  # Adding the single PDP from the second image
]

# Plot each model's PDP in its own subplot
for i, (model_name, values) in enumerate(models):
    ax = axes[i]
    ax.plot(x_values, values, 'b-', linewidth=2)
    ax.set_title(f'PDP for V2184 - {model_name}', fontsize=12)
    ax.set_xlabel('Graduate School Aspiration')
    ax.set_ylabel('Partial dependence')
    
    # Set x-axis ticks at each value
    ax.set_xticks(x_values)
    ax.set_xticklabels(aspiration_labels, rotation=45)
    
    # Add grid for better readability
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Set y-axis limits to better show the patterns
    ax.set_ylim(0.545, 0.562)  # Common range for all plots based on the images

# Use the last subplot for a legend explaining the categories
ax_legend = axes[5]
ax_legend.axis('off')  # Hide axes

# Create a table with explanations
table_data = [[f"{val}", label] for val, label in zip(x_values, aspiration_labels)]
column_labels = ["Value", "Aspiration Level"]
table = ax_legend.table(
    cellText=table_data,
    colLabels=column_labels,
    loc='center',
    cellLoc='center',
    colWidths=[0.2, 0.6]  # Adjust column widths for better fit
)
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 1.5)

# Add title for the legend
ax_legend.text(0.5, 0.85, "V2184: Attend graduate or professional school after college",
         horizontalalignment='center',
         verticalalignment='center',
         transform=ax_legend.transAxes,
         fontsize=12,
         fontweight='bold')

# Note about interpretation
plt.figtext(0.8, 0.01, 
           "Note: All models show an inverted U-shape with peak at 'Probably Won't' (2),\n" +
           "indicating a non-linear relationship with the target variable.",
           ha='center', fontsize=10, fontstyle='italic')

# Save the figure with high resolution
plt.savefig('graduate_school_aspiration_pdp_grid.png', dpi=300, bbox_inches='tight')
plt.show()


# In[19]:


import matplotlib.pyplot as plt
import numpy as np

# X-axis values (binge drinking frequency categories)
x_values = np.array([1, 2, 3, 4, 5, 6])

# Category labels
binge_labels = [
    "None", 
    "Once", 
    "Twice", 
    "Three to five times", 
    "Six to nine times", 
    "Ten or more times"
]

# Y-axis values for each model (approximated from the images)
# Model 1: Random Forest (Image 1)
random_forest = np.array([0.553, 0.548, 0.529, 0.528, 0.527, 0.527])

# Model 2: Gradient Boost (Image 2)
gradient_boost = np.array([0.556, 0.555, 0.525, 0.522, 0.521, 0.502])

# Model 3: Histogram Gradient Boost (Image 3)
hist_gradient_boost = np.array([0.553, 0.557, 0.548, 0.555, 0.535, 0.553])

# Model 4: XGBoost (Image 4)
xgboost = np.array([0.553, 0.552, 0.537, 0.550, 0.551, 0.552])

# Model 5: CatBoost (Image 5)
catboost = np.array([0.554, 0.548, 0.539, 0.538, 0.525, 0.545])

# Create a figure with a 2x3 grid (6 subplots, one will be used for legend)
fig, axes = plt.subplots(2, 3, figsize=(18, 12), constrained_layout=True)

# Set the main title for the entire figure
fig.suptitle('Partial Dependence Plots for Binge Drinking (V2108) Across Different Models', 
             fontsize=16, fontweight='bold')

# Flatten the axes array for easier indexing
axes = axes.flatten()

# Store model names and corresponding data
models = [
    ('Random Forest', random_forest),
    ('Gradient Boost', gradient_boost),
    ('Histogram Gradient Boost', hist_gradient_boost),
    ('XGBoost', xgboost),
    ('CatBoost', catboost)
]

# Plot each model's PDP in its own subplot
for i, (model_name, values) in enumerate(models):
    ax = axes[i]
    ax.plot(x_values, values, 'b-', linewidth=2)
    ax.set_title(f'PDP for V2108 - {model_name}', fontsize=12)
    ax.set_ylabel('Partial dependence')
    
    # Set x-axis ticks at each value
    ax.set_xticks(x_values)
    ax.set_xticklabels([str(x) for x in x_values], rotation=0)
    
    # Add grid for better readability
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Set y-axis limits to better show the patterns
    # Use a common y-limit across all plots for fair comparison
    ax.set_ylim(0.500, 0.560)

# Use the last subplot for a legend explaining the categories
ax_legend = axes[5]
ax_legend.axis('off')  # Hide axes

# Create a table with binge drinking explanations
table_data = [[f"{val}", label] for val, label in zip(x_values, binge_labels)]
column_labels = ["Value", "Frequency"]
table = ax_legend.table(
    cellText=table_data,
    colLabels=column_labels,
    loc='center',
    cellLoc='center',
    colWidths=[0.2, 0.6]  # Adjust column widths for better fit
)
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 1.8)


# Note about interpretation
plt.figtext(0.9, 0.01, 
           'Note: 1="None" 2="Once" 3="Twice"\n' +
           '4="Three to five times" 5="Six to nine times" 6="Ten or more times"',
           ha='center', fontsize=10, fontstyle='italic')

# Save the figure with high resolution
plt.savefig('binge_drinking_pdp_grid.png', dpi=300, bbox_inches='tight')
plt.show()


# In[10]:


import matplotlib.pyplot as plt
import numpy as np

# X-axis values (fun evenings per week categories)
x_values = np.array([1, 2, 3, 4, 5, 6])

# Abbreviated category labels for x-axis
short_labels = [
    "< 1", 
    "1", 
    "2", 
    "3", 
    "4-5", 
    "6-7"
]

# Y-axis values for each model (approximated from the images)
# Model 1: Random Forest (Image 1)
random_forest = np.array([0.5495, 0.5482, 0.5520, 0.5542, 0.5540, 0.5527])

# Model 2: Gradient Boost (Image 2)
gradient_boost = np.array([0.5485, 0.5455, 0.5520, 0.5550, 0.5575, 0.5540])

# Model 3: Histogram Gradient Boost (Image 3)
hist_gradient_boost = np.array([0.5512, 0.5472, 0.5515, 0.5575, 0.5635, 0.5470])

# Model 4: CatBoost (Image 4)
catboost = np.array([0.5528, 0.5475, 0.5495, 0.5555, 0.5622, 0.5482])

# Create a figure with a 2x2 grid (4 subplots)
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Add space at the bottom for the legend
plt.subplots_adjust(bottom=0.2)

# Set the main title for the entire figure
fig.suptitle('Partial Dependence Plots for Fun Evenings Per Week (V2194)', 
             fontsize=14)

# Flatten the axes array for easier indexing
axes = axes.flatten()

# Store model names and corresponding data
models = [
    ('Random Forest', random_forest),
    ('Gradient Boost', gradient_boost),
    ('Histogram Gradient Boost', hist_gradient_boost),
    ('CatBoost', catboost)
]

# Plot each model's PDP in its own subplot
for i, (model_name, values) in enumerate(models):
    ax = axes[i]
    ax.plot(x_values, values, 'b-', linewidth=2)
    ax.set_title(model_name, fontsize=12)
    
    # Add y-label to the leftmost plots
    if i % 2 == 0:
        ax.set_ylabel('Partial dependence')
    
    # Set x-axis ticks and labels
    ax.set_xticks(x_values)
    ax.set_xticklabels(short_labels)
    
    # Add subtle grid for better readability
    ax.grid(True, linestyle=':', alpha=0.4)
    
    # Set y-axis limits to better show the patterns
    ax.set_ylim(0.545, 0.565)

# Add a single legend for the x-axis categories at the bottom of the figure
# This replaces the cluttered x-axis labels with a cleaner legend
legend_text = "Fun Evenings Per Week Values:   < 1 = Less than one,   1 = One,   2 = Two,   3 = Three,   4-5 = Four or Five,   6-7 = Six or Seven"

# Create a text box with the legend
fig.text(0.5, 0.08, legend_text, 
         ha='center', 
         fontsize=10,
         bbox=dict(facecolor='white', edgecolor='none', alpha=0.7, boxstyle='round,pad=0.5'))

# Save the figure with high resolution
plt.savefig('fun_evenings_pdp_grid_clean.png', dpi=300, bbox_inches='tight')
plt.show()


# In[23]:


import matplotlib.pyplot as plt
import numpy as np

# Define the data for each model based on the images
# X-axis values (self-rated school ability categories)
x = np.arange(1, 8)  # 1-7 representing the ability rating categories

# Y-axis values (partial dependence) estimated from the images
models_data = {
    'Random Forest': [0.5482, 0.5498, 0.5493, 0.5505, 0.5525, 0.5545, 0.5557],
    'Gradient Boost': [0.5425, 0.5485, 0.5495, 0.5510, 0.5515, 0.5555, 0.5580],
    'XGBoost': [0.5510, 0.5530, 0.5475, 0.5510, 0.5525, 0.5545, 0.5585],
    'CatBoost': [0.5515, 0.5530, 0.5498, 0.5510, 0.5525, 0.5535, 0.5600]
}

# Ability rating category labels
ability_labels = [
    "Far Below\nAverage", 
    "Below\nAverage", 
    "Slightly Below\nAverage",
    "Average", 
    "Slightly Above\nAverage", 
    "Above\nAverage", 
    "Far Above\nAverage"
]

# Create a 2x2 grid of subplots
fig, axs = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Partial Dependence of Self-Rated School Ability Across Different Models', fontsize=16)

# Get global min and max for consistent y-axis
all_values = [val for data in models_data.values() for val in data]
y_min = min(all_values) - 0.001
y_max = max(all_values) + 0.001

# Plot each model in the 2x2 grid
models = list(models_data.keys())
for i in range(2):
    for j in range(2):
        idx = i * 2 + j
        if idx < len(models):
            model_name = models[idx]
            axs[i, j].plot(x, models_data[model_name], 'o-', color='#1f77b4', linewidth=2)
            axs[i, j].set_title(model_name, fontsize=14)
            axs[i, j].set_xlabel('Self-Rated School Ability', fontsize=12)
            axs[i, j].set_ylabel('Partial dependence', fontsize=12)
            axs[i, j].set_xticks(x)
            axs[i, j].set_xticklabels(ability_labels, rotation=45, ha='right')
            axs[i, j].set_ylim(y_min, y_max)
            axs[i, j].grid(True, linestyle='--', alpha=0.7)

# Add a common explanation for all plots
fig.text(0.5, 0.01, 
         'Variable V2173: "Compared with others your age throughout the country, how do you rate yourself on school ability?"', 
         ha='center', fontsize=12)

# Adjust layout
plt.tight_layout()
plt.subplots_adjust(top=0.9, bottom=0.12)

# Also create a single plot with all models for comparison
fig2, ax = plt.subplots(figsize=(10, 6))
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']  # Different colors for each model
markers = ['o', 's', '^', 'D']  # Different markers for each model

for i, (model_name, data) in enumerate(models_data.items()):
    ax.plot(x, data, marker=markers[i], linestyle='-', linewidth=2, 
            color=colors[i], label=model_name)

ax.set_title('Comparison of Partial Dependence Plots Across Models', fontsize=14)
ax.set_xlabel('Self-Rated School Ability', fontsize=12)
ax.set_ylabel('Partial dependence', fontsize=12)
ax.set_xticks(x)
ax.set_xticklabels(ability_labels, rotation=45, ha='right')
ax.grid(True, linestyle='--', alpha=0.7)
ax.legend(loc='best')

ax.text(0.5, -0.15, 
        'Variable V2173: "Compared with others your age throughout the country, how do you rate yourself on school ability?"', 
        ha='center', transform=ax.transAxes, fontsize=11)

plt.tight_layout()

# Show plots
plt.show()


# In[37]:


import matplotlib.pyplot as plt
import numpy as np

# Define the data for each model based on the images
# X-axis values (work hours categories)
x = np.arange(1, 9)  # 1-8 representing the hours categories

# Y-axis values (partial dependence) estimated from the images
models_data = {
    'Random Forest': [0.557, 0.556, 0.553, 0.5545, 0.550, 0.547, 0.546, 0.546],
    'Gradient Boost': [0.557, 0.5575, 0.553, 0.555, 0.548, 0.546, 0.545, 0.5455],
    'Histogram Gradient Boost': [0.555, 0.562, 0.553, 0.557, 0.548, 0.546, 0.540, 0.553],
    'CatBoost': [0.553, 0.561, 0.551, 0.561, 0.548, 0.546, 0.544, 0.553]
}

# Hour categories labels
hour_labels = ["None", "≤5 hrs", "6-10 hrs", "11-15 hrs", "16-20 hrs", "21-25 hrs", "26-30 hrs", ">30 hrs"]

# Create a figure with two options
# Option 1: 2x2 grid
fig1, axs = plt.subplots(2, 2, figsize=(12, 10))
fig1.suptitle('Partial Dependence of Work Hours per Week Across Different Models', fontsize=16)

# Get global min and max for consistent y-axis
all_values = [val for data in models_data.values() for val in data]
y_min = min(all_values) - 0.005
y_max = max(all_values) + 0.005

# Plot each model in the 2x2 grid
models = list(models_data.keys())
for i in range(2):
    for j in range(2):
        idx = i * 2 + j
        if idx < len(models):
            model_name = models[idx]
            axs[i, j].plot(x, models_data[model_name], 'o-', color='#0000C0', linewidth=2)
            axs[i, j].set_title(model_name)
            axs[i, j].set_xlabel('Work Hours per Week')
            axs[i, j].set_ylabel('Partial dependence')
            axs[i, j].set_xticks(x)
            axs[i, j].set_xticklabels(hour_labels, rotation=45)
            axs[i, j].set_ylim(y_min, y_max)
            axs[i, j].grid(True, linestyle='--', alpha=0.7)

# Add a common explanation for all plots
fig1.text(0.5, 0.01, 
         'Variable V2191: "On average over the school year, how many hours per week do you work in a paid or unpaid job?"', 
         ha='center', fontsize=11)

# Adjust layout
plt.tight_layout()
plt.subplots_adjust(top=0.9, bottom=0.12)


# In[36]:


import matplotlib.pyplot as plt
import numpy as np

# Define the data for each model based on the images
# X-axis values (self-rated school ability categories)
x = np.arange(1, 8)  # 1-7 representing the ability rating categories

# Y-axis values (partial dependence) estimated from the images
models_data = {
    'Random Forest': [0.5482, 0.5498, 0.5493, 0.5505, 0.5525, 0.5545, 0.5557],
    'Gradient Boost': [0.5425, 0.5485, 0.5495, 0.5510, 0.5515, 0.5555, 0.5580],
    'XGBoost': [0.5510, 0.5530, 0.5475, 0.5510, 0.5525, 0.5545, 0.5585],
    'CatBoost': [0.5515, 0.5530, 0.5498, 0.5510, 0.5525, 0.5535, 0.5600]
}

# Ability rating category labels
ability_labels = [
    "Far Below\nAverage", 
    "Below\nAverage", 
    "Slightly Below\nAverage",
    "Average", 
    "Slightly Above\nAverage", 
    "Above\nAverage", 
    "Far Above\nAverage"
]

# Create a 2x2 grid of subplots
fig, axs = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Partial Dependence of Self-Rated School Ability Across Different Models', fontsize=16)

# Get global min and max for consistent y-axis
all_values = [val for data in models_data.values() for val in data]
y_min = min(all_values) - 0.001
y_max = max(all_values) + 0.001

# Plot each model in the 2x2 grid
models = list(models_data.keys())
for i in range(2):
    for j in range(2):
        idx = i * 2 + j
        if idx < len(models):
            model_name = models[idx]
            axs[i, j].plot(x, models_data[model_name], 'o-',color='#0000C0' , linewidth=2)
            axs[i, j].set_title(model_name, fontsize=14)
            axs[i, j].set_ylabel('Partial dependence', fontsize=12)
            axs[i, j].set_xticks(x)
            axs[i, j].set_xticklabels(ability_labels, rotation=45, ha='right')
            axs[i, j].set_ylim(y_min, y_max)
            axs[i, j].grid(True, linestyle='--', alpha=0.7)

# Add a common explanation for all plots
fig.text(0.5, 0.01, 
         'Variable V2173: "Compared with others your age throughout the country, how do you rate yourself on school ability?"', 
         ha='center', fontsize=12)

# Adjust layout
plt.tight_layout()
plt.subplots_adjust(top=0.9, bottom=0.12)


# In[35]:


import matplotlib.pyplot as plt
import numpy as np

# Define the data for each model based on the images
# X-axis values (dating frequency categories)
x = np.arange(1, 7)  # 1-6 representing the dating frequency categories

# Y-axis values (partial dependence) estimated from the images
models_data = {
    'Random Forest': [0.5534, 0.5508, 0.5510, 0.5515, 0.5512, 0.5483],
    'Gradient Boost': [0.5542, 0.5492, 0.5492, 0.5520, 0.5525, 0.5455],
    'XGBoost': [0.5538, 0.5472, 0.5490, 0.5560, 0.5538, 0.5423],
    'CatBoost': [0.5542, 0.5472, 0.5490, 0.5545, 0.5535, 0.5445]
}

# Dating frequency labels
dating_labels = [
    "Never", 
    "Once a month\nor less", 
    "2 or 3 times\na month",
    "Once a\nweek", 
    "2 or 3 times\na week", 
    "Over 3 times\na week"
]

# Create a 2x2 grid of subplots
fig, axs = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Partial Dependence of Dating Frequency Across Different Models', fontsize=16)

# Get global min and max for consistent y-axis
all_values = [val for data in models_data.values() for val in data]
y_min = min(all_values) - 0.001
y_max = max(all_values) + 0.001

# Plot each model in the 2x2 grid
models = list(models_data.keys())
for i in range(2):
    for j in range(2):
        idx = i * 2 + j
        if idx < len(models):
            model_name = models[idx]
            axs[i, j].plot(x, models_data[model_name], 'o-',color='#0000C0' , linewidth=2)
            axs[i, j].set_title(model_name, fontsize=14)
            axs[i, j].set_ylabel('Partial dependence', fontsize=12)
            axs[i, j].set_xticks(x)
            axs[i, j].set_xticklabels(dating_labels, rotation=45, ha='right')
            axs[i, j].set_ylim(y_min, y_max)
            axs[i, j].grid(True, linestyle='--', alpha=0.7)

# Add a common explanation for all plots
fig.text(0.5, 0.01, 
         'Variable V2195: "On the average, how often do you go out with a date (or your spouse/partner, if you are married)?"', 
         ha='center', fontsize=12)

# Adjust layout
plt.tight_layout()
plt.subplots_adjust(top=0.9, bottom=0.12)



# In[34]:


import matplotlib.pyplot as plt
import numpy as np

# Define the data for each model based on the images
# X-axis values (hometown environment categories)
x = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])  # 0-9 representing the hometown categories

# Y-axis values (partial dependence) estimated from the images
# Note: Some values might need adjustment based on clearer reading of the graphs
models_data = {
    'Random Forest': [0.5508, 0.5502, 0.5501, 0.5510, 0.5532, 0.5545, 0.5541, 0.5558, 0.5556, 0.5550],
    'Gradient Boost': [0.5568, 0.5495, 0.5472, 0.5470, 0.5550, 0.5560, 0.5565, 0.5590, 0.5595, 0.5575],
    'XGBoost': [0.5555, 0.5520, 0.5475, 0.5470, 0.5565, 0.5580, 0.5525, 0.5585, 0.5600, 0.5545],
    'CatBoost': [0.5565, 0.5525, 0.5490, 0.5465, 0.5565, 0.5585, 0.5505, 0.5590, 0.5615, 0.5525]
}

# Hometown environment category labels
hometown_labels = [
    "Can't say;\nmixed",
    "On a\nfarm", 
    "In the\ncountry", 
    "Small city/town\n(<50K)",
    "Medium city\n(50-100K)", 
    "Suburb of\nmedium city", 
    "Large city\n(100-500K)",
    "Suburb of\nlarge city", 
    "Very large city\n(>500K)", 
    "Suburb of\nvery large city"
]

# Create a 2x2 grid of subplots
fig, axs = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('Partial Dependence of Hometown Environment Across Different Models', fontsize=16)

# Get global min and max for consistent y-axis
all_values = [val for data in models_data.values() for val in data]
y_min = min(all_values) - 0.001
y_max = max(all_values) + 0.001

# Plot each model in the 2x2 grid
models = list(models_data.keys())
for i in range(2):
    for j in range(2):
        idx = i * 2 + j
        if idx < len(models):
            model_name = models[idx]
            axs[i, j].plot(x, models_data[model_name], 'o-', color='#0000C0', linewidth=2)
            axs[i, j].set_title(model_name, fontsize=14)
            axs[i, j].set_xlabel('Hometown Environment', fontsize=12)
            axs[i, j].set_ylabel('Partial dependence', fontsize=12)
            axs[i, j].set_xticks(x)
            axs[i, j].set_xticklabels(hometown_labels, rotation=45, ha='right', fontsize=10)
            axs[i, j].set_ylim(y_min, y_max)
            axs[i, j].grid(True, linestyle='--', alpha=0.7)

# Add a common explanation for all plots
fig.text(0.5, 0.01, 
         'Variable V2152: "Where did you grow up mostly?"', 
         ha='center', fontsize=12)

# Adjust layout
plt.tight_layout()
plt.subplots_adjust(top=0.9, bottom=0.12)



# In[44]:


import matplotlib.pyplot as plt
import numpy as np

# Define the data for each model based on the images
# X-axis values (skipping school categories)
x = np.arange(1, 8)  # 1-7 representing the skipping school categories

# Y-axis values (partial dependence) estimated from the images
models_data = {
    'Random Forest': [0.5515, 0.5551, 0.5555, 0.5554, 0.5548, 0.5570, 0.5565],
    'Gradient Boost': [0.5500, 0.5585, 0.5580, 0.5570, 0.5540, 0.5720, 0.5600],
    'XGBoost': [0.5500, 0.5610, 0.5570, 0.5530, 0.5450, 0.5695, 0.5525]
}

# Skipping school category labels
skipping_labels = [
    "None", 
    "1 Day", 
    "2 Days",
    "3 Days", 
    "4-5 Days", 
    "6-10 Days",
    "11 or More"
]

# Create a figure with 2x2 grid (with one empty subplot since we only have 3 models)
fig, axs = plt.subplots(2, 2, figsize=(10, 8))

# Flatten the axes array for easy iteration
axs_flat = axs.flatten()

# Set the style to match the reference image
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.linestyle'] = '-'
plt.rcParams['grid.alpha'] = 0.3
plt.rcParams['grid.color'] = '#E0E0E0'
plt.rcParams['axes.edgecolor'] = '#000000'
plt.rcParams['axes.linewidth'] = 0.8

# Get global min and max for consistent y-axis
all_values = [val for data in models_data.values() for val in data]
y_min = min(all_values) - 0.001
y_max = max(all_values) + 0.001

# Plot titles
titles = [
    'PDP for V2176 - Random Forest',
    'PDP for V2176 - Gradient Boost',
    'PDP for V2176 - XGBoost'
]

# Plot each model
for i, (model_name, values) in enumerate(models_data.items()):
    ax = axs_flat[i]
    
    # Set title
    ax.set_title(titles[i], fontsize=10)
    
    # Plot data with matching blue color
    ax.plot(x, values, '-o', color='#0000C0', linewidth=1.5, markersize=5)
    
    # Configure axes
    ax.set_ylabel('Partial dependence', fontsize=9)
    ax.set_xticks(x)
    ax.set_ylim(y_min, y_max)
    
    # Add grid matching the reference
    ax.grid(True, linestyle='-', alpha=0.3, color='#E0E0E0')
    
    # Set tick parameters
    ax.tick_params(axis='both', which='major', labelsize=8)

# Hide the fourth (empty) subplot
axs_flat[3].axis('off')

# Add a note at the bottom
fig.text(0.5, 0.0001, 
         'Note: V2176 represents skipping school categories (1="None" to 7="11 or More Days")', 
         ha='center', fontsize=9, style='italic')

# Adjust layout
plt.tight_layout()
plt.subplots_adjust(bottom=0.07, hspace=0.35, wspace=0.25)

# Show the plot
plt.show()


# In[45]:


import matplotlib.pyplot as plt
import numpy as np

# Define the data for each model based on the images
# X-axis values (siblings in household indicator)
x = np.array([0, 1])  # 0="No" 1="Yes"

# Y-axis values (partial dependence) estimated from the images
models_data = {
    'Gradient Boost': [0.5487, 0.5545],
    'Histogram Gradient Boost': [0.5485, 0.5550]
}

# Create a figure with 1x2 grid (since we only have 2 models)
fig, axs = plt.subplots(1, 2, figsize=(10, 4))

# Set the style to match the reference image
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.linestyle'] = '-'
plt.rcParams['grid.alpha'] = 0.3
plt.rcParams['grid.color'] = '#E0E0E0'
plt.rcParams['axes.edgecolor'] = '#000000'
plt.rcParams['axes.linewidth'] = 0.8

# Get global min and max for consistent y-axis
all_values = [val for data in models_data.values() for val in data]
y_min = min(all_values) - 0.001
y_max = max(all_values) + 0.001

# Plot titles
titles = [
    'PDP for V2157 - Gradient Boost',
    'PDP for V2157 - Histogram Gradient Boost'
]

# Plot each model
for i, (model_name, values) in enumerate(models_data.items()):
    ax = axs[i]
    
    # Set title
    ax.set_title(titles[i], fontsize=10)
    
    # Plot data with matching blue color
    ax.plot(x, values, '-', color='#0000C0', linewidth=1.5)
    
    # Configure axes
    ax.set_xlabel('V2157 (Siblings in Household)', fontsize=9)
    ax.set_ylabel('Partial dependence', fontsize=9)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['No', 'Yes'])
    ax.set_ylim(0.548, 0.556)  # Adjusted for better visualization
    
    # Add grid matching the reference
    ax.grid(True, linestyle='-', alpha=0.3, color='#E0E0E0')
    
    # Set tick parameters
    ax.tick_params(axis='both', which='major', labelsize=8)
    
    # Set x-axis range
    ax.set_xlim(-0.05, 1.05)

# Add a note at the bottom
fig.text(0.5, 0.01, 
         'Note: V2157 represents whether brother(s) and/or sister(s) live in the same household\n0="No" 1="Yes"', 
         ha='center', fontsize=9, style='italic')

# Adjust layout
plt.tight_layout()
plt.subplots_adjust(bottom=0.2)

# Show the plot
plt.show()


# In[49]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Define the x-values (categories of days missed)
days_missed = [1, 2, 3, 4, 5, 6, 7]
x_labels = ["None", "1 Day", "2 Days", "3 Days", "4-5 Days", "6-10 Days", "11+ Days"]

# Sample PDP values for Random Forest model (replace with your actual values)
# These would be the estimated target values for each category
rf_pdp_values = [0.2, 0.25, 0.3, 0.35, 0.45, 0.6, 0.7]

# Sample PDP values for XGBoost model (replace with your actual values)
xgb_pdp_values = [0.15, 0.22, 0.28, 0.38, 0.5, 0.65, 0.75]

# Create a figure with 1 row and 2 columns (for the two models)
fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

# Plot for Random Forest
axes[0].plot(days_missed, rf_pdp_values, marker='o', linestyle='-', color='#0000C0', linewidth=2)
axes[0].set_title('Random Forest PDP', fontsize=12)
axes[0].set_xlabel('Days Missed', fontsize=10)
axes[0].set_ylabel('Partial Dependence', fontsize=10)
axes[0].set_xticks(days_missed)
axes[0].set_xticklabels(x_labels, rotation=45)
axes[0].grid(True, linestyle='--', alpha=0.7)

# Plot for XGBoost
axes[1].plot(days_missed, xgb_pdp_values, marker='o', linestyle='-', color='#0000C0', linewidth=2)
axes[1].set_title('XGBoost PDP', fontsize=12)
axes[1].set_xlabel('Days Missed', fontsize=10)
axes[1].set_xticks(days_missed)
axes[1].set_xticklabels(x_labels, rotation=45)
axes[1].grid(True, linestyle='--', alpha=0.7)

# Add a horizontal line at y=0 for reference if needed
for ax in axes:
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)

# Adjust layout
plt.tight_layout()
plt.suptitle('Partial Dependence Plots: Effect of Illness-Related School Absences', fontsize=14, y=1.05)

# Save the figure
plt.savefig('illness_absence_pdp_comparison.png', dpi=300, bbox_inches='tight')

# Show the plot
plt.show()


# In[55]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Define the x-values (categories of days missed)
days_missed = [1, 2, 3, 4, 5, 6, 7]
x_labels = ["None", "1 Day", "2 Days", "3 Days", "4-5 Days", "6-10 Days", "11+ Days"]

# Sample PDP values for Random Forest model (replace with your actual values)
# These would be the estimated target values for each category
rf_pdp_values = [0.2, 0.25, 0.3, 0.35, 0.45, 0.6, 0.7]

# Sample PDP values for XGBoost model (replace with your actual values)
xgb_pdp_values = [0.15, 0.22, 0.28, 0.38, 0.5, 0.65, 0.75]

# Create a figure with 1 row and 2 columns (for the two models)
fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

# Plot for Random Forest
axes[0].plot(days_missed, rf_pdp_values, marker='o', linestyle='-', color='#0000C0', linewidth=2)
axes[0].set_title('Random Forest PDP', fontsize=12)
axes[0].set_xlabel('Days Missed', fontsize=10)
axes[0].set_ylabel('Partial Dependence', fontsize=10)
axes[0].set_xticks(days_missed)
axes[0].set_xticklabels(x_labels, rotation=45)
axes[0].grid(True, linestyle='--', alpha=0.7)

# Plot for XGBoost
axes[1].plot(days_missed, xgb_pdp_values, marker='o', linestyle='-', color='#0000C0', linewidth=2)
axes[1].set_title('XGBoost PDP', fontsize=12)
axes[1].set_xlabel('Days Missed', fontsize=10)
axes[1].set_xticks(days_missed)
axes[1].set_xticklabels(x_labels, rotation=45)
axes[1].grid(True, linestyle='--', alpha=0.7)

# Add a horizontal line at y=0 for reference if needed
for ax in axes:
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)

# Adjust layout
plt.tight_layout()
plt.suptitle('Partial Dependence Plots: Effect of Illness-Related School Absences', fontsize=14, y=1.05)

# Save the figure
plt.savefig('illness_absence_pdp_comparison.png', dpi=300, bbox_inches='tight')

# Show the plot
plt.show()


# In[62]:


import matplotlib.pyplot as plt
import numpy as np

# Define the x-values (money categories)
x_values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
money_amounts = ["None", "$1-5", "$6-10", "$11-20", "$21-35", "$36-50", 
                "$51-75", "$76-125", "$126-175", "$176+"]

# PDP values for XGBoost model (extracted from the image)
pdp_values = [0.5425, 0.5560, 0.5525, 0.5460, 0.5575, 0.5540, 0.5460, 0.5590, 0.5480, 0.5425]

# Create the figure
fig, ax = plt.subplots(figsize=(10, 7))

# Plot the PDP line
ax.plot(x_values, pdp_values, marker='o', linestyle='-', color='#0000C0', linewidth=2)

# Set main title
ax.set_title('Partial Dependence of Weekly Money from Other Sources (XGBoost)', fontsize=14)

# Add subtitle to explain the variable
subtitle = "During an average week, how much money do you get from other sources (allowances, etc.)?"
plt.figtext(0.5, 0.02, subtitle, ha='center', fontsize=11, style='italic')

# Set axis labels
ax.set_ylabel('Partial dependence', fontsize=12)

# Set x-axis ticks and labels
ax.set_xticks(x_values)
ax.set_xticklabels(money_amounts, rotation=45, ha='right')
ax.set_xlim(0.5, 10.5)

# Set y-axis limits to match the original plot
ax.set_ylim(0.5420, 0.5600)

# Add small vertical tick marks at the bottom like in the original plot
for tick in x_values:
    ax.axvline(x=tick, ymin=0, ymax=0.01, color='black', linewidth=1)

# Remove grid lines to match the original
ax.grid(False)

# Adjust layout with extra space at bottom for subtitle
plt.subplots_adjust(bottom=0.15)

# Save the figure
plt.savefig('money_source_pdp_labeled.png', dpi=300, bbox_inches='tight')

# Show the plot
plt.show()


# In[64]:


import matplotlib.pyplot as plt
import numpy as np

# Define the x-values (binary response)
x_values = [0, 1]
response_labels = ["No", "Yes"]

# Sample PDP values for Histogram Gradient Boost model
# These are simulated since actual values weren't provided
# Based on the linear trend in the V2188 plot you shared
pdp_values = [0.5485, 0.5545]  # Sample values showing positive effect of wanting college

# Create the figure
fig, ax = plt.subplots(figsize=(10, 7))

# Plot the PDP line
ax.plot(x_values, pdp_values, marker='o', linestyle='-', color='#0000C0', linewidth=2)

# Set main title
ax.set_title('Partial Dependence of Want 4-Year College (Histogram Gradient Boost)', fontsize=14)

# Add subtitle to explain the variable
subtitle = "Suppose you could do just what you'd like and nothing stood in your way.\nWould you WANT to graduate from college (four-year program)?"
plt.figtext(0.5, 0.02, subtitle, ha='center', fontsize=11, style='italic')

# Set axis labels
ax.set_xlabel('Response', fontsize=12)
ax.set_ylabel('Partial dependence', fontsize=12)

# Set x-axis ticks and labels
ax.set_xticks(x_values)
ax.set_xticklabels(response_labels)
ax.set_xlim(-0.2, 1.2)

# Set y-axis limits (adjusted to show the difference clearly)
ax.set_ylim(0.548, 0.555)

# Add small vertical tick marks at the bottom like in the original plots
for tick in x_values:
    ax.axvline(x=tick, ymin=0, ymax=0.01, color='black', linewidth=1)

# Remove grid lines to match the original plot style
ax.grid(False)

# Adjust layout with extra space at bottom for subtitle
plt.subplots_adjust(bottom=0.15)

# Save the figure
plt.savefig('want_college_pdp.png', dpi=300, bbox_inches='tight')

# Show the plot
plt.show()


# In[68]:


import matplotlib.pyplot as plt
import numpy as np

# Define the x-values (binary response)
x_values = [0, 1]
response_labels = ["No", "Yes"]

# Sample PDP values for Histogram Gradient Boost model
# Using a similar linear trend to the V2185 plot you shared
pdp_values = [0.551, 0.562]  # Sample values showing positive effect of wanting tech/vocational school

# Create the figure
fig, ax = plt.subplots(figsize=(10, 7))

# Plot the PDP line
ax.plot(x_values, pdp_values, marker='o', linestyle='-', color='#0000C0', linewidth=2)

# Set main title
ax.set_title('Partial Dependence of Interest in Technical/Vocational School (Histogram Gradient Boost)', fontsize=14)

# Add subtitle to explain the variable
subtitle = "Suppose you could do just what you'd like and nothing stood in your way.\nWould you want to attend a technical or vocational school?"
plt.figtext(0.5, 0.02, subtitle, ha='center', fontsize=11, style='italic')

# Set axis labels
ax.set_xlabel('Response', fontsize=12)
ax.set_ylabel('Partial dependence', fontsize=12)

# Set x-axis ticks and labels
ax.set_xticks(x_values)
ax.set_xticklabels(response_labels, fontsize=11)
ax.set_xlim(-0.05, 1.05)

# Set y-axis limits to match the style of the original plot
ax.set_ylim(0.551, 0.563)

# Add small vertical tick marks at the bottom like in the original plots
for tick in x_values:
    ax.axvline(x=tick, ymin=0, ymax=0.01, color='black', linewidth=1)

# Remove grid lines to match the original plot style
ax.grid(False)

# Adjust layout with extra space at bottom for subtitle
plt.subplots_adjust(bottom=0.15)

# Save the figure
plt.savefig('want_vocational_school_pdp_labeled.png', dpi=300, bbox_inches='tight')

# Show the plot
plt.show()


# In[70]:


import matplotlib.pyplot as plt
import numpy as np

# Define the x-values (likelihood categories)
x_values = [1, 2, 3, 4]
likelihood_labels = ["Definitely Won't", "Probably Won't", "Probably Will", "Definitely Will"]

# PDP values for Histogram Gradient Boost model (based on the image provided)
pdp_values = [0.5525, 0.5445, 0.5525, 0.5540]

# Create the figure
fig, ax = plt.subplots(figsize=(10, 7))

# Plot the PDP line
ax.plot(x_values, pdp_values, marker='o', linestyle='-', color='#0000C0', linewidth=2)

# Set main title
ax.set_title('Partial Dependence of College Graduation Likelihood (Histogram Gradient Boost)', fontsize=14)

# Add subtitle to explain the variable
subtitle = "How likely is it that you will graduate from college (four-year program) after high school?"
plt.figtext(0.5, 0.02, subtitle, ha='center', fontsize=11, style='italic')

# Set axis labels
ax.set_xlabel('Likelihood', fontsize=12)
ax.set_ylabel('Partial dependence', fontsize=12)

# Set x-axis ticks and labels
ax.set_xticks(x_values)
ax.set_xticklabels(likelihood_labels, rotation=10, ha='center', fontsize=10)
ax.set_xlim(0.8, 4.2)

# Set y-axis limits to match the style of the original plot
ax.set_ylim(0.544, 0.558)

# Add small vertical tick marks at the bottom like in the original plots
for tick in x_values:
    ax.axvline(x=tick, ymin=0, ymax=0.01, color='black', linewidth=1)

# Remove grid lines to match the original plot style
ax.grid(False)

# Adjust layout with extra space at bottom for subtitle
plt.subplots_adjust(bottom=0.15)

# Save the figure
plt.savefig('college_graduation_likelihood_pdp.png', dpi=300, bbox_inches='tight')

# Show the plot
plt.show()


# In[72]:


import matplotlib.pyplot as plt
import numpy as np

# Define the x-values (high school program types)
x_values = [1, 2, 3, 4]
program_labels = ["Academic/\nCollege Prep", "General", "Vocational/\nTechnical", "Other/\nDon't Know"]

# PDP values for Histogram Gradient Boost model (based on the image provided)
pdp_values = [0.553, 0.5555, 0.5555, 0.548]

# Create the figure
fig, ax = plt.subplots(figsize=(10, 7))

# Plot the PDP line
ax.plot(x_values, pdp_values, marker='o', linestyle='-', color='#0000C0', linewidth=2)

# Set main title
ax.set_title('Partial Dependence of High School Program Type (Histogram Gradient Boost)', fontsize=14)

# Add subtitle to explain the variable
subtitle = "Which of the following best describes your present high school program?"
plt.figtext(0.5, 0.02, subtitle, ha='center', fontsize=11, style='italic')

# Set axis labels
ax.set_xlabel('Program Type', fontsize=12)
ax.set_ylabel('Partial dependence', fontsize=12)

# Set x-axis ticks and labels
ax.set_xticks(x_values)
ax.set_xticklabels(program_labels, fontsize=10)
ax.set_xlim(0.8, 4.2)

# Set y-axis limits to match the style of the original plot
ax.set_ylim(0.546, 0.556)

# Add small vertical tick marks at the bottom like in the original plots
for tick in x_values:
    ax.axvline(x=tick, ymin=0, ymax=0.01, color='black', linewidth=1)

# Remove grid lines to match the original plot style
ax.grid(False)

# Adjust layout with extra space at bottom for subtitle
plt.subplots_adjust(bottom=0.15)

# Save the figure
plt.savefig('high_school_program_pdp.png', dpi=300, bbox_inches='tight')

# Show the plot
plt.show()


# In[74]:


import matplotlib.pyplot as plt
import numpy as np

# Define the x-values (amphetamine use frequency)
x_values = [1, 2, 3, 4, 5, 6, 7]
use_labels = ["0\nOccasions", "1-2\nOccasions", "3-5\nOccasions", "6-9\nOccasions", 
              "10-19\nOccasions", "20-39\nOccasions", "40+\nOccasions"]

# PDP values for XGBoost model (based on the image provided)
pdp_values = [0.552, 0.561, 0.561, 0.553, 0.5545, 0.552, 0.552]

# Create the figure
fig, ax = plt.subplots(figsize=(10, 7))

# Plot the PDP line
ax.plot(x_values, pdp_values, marker='o', linestyle='-', color='#0000C0', linewidth=2)

# Set main title
ax.set_title('Partial Dependence of Amphetamine Use Frequency (XGBoost)', fontsize=14)

# Add subtitle to explain the variable
subtitle = "On how many occasions (if any) have you taken amphetamines on your own\n--that is, without a doctor telling you to take them--during the last 12 months?"
plt.figtext(0.5, 0.02, subtitle, ha='center', fontsize=11, style='italic')

# Set axis labels
ax.set_xlabel('Frequency of Use', fontsize=12)
ax.set_ylabel('Partial dependence', fontsize=12)

# Set x-axis ticks and labels
ax.set_xticks(x_values)
ax.set_xticklabels(use_labels, fontsize=9)
ax.set_xlim(0.8, 7.2)

# Set y-axis limits to match the style of the original plot
ax.set_ylim(0.552, 0.562)

# Add small vertical tick marks at the bottom like in the original plots
for tick in x_values:
    ax.axvline(x=tick, ymin=0, ymax=0.01, color='black', linewidth=1)

# Remove grid lines to match the original plot style
ax.grid(False)

# Adjust layout with extra space at bottom for subtitle
plt.subplots_adjust(bottom=0.15)

# Save the figure
plt.savefig('amphetamine_use_pdp.png', dpi=300, bbox_inches='tight')

# Show the plot
plt.show()


# In[75]:


import matplotlib.pyplot as plt
import numpy as np

# Define the x-values (narcotic use frequency)
x_values = [1, 2, 3, 4, 5, 6, 7]
use_labels = ["0\nOccasions", "1-2\nOccasions", "3-5\nOccasions", "6-9\nOccasions", 
              "10-19\nOccasions", "20-39\nOccasions", "40+\nOccasions"]

# PDP values for XGBoost model (based on the image provided)
pdp_values = [0.5525, 0.5545, 0.5575, 0.5525, 0.5525, 0.5525, 0.5525]

# Create the figure
fig, ax = plt.subplots(figsize=(10, 7))

# Plot the PDP line
ax.plot(x_values, pdp_values, marker='o', linestyle='-', color='#0000C0', linewidth=2)

# Set main title
ax.set_title('Partial Dependence of Lifetime Narcotic Use Frequency (XGBoost)', fontsize=14)

# Add subtitle to explain the variable
subtitle = "On how many occasions (if any) have you taken narcotics other than heroin on your own\n--that is, without a doctor telling you to take them--in your lifetime?"
plt.figtext(0.5, 0.02, subtitle, ha='center', fontsize=11, style='italic')

# Set axis labels
ax.set_xlabel('Frequency of Use', fontsize=12)
ax.set_ylabel('Partial dependence', fontsize=12)

# Set x-axis ticks and labels
ax.set_xticks(x_values)
ax.set_xticklabels(use_labels, fontsize=9)
ax.set_xlim(0.8, 7.2)

# Set y-axis limits to match the style of the original plot
ax.set_ylim(0.544, 0.558)

# Add small vertical tick marks at the bottom like in the original plots
for tick in x_values:
    ax.axvline(x=tick, ymin=0, ymax=0.01, color='black', linewidth=1)

# Remove grid lines to match the original plot style
ax.grid(False)

# Adjust layout with extra space at bottom for subtitle
plt.subplots_adjust(bottom=0.15)

# Save the figure
plt.savefig('narcotic_use_pdp.png', dpi=300, bbox_inches='tight')

# Show the plot
plt.show()

