"""
Generate a figure describing the rejection region of our UMP LRT test with size alpha for the log harmonic mean of n probabilities.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gamma
from matplotlib.ticker import FixedLocator

# Set font globally for the entire plot
plt.rcParams['font.family'] = 'serif'  # Change to desired font family (e.g., 'serif', 'sans-serif')
plt.rcParams['font.size'] = 12  # Adjust font size globally

# Set fonts for LaTeX-like expressions (mathtext)
plt.rcParams['mathtext.fontset'] = 'custom'  # Use custom fontset
plt.rcParams['mathtext.rm'] = 'serif'        # Roman (normal text) to serif
plt.rcParams['mathtext.it'] = 'serif:italic' # Italic text to serif italic
plt.rcParams['mathtext.bf'] = 'serif:bold'   # Bold text to serif bold

# Parameters for the gamma distribution (shape k and scale theta)
# WARNING: ARBITRARY. NOT INFORMED BY REALITY. FOR ILLUSTRATION AND INSTRUCTIONAL PURPOSES ONLY.
k = 2  # shape parameter (k > 0)
theta = 1  # scale parameter (theta > 0)
alpha = 0.05  # significance level

# X is gamma distributed, we will plot for Y = -X, so x values should be negative
x = np.linspace(-6, 0, 1000)  # range of x values from -6 to 0 (negative)
X_pos = -x  # corresponding positive x values (since Y = -X)

# Critical value for the left-hand tail (since Y = -X, it's reflected from X critical value)
critical_value = -gamma.ppf(1-alpha, k, scale=theta)  # Critical value for Y = -X

# Plot the PDF of the gamma distribution for Y = -X
pdf = gamma.pdf(X_pos, k, scale=theta)

# Create the figure and axis
fig, ax = plt.subplots(figsize=(8, 6))

# Line for the PDF
ax.plot(x, pdf, label='Density/Mass', color='#2C3E50', lw=2)

# Shading the left-hand alpha region for the rejection region
x_fill = np.linspace(-6, critical_value, 100)
ax.fill_between(x_fill, 0, gamma.pdf(-x_fill, k, scale=theta), color='#E74C3C', alpha=0.6, label=r'Rejection Region')

# Critical value line
ax.axvline(critical_value, color='#C0392B', linestyle='--', lw=2, label=f'Critical Value')

# Annotations
ax.text(critical_value, -0.04, r'$h_\alpha$', color='#C0392B', fontsize=12, ha='center')

# Remove numerical x-axis and y-axis labels except at the origin; avoids deceiving readers with non-illustrative scale
def remove_non_origin_labels(axis):
    ticks = axis.get_ticklocs()  # Get tick locations
    labels = []
    for tick in ticks:
        if tick == 0:  # Keep label at origin
            labels.append('0')
        else:
            labels.append('')  # Empty string for non-origin ticks
    return labels

# Apply FixedLocator to ensure we have a fixed number of ticks
x_ticks = np.linspace(-6, 0, 7)  # Fixed x tick positions from -6 to 0
y_ticks = ax.get_yticks()  # Keep default y tick positions

# Set the ticks explicitly
ax.xaxis.set_major_locator(FixedLocator(x_ticks))
ax.yaxis.set_major_locator(FixedLocator(y_ticks))

# Set tick labels, keeping only the origin labeled
ax.set_xticklabels(remove_non_origin_labels(ax.xaxis))
ax.set_yticklabels(remove_non_origin_labels(ax.yaxis))

# Softened gridlines and axis
ax.grid(True, which='both', linestyle='--', linewidth=0.5, color='gray', alpha=0.7)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_color('#95A5A6')  # Light gray for spines
ax.spines['bottom'].set_color('#95A5A6')

# Title and Labels with larger font size for clarity
ax.set_title(r'Size $\alpha$ UMP LRT Rejection Region', fontsize=16, pad=20)
ax.set_xlabel(r'$h =$ Log Harmonic Mean of $n$ Probabilities', fontsize=14, labelpad=10)
ax.set_ylabel('Density/Mass of h', fontsize=14, labelpad=10)

# Adjust tick size for better readability
ax.tick_params(axis='both', which='major', labelsize=12)

# Adding the legend with a semi-transparent background
legend = ax.legend(loc='upper left', fontsize=12, frameon=True)
legend.get_frame().set_alpha(0.8)

# Ensure layout is clean and elements aren't cramped
plt.tight_layout()

# Show the plot
plt.show()
