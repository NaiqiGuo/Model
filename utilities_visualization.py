from plotly import graph_objects as go
from plotly import io as pio

journal_template = go.layout.Template()
journal_template.layout.font = dict(family="Computer Modern, serif", size=16, color="black")
journal_template.layout.title.font = dict(size=20)
journal_template.layout.xaxis = dict(showgrid=True, gridcolor='lightgray', linecolor='black', showline=True, mirror=True, ticks='outside', ticklen=5)
journal_template.layout.yaxis = dict(showgrid=True, gridcolor='lightgray', linecolor='black', showline=True, mirror=True, ticks='outside', ticklen=5)
journal_template.layout.margin = dict(l=90, r=20, t=50, b=90)
journal_template.layout.legend.font = dict(size=16)
pio.templates['journal_theme'] = journal_template
pio.templates.default = "journal_theme"

import matplotlib.pyplot as plt
try:
    import scienceplots
    plt.style.use('science')
except:
    pass
plt.rcParams.update({'font.size': 16,       # Default font size for general text
                    'axes.titlesize': 16,   # Font size for plot titles
                    'axes.labelsize': 16,   # Font size for axis labels
                    'xtick.labelsize': 14,  # Font size for x-axis tick labels
                    'ytick.labelsize': 14,  # Font size for y-axis tick labels
                    'legend.fontsize': 14,  # Font size for legend
                    'xtick.minor.visible': False, # Minor x ticks
                    'ytick.minor.visible': False, # Minor y ticks
                    })
