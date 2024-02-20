import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import pandas as pd
import numpy as np

colors = px.colors.qualitative.Plotly

df = pd.read_csv('path_to_csv.csv')
df2 = pd.read_csv('path_to_csv2.csv')

fig = make_subplots(
    rows=2, cols=2,
    column_widths=[0.5, 0.5],
    row_heights=[0.8, 0.2],
    specs=[[{"type": "scattergeo"}, {"type": "bar"}],
           [{"type": "scatter", "colspan": 2}, None]]
)

for i in range(len(df)):
    fig.add_trace(
        go.Scattergeo(
            lon=[df['col_x1'][i], df['col_x2'][i]],
            lat=[df['col_y1'][i], df['col_y2'][i]],
            mode='lines',
            line=dict(
                width=float(df['col_y'][i]) / float(df['col_y'].max())*4,
                color='red'
            ),
            hoverinfo='none',
            opacity=0.8,
            showlegend=False
        )
    )

fig.add_trace(go.Scattergeo(
    lon=df2['x'],
    lat=df2['y'],
    text=df2['text'], 
    hoverinfo='text',
    mode='markers',
    marker=dict(
        size=df2['size'],
        color='black', 
        line=dict(
            color='white'
        )
    ),
))

fig.add_trace(
    go.Bar(x=df["col_x"][0:10], y=df["col_y"][0:10], marker=dict(color="red"), showlegend=False),
    row=1, col=2,
)

color_map = {z: colors[i % len(colors)] for i, z in enumerate(df['col_z'].unique())}
marker_colors = df['col_z'].map(color_map)  

fig.add_trace(
    go.Scatter(
        x=df["col_d"],
        y=df["col_y"],
        mode="markers",  
        marker=dict(color=marker_colors, size=10),  
        showlegend=False
    ),
    row=2, col=1
)

fig.update_geos(
    projection_type="orthographic",
    landcolor="gainsboro",
    oceancolor="dodgerblue",
    showocean=True,
    lakecolor="lightskyblue",
    showcountries=True,
)

fig.update_layout(
    template="plotly_dark",
    margin=dict(r=60, t=60, b=60, l=60),
)

fig.write_html('test.html', auto_open=True)
