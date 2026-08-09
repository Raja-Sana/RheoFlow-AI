import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np

def plot_2d_rheology_curves(results: list):
    """
    Creates an interactive 2D Flow Curve chart (Shear Rate vs. Shear Stress) 
    comparing all samples.
    """
    fig = go.Figure()
    
    # Shear rates for 300 RPM (511 s^-1) and 600 RPM (1022 s^-1)
    shear_rates = [0, 511, 1022]
    
    for res in results:
        # Shear stress values at origin (YP), 300 RPM, and 600 RPM
        yp = res['YP (lb/100ft²)']
        t300 = res['300 RPM']
        t600 = res['600 RPM']
        
        shear_stresses = [yp, t300, t600]
        
        fig.add_trace(go.Scatter(
            x=shear_rates,
            y=shear_stresses,
            mode='lines+markers',
            name=res['Sample Name'],
            line=dict(width=3),
            marker=dict(size=8),
            hovertemplate="<b>%{text}</b><br>Shear Rate: %{x} s⁻¹<br>Shear Stress: %{y:.1f} lb/100ft²<extra></extra>",
            text=[res['Sample Name']]*3
        ))
        
    fig.update_layout(
        title="📈 2D Rheological Flow Curves (Shear Rate vs. Shear Stress)",
        xaxis_title="Shear Rate (s⁻¹)",
        yaxis_title="Shear Stress (lb/100 ft²)",
        template="plotly_dark",
        hovermode="x unified",
        margin=dict(l=40, r=40, t=60, b=40)
    )
    return fig


def plot_comparative_bar_chart(results: list):
    """
    Creates a grouped bar chart comparing key properties (PV, YP, TI) across samples.
    """
    df = pd.DataFrame(results)
    
    fig = go.Figure()
    fig.add_trace(go.Bar(x=df['Sample Name'], y=df['PV (cP)'], name='Plastic Viscosity (PV, cP)', marker_color='#38BDF8'))
    fig.add_trace(go.Bar(x=df['Sample Name'], y=df['YP (lb/100ft²)'], name='Yield Point (YP, lb/100ft²)', marker_color='#FACC15'))
    fig.add_trace(go.Bar(x=df['Sample Name'], y=df['TI (YP/PV)'], name='Transport Index (TI)', marker_color='#4ADE80'))
    
    fig.update_layout(
        title="📊 Comparative Properties Overview (PV, YP, TI)",
        barmode='group',
        template="plotly_dark",
        xaxis_title="Formulation Sample",
        yaxis_title="Value",
        margin=dict(l=40, r=40, t=60, b=40)
    )
    return fig


