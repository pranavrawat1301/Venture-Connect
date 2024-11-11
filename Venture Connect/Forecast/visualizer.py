from plotly.subplots import make_subplots
import plotly.graph_objects as go

class StartupVisualizer:
    def __init__(self, analyzer):
        self.analyzer = analyzer
        self.df = analyzer.df
        
    def create_visualizations(self):
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=(
                'Revenue & Profit Trends', 
                'Growth Metrics',
                'Unit Economics', 
                'Operational Metrics',
                'VAR Forecasts', 
                'Key Performance Indicators'
            ),
            specs=[
                [{"secondary_y": True}, {"secondary_y": True}],
                [{"secondary_y": True}, {"secondary_y": True}],
                [{"secondary_y": True}, {"type": "domain"}]
            ]
        )

        # 1. Revenue & Profit Trends
        fig.add_trace(
            go.Scatter(
                x=self.df['Date'], 
                y=self.df['Revenue'],
                name='Revenue',
                line=dict(color='blue')
            ),
            row=1, col=1, secondary_y=False
        )
        fig.add_trace(
            go.Scatter(
                x=self.df['Date'], 
                y=self.df['Profit'],
                name='Profit',
                line=dict(color='green')
            ),
            row=1, col=1, secondary_y=True
        )

        # 2. Growth Metrics
        fig.add_trace(
            go.Scatter(
                x=self.df['Date'],
                y=self.df['Customer_Count'],
                name='Customer Count',
                line=dict(color='orange')
            ),
            row=1, col=2
        )
        
        # Calculate and plot growth rates
        revenue_growth = self.df['Revenue'].pct_change() * 100
        fig.add_trace(
            go.Scatter(
                x=self.df['Date'],
                y=revenue_growth,
                name='Revenue Growth %',
                line=dict(color='red', dash='dot')
            ),
            row=1, col=2, secondary_y=True
        )

        # 3. Unit Economics
        fig.add_trace(
            go.Scatter(
                x=self.df['Date'],
                y=self.df['Customer_Lifetime_Value'],
                name='LTV',
                line=dict(color='purple')
            ),
            row=2, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=self.df['Date'],
                y=self.df['Customer_Acquisition_Cost'],
                name='CAC',
                line=dict(color='brown')
            ),
            row=2, col=1, secondary_y=True
        )

        # 4. Operational Metrics
        fig.add_trace(
            go.Scatter(
                x=self.df['Date'],
                y=self.df['Revenue_Per_Employee'],
                name='Revenue/Employee',
                line=dict(color='cyan')
            ),
            row=2, col=2
        )
        fig.add_trace(
            go.Scatter(
                x=self.df['Date'],
                y=self.df['Cash_Burn'],
                name='Cash Burn',
                line=dict(color='red')
            ),
            row=2, col=2, secondary_y=True
        )

        forecast_df = self.analyzer.forecast_var()
        
        for metric in ['Revenue', 'Profit', 'Valuation']:
            # Historical data
            fig.add_trace(
                go.Scatter(
                    x=self.df['Date'],
                    y=self.df[metric],
                    name=f'Historical {metric}',
                    line=dict(color='blue' if metric == 'Revenue' else 'green' if metric == 'Profit' else 'gold')
                ),
                row=3, col=1
            )
            
            # Forecast data
            fig.add_trace(
                go.Scatter(
                    x=forecast_df['Date'],
                    y=forecast_df[metric],
                    name=f'Forecast {metric}',
                    line=dict(dash='dash', color='blue' if metric == 'Revenue' else 'green' if metric == 'Profit' else 'gold')
                ),
                row=3, col=1
            )
            
            # Confidence intervals
            fig.add_trace(
                go.Scatter(
                    x=forecast_df['Date'].tolist() + forecast_df['Date'].tolist()[::-1],
                    y=forecast_df[f'{metric}_upper'].tolist() + forecast_df[f'{metric}_lower'].tolist()[::-1],
                    fill='toself',
                    fillcolor=f'rgba{(0,0,255,0.1) if metric=="Revenue" else (0,255,0,0.1) if metric=="Profit" else (255,215,0,0.1)}',
                    line=dict(color='rgba(255,255,255,0)'),
                    name=f'{metric} Confidence Interval'
                ),
                row=3, col=1
            )

        # 6. Key Performance Indicators (Gauge chart)
        score = self.analyzer.calculate_score()
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=score,
                title={'text': "Overall Score"},
                gauge={
                    'axis': {'range': [1, 10]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [1, 4], 'color': "red"},
                        {'range': [4, 7], 'color': "yellow"},
                        {'range': [7, 10], 'color': "green"}
                    ],
                    'threshold': {
                        'line': {'color': "black", 'width': 4},
                        'thickness': 0.75,
                        'value': score
                    }
                }
            ),
            row=3, col=2
        )

        fig.update_layout(
            height=1200,
            width=1000,
            title_text="Startup Analytics Dashboard",
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )

        fig.update_xaxes(title_text="Date", row=1, col=1)
        fig.update_xaxes(title_text="Date", row=1, col=2)
        fig.update_xaxes(title_text="Date", row=2, col=1)
        fig.update_xaxes(title_text="Date", row=2, col=2)
        fig.update_xaxes(title_text="Date", row=3, col=1)

        fig.update_yaxes(title_text="Revenue ($)", row=1, col=1, secondary_y=False)
        fig.update_yaxes(title_text="Profit ($)", row=1, col=1, secondary_y=True)
        fig.update_yaxes(title_text="Customer Count", row=1, col=2, secondary_y=False)
        fig.update_yaxes(title_text="Growth Rate (%)", row=1, col=2, secondary_y=True)
        fig.update_yaxes(title_text="LTV ($)", row=2, col=1, secondary_y=False)
        fig.update_yaxes(title_text="CAC ($)", row=2, col=1, secondary_y=True)
        fig.update_yaxes(title_text="Revenue/Employee ($)", row=2, col=2, secondary_y=False)
        fig.update_yaxes(title_text="Cash Burn ($)", row=2, col=2, secondary_y=True)
        fig.update_yaxes(title_text="Value ($)", row=3, col=1)

        return fig