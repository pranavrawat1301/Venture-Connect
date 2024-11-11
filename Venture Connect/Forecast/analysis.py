from analyser_using_var import StartupAnalyzer
import pandas as pd

custom_data = pd.read_csv("D:\Projects(internship)\Venture Connect\Forecast\startup_data_generated.csv")
custom_data['Date'] = pd.to_datetime(custom_data['Date'])
def analyze_startup(data=None):

    if data is None:
        print("No data provided - using synthetic data generator")
        analyzer = StartupAnalyzer()
    else:
        print("Using provided custom data")
        analyzer = StartupAnalyzer(data)
    
    metrics = analyzer.calculate_metrics()
    print("\nCurrent Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.2f}")
    
    print("\nForecasts for next 8 quarters:")
    forecast = analyzer.forecast_var()
    for metric in ['Revenue', 'Profit', 'Valuation']:
        print(f"\n{metric} Forecast (End of Period):")
        print(f"Expected: ${forecast[metric].iloc[-1]:,.2f}")
        print(f"Range: ${forecast[f'{metric}_lower'].iloc[-1]:,.2f} to ${forecast[f'{metric}_upper'].iloc[-1]:,.2f}")
    
    score = analyzer.calculate_score()
    print(f"\nOverall Startup Score (1-10): {score}")
    
    improvements = analyzer.get_improvement_areas()
    print("\nAreas for Improvement:")
    for imp in improvements:
        print(f"\n{imp['area']} (Severity: {imp['severity']}):")
        print(f"Suggestion: {imp['suggestion']}")
        print("Specific Actions:")
        for action in imp['specific_actions']:
            print(f"- {action}")
    
    fig = analyzer.create_visualizations()
    fig.show()


if __name__ == "__main__":
    #put data = custor_data if want to use custom data
    analyze_startup(data = custom_data)
