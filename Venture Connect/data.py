from data_synthesis import DataSynthesizer

data = DataSynthesizer()

startups_df = data.generate_startup_data(num_startups=200)
investor_df = data.generate_investor_data(num_investors=50)

data.save_data(startups_df , investor_df)

