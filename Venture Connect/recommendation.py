from data import startups_df , investor_df
from recommendor import StartupInvestorRecommender

recommender = StartupInvestorRecommender(startups_df , investor_df)
n = str(input("Are you an Investor or a Startup : "))

if n == "Investor":
    id = int(input("Enter you ID : "))
    investor_details = investor_df.loc[investor_df['id'] == id, ['id', 'name', 'preferred_industry']]
    print(f"{investor_details['id'].values[0]} | {investor_details['name'].values[0]} | {investor_details['preferred_industry'].values[0]}")
    matches = recommender.get_top_matches_for_investor(id , top_n=10)
elif n == "Startup":
    id = int(input("Enter you ID : "))
    startup_details = startups_df.loc[startups_df['id'] == id, ['id', 'name', 'industry']]
    print(f"{startup_details['id'].values[0]} | {startup_details['name'].values[0]} | {startup_details['industry'].values[0]}")
    matches = recommender.get_top_matches_for_startup(id , top_n=10)
else:
    print("Invalid Choice")

print(matches)

