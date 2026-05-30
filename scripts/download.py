import os
import tempfile
import requests
import pandas as pd
from nomic import atlas
import pyreadr

if __name__ == "__main__":

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(root)

    if not os.path.exists("data"):
        os.mkdir("data")

    # Tweets from Members of US Congress from All Time (updated 2024-12-05): https://atlas.nomic.ai/data/hivemind/tweets-from-members-of-us-congress-from-all-time-updated-2024-12-05-3 

    if os.path.exists("data/tweets.csv"):
        print("data/tweets.csv already exists, skipping.")
        tweets_df = pd.read_csv("data/tweets.csv")
    else:
        dataset = atlas.AtlasDataset(identifier="hivemind/tweets-from-members-of-us-congress-from-all-time-updated-2024-12-05-3")
        tweets_df = dataset.maps[0].data.df
        tweets_df = tweets_df.rename({"tweetId": "tweet_id", "postedAt": "posted_at", "twitter_lower": "twitter"}, axis=1)
        tweets_df["name"] = tweets_df["name"].str.replace(r"([A-Z]-[A-Z][A-Z])", "", regex=True)
        tweets_df = tweets_df.drop(["state", "party", "chamber", "years", "source", "id"], axis=1)
        tweets_df = tweets_df[tweets_df["text"] != "null"]

        tweets_df.to_csv("data/tweets.csv", index=False)
        print(f"Saved data/tweets.csv with {tweets_df.shape[0]} rows, columns: {list(tweets_df.columns)}")

    # Kaslovsky, Jaclyn and Michael R. Kistner 2025: https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/CCVUFY

    if os.path.exists("data/representatives.csv"):
        print("data/representatives.csv already exists, skipping.")
    else:
        # cook pvi
        r = requests.get("https://dataverse.harvard.edu/api/access/datafile/10407850")
        with tempfile.NamedTemporaryFile(suffix=".rda") as f:
            f.write(r.content)
            cook_pvi_df = pyreadr.read_r(f.name)["prepost_df"]
        cook_pvi_df["district_old"] = cook_pvi_df.apply(lambda row: row["State"] + "-" + str(int(row["District_116"])).zfill(2), axis=1)
        cook_pvi_df["district_new"] = cook_pvi_df.apply(lambda row: row["State"] + "-" + str(int(row["District_117"])).zfill(2), axis=1)
        cook_pvi_df[["cook_pvi_old", "cook_pvi_new", "ran_for_reelection"]] = cook_pvi_df[["CookPVI_116", "CookPVI_117", "RanForReelection"]].astype(int)
        cook_pvi_df = cook_pvi_df.rename({"Party": "party", "State": "state"}, axis=1)
        cook_pvi_df = cook_pvi_df[["state", "party", "district_old", "district_new", "cook_pvi_old", "cook_pvi_new", "ran_for_reelection"]]

        # info
        COLUMNS = [
            'name.first',
            'name.last',
            'name.middle',
            'name.suffix',
            'name.official_full',
            'term.type',
            'term.start',
            'term.end',
            'term.state',
            'term.district',
            'id.bioguide',
        ]

        legislators_df = pd.concat([
            pd.read_json("https://unitedstates.github.io/congress-legislators/legislators-historical.json"),
            pd.read_json("https://unitedstates.github.io/congress-legislators/legislators-current.json")
        ])
        legislators_df = legislators_df.explode("terms").rename({"terms": "term"}, axis=1).reset_index(drop=True).to_dict(orient="records")
        legislators_df = pd.json_normalize(legislators_df)
        legislators_df = legislators_df[COLUMNS]
        legislators_df = legislators_df.rename({'id.bioguide': 'bioguide'}, axis=1)
        legislators_df["term.start"] = pd.to_datetime(legislators_df["term.start"])
        legislators_df["term.end"] = pd.to_datetime(legislators_df["term.end"])

        legislators_df = legislators_df[legislators_df["term.type"] == "rep"]
        legislators_df = legislators_df.drop("term.type", axis=1)

        name_construct = (
            legislators_df[["name.first", "name.middle", "name.last", "name.suffix"]]
            .fillna("")
            .agg(lambda x: " ".join(part for part in x if part), axis=1)
        )

        legislators_df["name"] = (
            legislators_df["name.official_full"]
            .replace("", pd.NA)
            .fillna(name_construct)
        )
        legislators_df = legislators_df.drop(["name.first", "name.middle", "name.last", "name.suffix", "name.official_full"], axis=1)

        g = legislators_df.groupby("bioguide")
        legislators_df["first_day_in_office"] = g["term.start"].transform("min")
        legislators_df["last_day_in_office"] = g["term.end"].transform("max")
        served_in_117 = set(legislators_df[legislators_df["term.start"].dt.year == 2021]["bioguide"])
        served_in_116 = set(legislators_df[legislators_df["term.start"].dt.year.isin([2019, 2020])]["bioguide"])
        legislators_df = legislators_df[legislators_df["bioguide"].isin(served_in_117 & served_in_116) & (legislators_df["term.end"].dt.year == 2021)]
        legislators_df = legislators_df.drop(["term.start", "term.end"], axis=1)

        legislators_df["term.district"] = legislators_df["term.district"].astype(int).astype(str).str.zfill(2).str.replace("00", "01")
        legislators_df["district_old"] = legislators_df["term.state"] + "-" + legislators_df["term.district"]
        legislators_df = legislators_df.drop(["term.state", "term.district"], axis=1)

        # dates: https://en.wikipedia.org/wiki/2020_United_States_redistricting_cycle
        data = [
            ("CA", "2021-12-27", "2021-12-27"),
            ("TX", "2021-10-25", "2021-10-25"),
            ("FL", "2022-04-22", "2022-04-22"),
            ("NY", "2022-02-03", "2022-03-31"),   # Overturned on March 31, 2022
            ("PA", "2022-02-23", "2022-02-23"),
            ("IL", "2021-11-23", "2021-11-23"),
            ("OH", "2021-11-20", "2022-01-14"),   # Overturned by state Supreme Court on Jan 14, 2022
            ("GA", "2021-12-30", "2021-12-30"),
            ("NC", "2022-02-23", "2022-02-23"),
            ("MI", "2021-12-28", "2021-12-28"),
            ("NJ", "2021-12-22", "2021-12-22"),
            ("VA", "2021-12-28", "2021-12-28"),
            ("WA", "2022-02-08", "2022-02-08"),
            ("AZ", "2021-12-22", "2021-12-22"),
            ("MA", "2021-11-22", "2021-11-22"),
            ("TN", "2022-02-06", "2022-02-06"),
            ("IN", "2021-10-04", "2021-10-04"),
            ("MD", "2022-04-04", "2022-04-04"),
            ("MO", "2022-05-18", "2022-05-18"),
            ("WI", "2022-03-03", "2022-03-03"),
            ("CO", "2021-11-01", "2021-11-01"),
            ("MN", "2022-02-15", "2022-02-15"),
            ("SC", "2022-01-26", "2022-01-26"),
            ("AL", "2021-11-04", "2021-11-04"),
            ("LA", "2022-03-30", "2022-03-30"),
            ("KY", "2022-01-20", "2022-01-20"),
            ("OR", "2021-09-27", "2021-09-27"),
            ("OK", "2021-11-22", "2021-11-22"),
            ("CT", "2022-02-10", "2022-02-10"),
            ("UT", "2021-11-12", "2021-11-12"),
            ("IA", "2021-11-04", "2021-11-04"),
            ("NV", "2021-11-16", "2021-11-16"),
            ("AR", "2021-01-14", "2021-01-14"),
            ("MS", "2022-01-25", "2022-01-25"),
            ("KS", "2022-02-09", "2022-02-09"),
            ("NM", "2021-12-17", "2021-12-17"),
            ("NE", "2021-09-30", "2021-09-30"),
            ("ID", "2021-11-05", "2021-11-05"),
            ("WV", "2021-10-22", "2021-10-22"),
            ("HI", "2022-01-28", "2022-01-28"),
            ("NH", "2022-05-31", "2022-05-31"),
            ("ME", "2021-09-29", "2021-09-29"),
            ("RI", "2022-02-18", "2022-02-18"),
            ("MT", "2021-11-12", "2021-11-12"),
        ]
        
        dates_df = pd.DataFrame(data, columns=["state", "maps_proposed", "maps_finalized"])
        dates_df["maps_proposed"] = pd.to_datetime(dates_df["maps_proposed"])
        dates_df["maps_finalized"] = pd.to_datetime(dates_df["maps_finalized"])

        # social media
        social_media_df_1 = pd.json_normalize(pd.read_json("https://unitedstates.github.io/congress-legislators/legislators-social-media.json").to_dict(orient="records"))
        social_media_df_1["twitter"] = social_media_df_1["social.twitter"].str.lower()

        social_media_df_2 = pd.read_json("https://raw.githubusercontent.com/alexlitel/congresstweets-automator/refs/heads/master/data/historical-users-filtered.json")
        social_media_df_2 = social_media_df_2.explode("accounts").rename({"accounts": "account"}, axis=1).reset_index(drop=True).to_dict(orient="records")
        social_media_df_2 = pd.json_normalize(social_media_df_2)
        social_media_df_2["twitter"] = social_media_df_2["account.screen_name"].str.lower()

        social_media_df = pd.concat([social_media_df_1[["id.bioguide", "twitter"]], social_media_df_2[["id.bioguide", "twitter"]]]).dropna().drop_duplicates()

        tweets_df = pd.read_csv("data/tweets.csv")
        social_media_df = social_media_df[social_media_df["twitter"].isin(tweets_df["twitter"])]
        social_media_df = social_media_df.rename({"id.bioguide": "bioguide", "twitter": "twitter"}, axis=1)

        # join
        final_df = cook_pvi_df \
            .merge(legislators_df, on="district_old", how="left") \
            .merge(dates_df, on="state", how="left") \
            .merge(social_media_df, on="bioguide", how="left")
    
        final_df = final_df[
            final_df["maps_proposed"].isna()
            | (
                (final_df["maps_proposed"] >= final_df["first_day_in_office"])
                & (final_df["maps_finalized"] <= final_df["last_day_in_office"])
            )
        ]
        final_df = final_df.dropna(subset=["twitter"])
        
        final_df = final_df[[
            "bioguide",  
            "name", 
            "twitter", 
            "party", 
            "state", 
            "district_old", 
            "district_new", 
            "cook_pvi_old", 
            "cook_pvi_new", 
            "ran_for_reelection",
            "first_day_in_office",
            "last_day_in_office",
            "maps_proposed",
            "maps_finalized"
        ]]
        final_df = final_df.sort_values(by="district_old")

        final_df.to_csv("data/representatives.csv", index=False)
        print(f"Saved data/representatives.csv with {final_df.shape[0]} rows, columns: {list(final_df.columns)}")
        
    print("Done.")