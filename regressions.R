library(rhdf5)
library(Matrix)
library(data.table)

# data selection
START_DATE <- as.Date("2019-01-03") # 116th Congress start
END_DATE <- as.Date("2022-11-08") # 118th Congress election day

representatives <- fread("data/representatives.csv")
tweets <- fread("data/tweets.csv")

data <- tweets[representatives, on = "twitter"
][, tweet_id := as.character(tweet_id)
][, posted_at := as.Date(posted_at)
][posted_at >= START_DATE &
  posted_at < END_DATE &
  (posted_at < maps_proposed | posted_at > maps_finalized) &
  posted_at <= last_day_in_office &
  posted_at >= first_day_in_office
][, pvi_change := cook_pvi_new - cook_pvi_old
][, post := posted_at > maps_finalized
][, .(tweet_id, posted_at, bioguide, pvi_change, post, party, ran_for_reelection)]

# sae activations
ACTIVATIONS_FILE <- "sae/runs/rare-almanac-24/activations.h5"

ids <- as.character(h5read(ACTIVATIONS_FILE, "ids", bit64conversion = 'bit64'))
data <- h5read(ACTIVATIONS_FILE, "data")
indices <- h5read(ACTIVATIONS_FILE, "indices")
indptr  <- h5read(ACTIVATIONS_FILE, "indptr")
shape <- h5readAttributes(ACTIVATIONS_FILE, "/")$shape
N <- shape[1]
D = shape[2]

activations <- sparseMatrix(
    i = indices + 1,
    p = indptr,
    x = as.numeric(data),
    dims = shape
)
rownames(activations) <- ids

# regressions: TODO
