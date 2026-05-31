library(rhdf5)
library(Matrix)
library(data.table)
library(fixest)
library(progress)
setFixest_nthreads(0)
setFixest_notes(FALSE)

# data selection
START_DATE <- as.Date("2019-01-03") # 116th Congress start
END_DATE <- as.Date("2022-11-08") # 118th Congress election day

representatives <- fread("data/representatives.csv")
tweets <- fread("data/tweets.csv")

dataset <- tweets[representatives, on = "twitter"
][, tweet_id := as.character(tweet_id)
][, posted_at := as.Date(posted_at)
][posted_at >= START_DATE &
  posted_at < END_DATE &
  (posted_at < maps_proposed | posted_at > maps_finalized) &
  posted_at <= last_day_in_office &
  posted_at >= first_day_in_office
][ran_for_reelection == 1
][, pvi_change := cook_pvi_new - cook_pvi_old
][, post := as.numeric(posted_at > maps_finalized)
][, posted_ym := format(posted_at, "%Y-%m")
][, .(tweet_id, posted_ym, bioguide, pvi_change, post, party, ran_for_reelection)]

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

# regressions
specification <- act_i ~ pvi_change:post | bioguide + posted_ym
dataset_acts <- activations[dataset[, tweet_id], ]
result <- data.table(idx = 1:D)

for (i in 1:D) {
    act_i <- dataset_acts[, i]
    if (var(act_i) == 0) next
    model <- feols(
        specification, 
        data = data.table(act_i, dataset),
        cluster = ~bioguide
    )
    result[i, beta := coef(model)["pvi_change:post"]]
    result[i, pval := pvalue(model)]

    pb$tick()
}