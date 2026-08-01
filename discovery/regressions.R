library(rhdf5)
library(Matrix)
library(data.table)
library(fixest)
library(progress)
library(stats)
setFixest_nthreads(0)
setFixest_notes(FALSE)

PATH <- "sae/runs/cynical-credit-37"

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
][, is_republican := as.numeric(party == "R")
][, pvi_change_relative := fifelse(party == "R", pvi_change, -pvi_change)
][, post := as.numeric(posted_at > maps_finalized)
][, posted_ym := format(posted_at, "%Y-%m")
][, .(tweet_id, posted_ym, bioguide, is_republican, pvi_change_relative, post, party, ran_for_reelection)]

# loading sae activations
ACTIVATIONS_FILE <- file.path(PATH, "activations.h5")

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
colnames(activations) <- paste0("act_", 1:D)

dataset_acts <- activations[dataset[, tweet_id], ]

# filtering activations
MIN_NUM_REPS <- 50
MIN_PCT_ACTS <- 0.01

dataset[, tweet_idx := .I]
dataset[, rep_idx := .GRP, by = bioguide]
G = sparseMatrix(
    i = dataset[, rep_idx],
    j = dataset[, tweet_idx],
    x = 1
)
rep_acts <- G %*% (dataset_acts > 0)
keep <- (colSums(dataset_acts > 0) >= nrow(dataset_acts) * MIN_PCT_ACTS) & (colSums(rep_acts > 0) >= MIN_NUM_REPS)
keep_names <- colnames(activations)[keep]
dataset_acts <- dataset_acts[, keep]
print(paste0("Kept ", sum(keep), " of ", D, " activations"))

# aggregating by representative-month
dataset[, rep_ym_idx := .GRP, by = .(bioguide, posted_ym, post)]

G = sparseMatrix(
    i = dataset[, rep_ym_idx],
    j = dataset[, tweet_idx],
    x = 1
)
grouped_acts = G %*% dataset_acts / rowSums(G)

grouped_dataset <- dataset[, .(
    n_tweets = .N,
    party = party[1],
    ran_for_reelection = ran_for_reelection[1],
    pvi_change_relative = pvi_change_relative[1],
    is_republican = is_republican[1]
), by = .(bioguide, posted_ym, post)]

grouped_dataset <- cbind(grouped_dataset, as.data.table(as.matrix(grouped_acts)))

# running regressions
n_keep <- sum(keep)
result <- data.table(orig_idx = keep_names)

fml <- as.formula(paste0(
    "c(", paste(keep_names, collapse = ","), ") ~ post + post:pvi_change_relative + post:is_republican | bioguide + posted_ym"
))
models <- feols(
    fml, 
    data = grouped_dataset, 
    weights = ~n_tweets, 
    cluster = ~bioguide
)

result[, `:=`(
    beta      = sapply(models, \(m) coef(m)["post:pvi_change_relative"]),
    se        = sapply(models, \(m) se(m)["post:pvi_change_relative"]),
    pval      = sapply(models, \(m) pvalue(m)["post:pvi_change_relative"]),
    beta_post = sapply(models, \(m) coef(m)["post"]),
    se_post   = sapply(models, \(m) se(m)["post"]),
    pval_post = sapply(models, \(m) pvalue(m)["post"]),
    beta_rep  = sapply(models, \(m) coef(m)["post:is_republican"]),
    se_rep    = sapply(models, \(m) se(m)["post:is_republican"]),
    pval_rep  = sapply(models, \(m) pvalue(m)["post:is_republican"]),
    nobs      = sapply(models, nobs)
)]

fwrite(result[order(pval)], file.path(PATH, "regressions.csv"))
