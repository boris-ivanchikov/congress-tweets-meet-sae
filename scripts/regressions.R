library(rhdf5)
library(Matrix)
library(data.table)
library(fixest)
library(progress)
library(stats)
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

# loading sae activations
ACTIVATIONS_FILE <- "sae/runs/convenient-advertising-92/activations.h5"

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
MIN_PCT_ACTS <- 0.1
MAX_PCT_ACTS <- 10.0

chunk_size <- 100
num_chunks <- D %/% chunk_size + (D %% chunk_size > 0)

pb <- progress_bar$new(
  format = " Filtering Activations [:bar] :percent | ETA: :eta | Step :current/:total",
  total = num_chunks,
  clear = FALSE,
  width = 120
)

keep <- integer(0)
for (i in 1:num_chunks) {
    start <- (i - 1) * chunk_size + 1
    end <- min(i * chunk_size, D)
    chunk_idx <- start:end

    pos <- as.matrix(dataset_acts[, chunk_idx]) > 0
    num_acts <- colSums(pos)
    num_reps <- sapply(seq_along(chunk_idx), \(j) uniqueN(dataset$bioguide[pos[, j]]))

    keep <- c(keep, chunk_idx[
      num_reps > MIN_NUM_REPS
      & num_acts > N * MIN_PCT_ACTS / 100  
      & num_acts < N * MAX_PCT_ACTS / 100
    ])
    pb$tick()
}

dataset_acts <- dataset_acts[, keep]
print(paste0("Kept ", length(keep), " of ", D, " activations"))

# tweet-level regressions
n_keep <- ncol(dataset_acts)
result <- data.table(orig_idx = keep)

chunk_size <- 100
num_chunks <- n_keep %/% chunk_size + (n_keep %% chunk_size > 0)

pb <- progress_bar$new(
  format = " Fitting Regressions [:bar] :percent | ETA: :eta | Step :current/:total",
  total = num_chunks,
  clear = FALSE,
  width = 120
)

for (i in 1:num_chunks) {
    start <- (i - 1) * chunk_size + 1
    end <- min(i * chunk_size, n_keep)

    act_names <- colnames(dataset_acts)[start:end]
    acts_chunk <- as.matrix(dataset_acts[, start:end])
    colnames(acts_chunk) <- act_names

    dt <- cbind(as.data.table(acts_chunk), dataset)

    fml <- as.formula(paste0(
        "c(", paste(act_names, collapse = ","), ") ~ pvi_change:post | bioguide + posted_ym"
    ))

    models <- feols(fml, data = dt, cluster = ~bioguide)

    result[start:end, beta := sapply(models, \(m) coef(m)["pvi_change:post"])]
    result[start:end, pval := sapply(models, \(m) pvalue(m)["pvi_change:post"])]

    pb$tick()
}