#This script creates the mclust data for Figure 4 in the pyclust paper
#Note that the results may be saved as minutes - in which case you should convert the resulting csvs into seconds


library("mclust")

base <- 40
factor <- 2
num_sets <- 13 #indicates the maximally sized dataset (15 in the paper)
exps <- 0:num_sets #************************8
ns <- base * factor ^ exps
output_file <- 'mclust_option_times.csv' #******************************8
results <- data.frame(matrix(ncol = 3, nrow = 0))
cols <- c('N','Model','Time')
colnames(results) <- cols


x <- c("N","Time")
colnames(results) <- x

modelNames=mclust.options("emModelNames")
ks <- 2:6
nrows = length(modelNames)*length(ns)
results <- data.frame(matrix(ncol = 3, nrow = nrows))
cols <- c('N','Model','Time')
colnames(results) <- cols
idx=1


for(modelName in modelNames) {
  for(n in ns) {
    file = paste('data/', toString(n), '.csv', sep='')
    X <- read.csv(file=file,header=FALSE,sep = ',')[,-1]
    c <- read.csv(file=file,header=FALSE, sep = ',')[,1]+1
    
    start_time <- Sys.time()
    model <- Mclust(X,ks,verbose=FALSE,modelNames=modelName)
    end_time <- Sys.time()
    entry = c(n,modelName,as.numeric(difftime(end_time,start_time,units='secs')))
    print(entry)
    results[idx,] = entry
    idx <- idx+1
  }
}

write.csv(results, output_file)