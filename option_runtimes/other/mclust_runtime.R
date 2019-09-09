#This script creates the mclust data for Figure 4 in the pyclust paper
#Note that the results may be saved as minutes - in which case you should convert the resulting csvs into seconds


library("mclust")

base <- 40
factor <- 2
num_sets <- 18
exps <- 0:num_sets 
ns <- base * factor ^ exps
output_file <- 'mclust_times_EEE.csv' 
results <- data.frame(matrix(ncol = 2, nrow = num_sets+1))


x <- c("N","Time")
colnames(results) <- x

modelNames='EEE'#mclust.options("emModelNames")
ks <- 2:6
idx <- 1

for(n in ns) {
  file = paste('data/', toString(n), '.csv', sep='')
  X <- read.csv(file=file,header=FALSE,sep = ',')[,-1]
  c <- read.csv(file=file,header=FALSE, sep = ',')[,1]+1
  
  start_time <- Sys.time()
  model <- Mclust(X,ks,verbose=FALSE,modelNames=modelNames)
  end_time <- Sys.time()
  print(n)
  print(end_time-start_time)
  results$Time[idx] = end_time-start_time
  results$N[idx] = n
  idx <- idx+1
}

write.csv(results, output_file)