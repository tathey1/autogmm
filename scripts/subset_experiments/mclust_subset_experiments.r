library("mclust")

#Synthetic
X <- read.csv(file='../../data/synthetic.csv',header=FALSE,sep = ',')[,-1]
idx_mat <- read.csv(file='idxs_synthetic.csv',header=TRUE, sep=',')
idx_mat <- idx_mat[,2:ncol(idx_mat)]
c <- read.csv(file='../../data/synthetic.csv',header=FALSE, sep = ',')[,1]+1
modelNames=mclust.options("emModelNames")
ks <- 1:20

run_nums = 10
results <- data.frame(matrix(ncol = 2, nrow = run_nums))
x <- c("ARI","Time")
colnames(results) <- x
n_full <- nrow(X)
n <-  floor(0.8*n_full)
X_full <- X
c_full <- c


for(r in 1:run_nums) {
  print(paste('Run number: ', toString(r)))
  idxs <- idx_mat[,r]
  X <- X_full[idxs[1:n],]
  c <- c_full[idxs[1:n]]
  
  start_time <- Sys.time()
  plots <- data.frame('x1'=unlist(X[1]),"x2"=unlist(X[2]),"c_true"=c)
  
  #best bic***************************************
  model <- Mclust(X,ks,verbose=FALSE,modelNames=modelNames)
  best_combo_bic <- model$modelName
  best_g_bic <- model$G
  best_bic <- max(model$BIC,na.rm=T)
  c_hat_bic <- factor(model$classification)
  
  plots$c_hat_bic = c_hat_bic
  best_ari_bic = adjustedRandIndex(plots$c_true,plots$c_hat_bic)
  
  results$ARI[r] = best_ari_bic
  
  end_time <- Sys.time()
  results$Time[r] = end_time-start_time
  
}
write.csv(results, "mclust_synthetic.csv")

#Breast Cancer
#read mean texture, extreme area, and extreme smoothness
X <- read.csv(file='../../data/wdbc.data',header=FALSE,sep=',')[,c(4,26,27)]
idx_mat <- read.csv(file='idxs_bc.csv',header=TRUE, sep=',')
idx_mat <- idx_mat[,2:ncol(idx_mat)]
c <- read.csv(file='../../data/wdbc.data',header=FALSE,sep=',')[,2]
modelNames=mclust.options("emModelNames")
ks <- 1:20

run_nums = 10
results <- data.frame(matrix(ncol = 2, nrow = run_nums))
x <- c("ARI","Time")
colnames(results) <- x
n_full <- nrow(X)
n <-  floor(0.8*n_full)
X_full <- X
c_full <- c


for(r in 1:run_nums) {
  print(paste('Run number: ', toString(r)))
  idxs <- idx_mat[,r]
  X <- X_full[idxs[1:n],]
  c <- c_full[idxs[1:n]]
  
  start_time <- Sys.time()
  plots <- data.frame('x1'=unlist(X[1]),"x2"=unlist(X[2]),"c_true"=c)
  
  #best bic***************************************
  model <- Mclust(X,ks,verbose=FALSE,modelNames=modelNames)
  best_combo_bic <- model$modelName
  best_g_bic <- model$G
  best_bic <- max(model$BIC,na.rm=T)
  c_hat_bic <- factor(model$classification)
  
  plots$c_hat_bic = c_hat_bic
  best_ari_bic = adjustedRandIndex(plots$c_true,plots$c_hat_bic)
  
  results$ARI[r] = best_ari_bic
  
  end_time <- Sys.time()
  results$Time[r] = end_time-start_time
  
}
write.csv(results, "mclust_bc.csv")


#Drosophila
X <- read.csv(file='../../data/embedded_right.csv',header=TRUE,sep=',')
c_file <- read.csv(file='../../data/classes.csv',header=TRUE,sep=',')
idx_mat <- read.csv(file='idxs_drosophila.csv',header=TRUE, sep=',')
idx_mat <- idx_mat[,2:ncol(idx_mat)]
c <- factor(c_file$x)
modelNames=mclust.options("emModelNames")
ks <- 1:20

run_nums = 10
results <- data.frame(matrix(ncol = 2, nrow = run_nums))
x <- c("ARI","Time")
colnames(results) <- x
n_full <- nrow(X)
n <-  floor(0.8*n_full)
X_full <- X
c_full <- c


for(r in 1:run_nums) {
  print(paste('Run number: ', toString(r)))
  idxs = idx_mat[,r]
  X <- X_full[idxs[1:n],]
  c <- c_full[idxs[1:n]]
  
  start_time <- Sys.time()
  plots <- data.frame('x1'=unlist(X[1]),"x2"=unlist(X[2]),"c_true"=c)
  
  #best bic***************************************
  model <- Mclust(X,ks,verbose=FALSE,modelNames=modelNames)
  best_combo_bic <- model$modelName
  best_g_bic <- model$G
  best_bic <- max(model$BIC,na.rm=T)
  c_hat_bic <- factor(model$classification)
  
  plots$c_hat_bic = c_hat_bic
  best_ari_bic = adjustedRandIndex(plots$c_true,plots$c_hat_bic)
  
  results$ARI[r] = best_ari_bic
  
  end_time <- Sys.time()
  results$Time[r] = end_time-start_time
  
}
write.csv(results, "mclust_drosophila.csv")
