#This script reproduces information found in the pyclust paper
#First, it shows the clustering, found in the appendix of the paper
#Then it prints information about the model selected, found in Table 2
#Lastly, it constructs the bicplot. The bicplot for the drosophila data is found in figure 3

library("mclust")
setwd(dirname(sys.frame(1)$ofile))

#0-synthetic, 1-BC, 2-drosophila
dataset = 2 #****************************Change this to view different datasets


if (dataset==0) {
  X <- read.csv(file='../data/synthetic.csv',header=FALSE,sep = ',')[,-1]
  first_dim <- X$V2
  second_dim <- X$V3
  c <- read.csv(file='../data/synthetic.csv',header=FALSE, sep = ',')[,1]+1
  modelNames=mclust.options("emModelNames")
  ks <- 1:20
} else if (dataset==1) {
  #read mean texture, extreme area, and extreme smoothness
  X <- read.csv(file='../data/wdbc.data',header=FALSE,sep=',')[,c(4,26,27)]
  first_dim <- X$V4
  second_dim <- X$V26
  c <- read.csv(file='../data/wdbc.data',header=FALSE,sep=',')[,2]
  modelNames=mclust.options("emModelNames")
  ks <- 1:20
} else if (dataset==2) {
  X <- read.csv(file='../data/embedded_right.csv',header=TRUE,sep=',')
  first_dim <- X$V1
  second_dim <- X$V2
  c_file <- read.csv(file='../data/classes.csv',header=TRUE,sep=',')
  c <- factor(c_file$x)
  modelNames=mclust.options("emModelNames")
  ks <- 1:20
}

colors = c('red','blue','green','yellow','brown','black', 'orange',"coral","cyan","darkolivegreen1","gold2","burlywood","gray64","deeppink")

model <- Mclust(X,ks,verbose=FALSE, modelNames=modelNames)
combo <- model$modelName
k <- model$G
bic <- max(model$BIC,na.rm=T)
ari <- adjustedRandIndex(c,model$classification)


plot(first_dim,second_dim,col=colors[model$classification],pch=19,xlab='feature 1',ylab='feature 2',main='mclust clustering') 
print(paste('model: ',combo))
print(paste('k: ',k))
print(paste('bic: ',bic))
print(paste('ari: ',ari))

BIC <- mclustBIC(X,ks,verbose=FALSE, modelNames=modelNames)

par(cex.axis=1.2,cex.lab=1.5,pty='s')
plot(BIC)
title(main='b) mclust')



