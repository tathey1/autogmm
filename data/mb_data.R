install.packages("mvtnorm", repos='http://cran.us.r-project.org')
install.packages("devtools")
require(devtools)
library(devtools)
devtools::install_github("youngser/mbstructure")
require(mbstructure)
data(MBconnectome)

set.seed(1)
# R hemi
out <- generate.graph(newrdat, vdf.right)
g <- out$g
dmax <- 50
Xhat <- doEmbed(g, dmax)
write.csv(Xhat, "Xhat_R.csv")
adj <- as_adjacency_matrix(g, "both")
vdf <- out$vdf
type <- vdf$type
write.csv(type, "classes_R.csv")
sub_class <- out$vdf$v
write.csv(sub_class, "out_vdf_v_R.csv")

# L hemi
out <- generate.graph(newldat, vdf.left)
g <- out$g
Xhat <- doEmbed(g, dmax)
write.csv(Xhat, "Xhat_L.csv")
adj <- as_adjacency_matrix(g, "both")
vdf <- out$vdf
type <- vdf$type
write.csv(type, "classes_L.csv")
sub_class <- out$vdf$v
write.csv(sub_class, "out_vdf_v_L.csv")




