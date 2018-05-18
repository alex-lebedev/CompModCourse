m <- as.data.frame(matrix(NA,1000, 2))
colnames(m) <- c('x', 'beta')

m$x <- rnorm(1000,10,2)
m$beta <- rnorm(1000,3,2)
m$y <- m[,1]*m[,2]
