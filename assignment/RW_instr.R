# Load libraries:

rm(list=ls())

library(R.matlab)
library(RSEIS)
library(xlsx)
library(ggplot2)

rdat <- readMat('/Users/alebedev/GitHub/CompModCourse/assignment/data/ildp2_pilot_gsr.mat')$rdat

endp <- which(rdat[,1]< c(mean(rdat[,1]) - 3*sd(rdat[,1])))[1]-500
rdat <- rdat[1:endp,]
gsr <- rdat[,1]
gsr_d <- detrend(gsr)
shock <- rdat[,3]

d <- as.data.frame(cbind(gsr, gsr_d,shock))

#ts.plot(d,gpars=list(yaxt='n', col=c(1:3)))
d$ms <- as.numeric(as.vector(row.names(d)))
# show data with two shock trials:
#ts.plot(d[c(16000*5):c(16000*9),2:3],gpars=list(yaxt='n', col=c(1:2)))

timing <- read.xlsx2('/Users/alebedev/GitHub/CompModCourse/assignment/data/pilot_TaskDesign.xlsx',1)

timing$trial <- as.numeric(as.vector(timing$trial))
timing$shockstart <- as.numeric(as.vector(timing$shockstart))
timing$shock <- as.numeric(as.vector(timing$shock))

timing_long <- data.frame(trial=rep(NA,max(timing$trial)*16000+16000))
timing_long$shock <- NA

for (i in 1:max(timing$trial)){
  tmp <- subset(timing, timing$trial == i)
  stamp <- (tmp$trial-1)*16000+1
  timing_long$trial[stamp:c(stamp+16000)] <- i
  if (tmp$shock==0.2){
    timing_long$shock[c(stamp+tmp$shockstart*1000):c(stamp+tmp$shockstart*1000+200)] <- 1
    }
}
timing_long <- timing_long[!is.na(timing_long$trial),]
timing_long$ms <- as.numeric(as.vector(row.names(timing_long)))
timing_long$shock[is.na(timing_long$shock)]<-0



# Check between CS+ timing (should be roughly 31.78 seconds):
(d$ms[which(d$shock>0)[185]]-d$ms[which(d$shock>0)[1]])/1000
(16*7+5.12)-(16*5+5.34)
(timing_long$ms[which(timing_long$shock>0)[202]]-timing_long$ms[which(timing_long$shock>0)[1]])/1000

# ms until the first electric shock (theory):
timing_long$ms[which(timing_long$shock>0)[1]]
# ms until the first electric shock (practice: actual gsr data):
d$ms[which(d$shock>0)[1]]

# Select based on timing model:

stamp1 <- d$ms[which(d$shock>0)[1]]-timing_long$ms[which(timing_long$shock>0)[1]]
d_sel <- d[stamp1:dim(d)[1],]

d_sel$ms <- as.numeric(c(1:dim(d_sel)[1]))

dd <- merge(d_sel, timing_long, by='ms')
dd$gsr_d<- dd$gsr_d-min(dd$gsr_d)

cor(dd$shock.x,dd$shock.y)
#ts.plot(cbind(dd$shock.x,dd$shock.y),gpars=list(col=c(1:2)))


final_df <- data.frame(trial=c(1:max(dd$trial)))
upperBound <- max(timing$shockstart)*2000 # defines upper boundary of the time-window
for (i in 1:max(dd$trial)){
  tmp <- subset(dd, dd$trial == i)
  final_df$gsr[i]<- max(tmp$gsr[2000:upperBound])-min(tmp$gsr[1:c(upperBound/2)])
  final_df$gsr_d[i]<- max(tmp$gsr_d[2000:upperBound])-min(tmp$gsr_d[1:c(upperBound/2)])
}

# Plot individual trial responses:
i=41; tmp <- subset(dd, dd$trial == i); ts.plot(tmp$gsr)



final_d <- merge(final_df, timing, by='trial')

final_d$cs <- 0

final_d$cs[is.element(final_d$block,c('block1','block3')) & final_d$face=='stim/FaceA.JPG']<-1
final_d$cs[is.element(final_d$block,c('block2','block4')) & final_d$face=='stim/FaceB.JPG']<-1
t.test(final_d$gsr[final_d$cs==0],final_d$gsr[final_d$cs==1])
boxplot(final_d$gsr[final_d$cs==0],final_d$gsr[final_d$cs==1])


# Ploting responses per trial:
final_d$cs <- as.factor(final_d$cs)
ggplot(final_d) + geom_point(aes(x = cs, y = gsr, color=cs), size = 1, data=final_d)+
  theme_bw()+theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank())

# Time-series plot of individual responses:
ts.plot(cbind(final_d$gsr, as.numeric(final_d$block),as.numeric(final_d$cs)*3, (as.numeric(final_d$fix)-1)*10), gpars=list(yaxt='n', col=c(1:4)))


ddd <- merge(dd, final_d[,c('trial','stype', 'block', 'face', 'shock', 'shockstart', 'fix', 'cs')], by='trial')
ddd$timing <- c(rep(1:16000,80),1)

d <- aggregate(. ~ timing+cs, ddd[,], mean)

ts.plot(cbind(d$gsr_d[d$cs==0],d$gsr_d[d$cs==1]), gpars=list(col=c(1:2)))
  

ddd$cs <- as.factor(ddd$cs+1)  

ddd1 <- ddd

ggplot(ddd1) + 
  geom_point(aes(x = timing, y = gsr_d, color=cs), size = 1, data=ddd1) +
  stat_smooth(aes(x = timing, y = gsr_d, color=cs, method = "loess"), size=1, se=T, span=0.95, data=ddd1) +
#  coord_cartesian(ylim = c(0.1, 0.23)) + 
  theme_bw()+theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank())




ddd1 <- d

ddd1$cs <- as.factor(ddd1$cs)
ggplot(ddd1) + 
  #geom_point(aes(x = timing, y = gsr_d, color=cs), size = 1, data=ddd1) +
  
  stat_smooth(aes(x = timing, y = gsr_d, color=cs, method = "loess"), size=1, se=T, span=0.95, data=ddd1) +
  theme_bw()+theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank())

# save the data:
save(final_d, file='/Users/alebedev/GitHub/CompModCourse/assignment/data/final_d.rda')
final_d$stim <- as.numeric(final_d$face)
final_d$shock <- as.numeric(final_d$shock)*5
final_d$block <- as.numeric(final_d$block)
rdata <- final_d[,c('trial','block','stim','shock','gsr')]
save(rdata, file='/Users/alebedev/GitHub/CompModCourse/assignment/data/rdata.rda')


############
# ANALYSIS #
############
rm(list=ls())
library(rstan)
range01 <- function(x){(x-min(x))/(max(x)-min(x))}

rstan_options(auto_write = TRUE)
options(mc.cores = parallel::detectCores())

#load('/Users/alebedev/GitHub/CompModCourse/assignment/data/final_d.rda')
load('/Users/alebedev/GitHub/CompModCourse/assignment/data/rdata.rda')


rdata$gsr <- range01(rdata$gsr)

rdata <- rbind(rdata,rdata,rdata,rdata,rdata,rdata,rdata,rdata,rdata,rdata)
rdata$subject <- c(rep(1,max(rdata$trial)),
                   rep(2,max(rdata$trial)),
                   rep(3,max(rdata$trial)),
                   rep(4,max(rdata$trial)),
                   rep(5,max(rdata$trial)),
                   rep(6,max(rdata$trial)),
                   rep(7,max(rdata$trial)),
                   rep(8,max(rdata$trial)),
                   rep(9,max(rdata$trial)),
                   rep(10,max(rdata$trial))
                   )


rdata$gsr <- rdata$gsr+sqrt(rnorm(800,0,0.5)^2)
                   

#rdata <- rdata[,c('subject', 'trial', 'gsr', 'face', 'shock', 'cs', 'reversal_first')]
# rdata$choice <- 1
# rdata$choice[rdata$gsr>mean(rdata$gsr)]<-2
# rdata$gsr <- rdata$choice

# Reorganize the data:
subjList <- unique(rdata[,'subject'])  # list of subjects x blocks
numSubjs <- length(subjList)  # number of subjects
maxTrials <- max(rdata$trial)


Tsubj <- as.vector(rep(0, numSubjs))
choice <- array(0, c(numSubjs, maxTrials) )
outcome <- array(0, c(numSubjs, maxTrials) )

for (i in 1:numSubjs) {
  curSubj      <- subjList[i]
  Tsubj[i] <- sum(rdata$subject == curSubj)  # Tsubj[N]
  useTrials    <- Tsubj[i]
  tmp          <- subset(rdata, rdata$subject == curSubj)
  choice[i, 1:useTrials]   <- tmp$gsr
  outcome[i, 1:useTrials]   <- tmp$cs
}


dataList <- list(
  N             = numSubjs,
  T             = maxTrials,
  choice        = choice,
  outcome       = outcome,
  Tsubj         = Tsubj
)
#########
# Set sampler parameters
adapt_delta   = 0.99
stepsize      = 1
max_treedepth = 10
nchain        = 4
nthin         = 1
niter         = 2000
nwarmup       = 1000
nthin         = 1

### Run Model:
myfit <- stan(file    = "/Users/alebedev/GitHub/CompModCourse/assignment/stan_models/bandit2arm_instr.stan",
              data    = dataList, 
              #                  pars    = parameters,
              warmup  = nwarmup,
              #                  init    = genInitList, 
              iter    = niter, 
              chains  = nchain,
              thin    = nthin,
              control = list(adapt_delta   = adapt_delta, 
                             max_treedepth = max_treedepth, 
                             stepsize      = stepsize) )