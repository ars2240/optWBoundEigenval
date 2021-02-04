header="Cov_norm_5"
tail = "_zoom"

home = "~/Documents/Northwestern/Research/Optimization with Bounded Eigenvalues/"
indices = read.csv(paste0(home, "logs/", header, "_cov_shift_indices.csv"), header=FALSE)
acc = read.csv(paste0(home, "logs/", header, "_cov_shift_acc.csv"), header=FALSE)
f1 = read.csv(paste0(home, "logs/", header, "_cov_shift_f1.csv"), header=FALSE)

perterbs = colSums(abs(indices))

models = c(expression(paste(mu, "=0.01, K=1, ", rho, "=1.358", sep='')), expression(paste(mu, "=0.01, K=0, ", rho, "=1.288", sep='')),
           expression(paste(mu, "=0.001, K=5, ", rho, "=10.207", sep='')), expression(paste(mu, "=0.001, K=0, ", rho, "=10.520", sep='')),
           expression(paste(mu, "=0.005, K=1, ", rho, "=2.114", sep='')), expression(paste("Unregularized, ", rho, "=40.949", sep='')),
           expression(paste("Entropy-SGD, ", rho, "=7.362", sep='')), expression(paste("K-FAC, ", rho, "=5.819", sep='')))
baseline_acc = c(69.70990422, 70.38544616, 70.67201363, 70.8656403, 70.96632617, 71.74169341, 69.68580846, 70.82519384)
baseline_f1 = baseline_acc/100

base=6 #comparison modes
nmods = length(models)
pch = 20
cex = 0.5
cex_leg = 0.25

if (tail=="_zoom"){
  ymax = 80
  ymin = 60
} else{
  ymax = max(acc)
  ymin = min(acc)
}
xmin = min(perterbs)
xmax = max(perterbs)
png(filename=paste0(home, "plots/", header, "_cov_shift_acc", tail, ".png"), width = 4, height = 4, units = 'in', res = 300)
plot(NULL, xlim=c(xmin, xmax), ylim=c(ymin, ymax), xlab=expression('L'[1]*'-Norm of Shifts'), ylab='Accuracy')#, main=expression(paste("Affect of ", rho, " on accuracy of covariate shifts")))
for (j in 1:nmods){
  abline(h=baseline_acc[j], col=j+1, lty="dashed")
  abline(lm(as.numeric(acc[j, ])~perterbs), col=j+1, lty="dotted")
  print(summary(lm(as.numeric(acc[j, ])~perterbs)))
  points(perterbs, acc[j, ], pch=pch, col=j+1, cex=cex)
}
legend("bottomright", legend=c(models, "Baseline", "Trendline"), col=c(2:(nmods+1),1,1), lty=c(rep(NA,nmods),"dashed", "dotted"), pch=c(rep(pch,nmods),NA,NA), cex=cex_leg)
dev.off()

if (tail=="_zoom"){
  ymax = 10
  ymin = -10
} else{
  ymax = max(acc-baseline_acc)
  ymin = min(acc-baseline_acc)
}
xmin = min(perterbs)
xmax = max(perterbs)
png(filename=paste0(home, "plots/", header, "_cov_shift_acc_diff", tail, ".png"), width = 4, height = 4, units = 'in', res = 300)
plot(NULL, xlim=c(xmin, xmax), ylim=c(ymin, ymax), xlab=expression('L'[1]*'-Norm of Shifts'), ylab=expression(paste(Delta,' Accuracy')))#, main=expression(paste("Affect of ", rho, " on accuracy of covariate shifts")))
for (j in 1:nmods){
  abline(lm(as.numeric(acc[j, ]-baseline_acc[j])~perterbs), col=j+1, lty="dotted")
  print(summary(lm(as.numeric(acc[j, ]-baseline_acc[j])~perterbs))$coefficients[,4])
  points(perterbs, acc[j, ]-baseline_acc[j], pch=pch, col=j+1, cex=cex)
}
legend("bottomright", legend=c(models, "Trendline"), col=c(2:(nmods+1),1), lty=c(rep(NA,nmods),"dotted"), pch=c(rep(pch,nmods),NA), cex=cex_leg)
dev.off()

acc_ben = acc[c(1:(base-1),(base+1):nmods),]
for (j in 1:nmods){
  acc_ben[j,] = acc[j,]-acc[base,]
}
if (tail=="_zoom"){
  ymax = 10
  ymin = -5
} else{
  ymax = max(acc_ben)
  ymin = min(acc_ben)
}
xmin = min(perterbs)
xmax = max(perterbs)
png(filename=paste0(home, "plots/", header, "_cov_shift_acc_ben", tail, ".png"), width = 4, height = 4, units = 'in', res = 300)
plot(NULL, xlim=c(xmin, xmax), ylim=c(ymin, ymax), xlab=expression('L'[1]*'-Norm of Shifts'), ylab=expression(paste(Delta,' Accuracy')))#, main=expression(paste("Affect of ", rho, " on accuracy of covariate shifts")))
for (j in 1:nmods){
  if (j != base){
    abline(h=(baseline_acc[j]-baseline_acc[base]), col=j+1, lty="dashed")
    abline(lm(as.numeric(acc_ben[j, ])~perterbs), col=j+1, lty="dotted")
    print(summary(lm(as.numeric(acc_ben[j, ])~perterbs))$coefficients[,4])
    points(perterbs, acc_ben[j, ], pch=pch, col=j+1, cex=cex)
  }
}
legend("topright", legend=c(models[1:(nmods-1)], "Baseline", "Trendline"), col=c(2:nmods,1,1), lty=c(rep(NA,nmods-1),"dashed","dotted"), pch=c(rep(pch,nmods-1),NA,NA), cex=cex_leg)
dev.off()

png(filename=paste0(home, "plots/", header, "_cov_shift_acc_ben", tail, "_panel.png"), width = 4, height = 4, units = 'in', res = 300)
layout(matrix(c(1,1,2,2,3,3,4,4,5,5,6,6,0,7,7,0), 4, 4, byrow = TRUE))
par(cex=.5, mar=c(3, 3, 1, 1), mgp=c(2,1,0))#, mfrow=c(3,2))
for (j in 1:nmods){
  if (j != base){
    plot(NULL, xlim=c(xmin, xmax), ylim=c(ymin, ymax), xlab=expression('L'[1]*'-Norm of Shifts'), ylab=expression(paste(Delta,' Accuracy')), main=models[j])
    abline(h=(baseline_acc[j]-baseline_acc[base]), lty="dashed")
    abline(lm(as.numeric(acc_ben[j, ])~perterbs), lty="dotted")
    points(perterbs, acc_ben[j, ], pch=pch, cex=cex)
  }
}
dev.off()

if (tail=="_zoom"){
  ymax = .8
  ymin = .6
} else{
  ymax = max(f1)
  ymin = min(f1)
}
xmin = min(perterbs)
xmax = max(perterbs)
png(filename=paste0(home, "plots/", header, "_cov_shift_f1", tail, ".png"), width = 4, height = 4, units = 'in', res = 300)
plot(NULL, xlim=c(xmin, xmax), ylim=c(ymin, ymax), xlab=expression('L'[1]*'-Norm of Shifts'), ylab='F1 Score')#, main=expression(paste("Affect of ", rho, " on F1 score of covariate shifts")))
for (j in 1:nmods){
  abline(h=baseline_f1[j], col=j+1, lty="dashed")
  abline(lm(as.numeric(f1[j, ])~perterbs), col=j+1, lty="dotted")
  print(summary(lm(as.numeric(f1[j, ])~perterbs))$coefficients[,4])
  points(perterbs, f1[j, ], pch=pch, col=j+1, cex=cex)
}
legend("bottomright", legend=c(models, "Baseline", "Trendline"), col=c(2:(nmods+1),1,1), lty=c(rep(NA,nmods),"dashed", "dotted"), pch=c(rep(pch,nmods),NA, NA), cex=cex_leg)
dev.off()

if (tail=="_zoom"){
  ymax = .1
  ymin = -.1
} else{
  ymax = max(f1-baseline_f1)
  ymin = min(f1-baseline_f1)
}
xmin = min(perterbs)
xmax = max(perterbs)
png(filename=paste0(home, "plots/", header, "_cov_shift_f1_diff", tail, ".png"), width = 4, height = 4, units = 'in', res = 300)
plot(NULL, xlim=c(xmin, xmax), ylim=c(ymin, ymax), xlab=expression('L'[1]*'-Norm of Shifts'), ylab=expression(paste(Delta,' F1 Score')))#, main=expression(paste("Affect of ", rho, " on F1 score of covariate shifts")))
for (j in 1:nmods){
  abline(lm(as.numeric(f1[j, ]-baseline_f1[j])~perterbs), col=j+1, lty="dotted")
  print(summary(lm(as.numeric(f1[j, ]-baseline_f1[j])~perterbs))$coefficients[,4])
  points(perterbs, f1[j, ]-baseline_f1[j], pch=pch, col=j+1, cex=cex)
}
legend("bottomright", legend=c(models, "Trendline"), col=c(2:(nmods+1),1), lty=c(rep(NA,nmods),"dotted"), pch=c(rep(pch,nmods),NA), cex=cex_leg)
dev.off()


f1_ben = f1[1:(nmods-1),]
for (j in 1:nmods){
  f1_ben[j,] = f1[j,]-f1[nmods,]
}
if (tail=="_zoom"){
  ymax = .1
  ymin = -.05
} else{
  ymax = max(f1_ben)
  ymin = min(f1_ben)
}
xmin = min(perterbs)
xmax = max(perterbs)
png(filename=paste0(home, "plots/", header, "_cov_shift_f1_ben", tail, ".png"), width = 4, height = 4, units = 'in', res = 300)
plot(NULL, xlim=c(xmin, xmax), ylim=c(ymin, ymax), xlab=expression('L'[1]*'-Norm of Shifts'), ylab=expression(paste(Delta,' F1 Score')))#, main=expression(paste("Affect of ", rho, " on F1 score of covariate shifts")))
for (j in 1:(nmods-1)){
  abline(h=(baseline_f1[j]-baseline_f1[nmods]), col=j+1, lty="dashed")
  abline(lm(as.numeric(f1_ben[j, ])~perterbs), col=j+1, lty="dotted")
  print(summary(lm(as.numeric(f1_ben[j, ])~perterbs))$coefficients[,4])
  points(perterbs, f1_ben[j, ], pch=pch, col=j+1, cex=cex)
}
legend("topright", legend=c(models[1:(nmods-1)], "Baseline", "Trendline"), col=c(2:nmods,1,1), lty=c(rep(NA,nmods-1),"dashed","dotted"), pch=c(rep(pch,nmods-1),NA,NA), cex=cex_leg)
dev.off()