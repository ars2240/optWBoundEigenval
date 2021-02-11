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
           expression(paste("Entropy-SGD, ", rho, "=7.362", sep='')), expression(paste("K-FAC, ", rho, "=5.819", sep='')),
           expression(paste("Asym. Valley, ", rho, "=8.985", sep='')))
baseline_acc = c(69.70990422, 70.38544616, 70.67201363, 70.8656403, 70.96632617, 71.74169341, 69.68580846, 70.82519384, 70.99459994)
baseline_f1 = baseline_acc/100

nmods = length(models)

mode = 9 #comparison mode
acc_ben = acc[1:(nmods-1),]
for (j in 1:nmods){
  if (mode>=1 && mode<=nmods){
    acc_ben[j,] = acc[j,]-acc[mode,]
  } else{
    acc_ben[j,] = acc[j,]
  }
}
for (j in 1:nmods){
  if (j != mode){
    print(summary(lm(as.numeric(acc_ben[j, ])~perterbs))$r.squared)#$coefficients[2,4])
  }
}