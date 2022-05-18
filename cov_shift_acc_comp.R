header="Cov_norm_5"
tail = "_zoom"

home = "~/Documents/Northwestern/Research/Optimization with Bounded Eigenvalues/"
indices = read.csv(paste0(home, "logs/", header, "_cov_shift_indices.csv"), header=FALSE)
acc = read.csv(paste0(home, "logs/", header, "_cov_shift_acc.csv"), header=FALSE)
f1 = read.csv(paste0(home, "logs/", header, "_cov_shift_f1.csv"), header=FALSE)

acc = acc[-10,]; f1 = f1[-10,]

perterbs = colSums(abs(indices))

nmods = 14
mode = 0 #comparison mode
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
    print(summary(lm(as.numeric(acc_ben[j, ])~perterbs)))#$r.squared)#$coefficients[2,4])
  }
}