from rop import ROp

rop = ROp('hessTestRand')

rop.compute()

norm = rop.gradCompare()
print('Gradient difference: %e' % norm)

norm = rop.ropCompare()
print('ROp difference: %e' % norm)

norm = rop.compare()
print('R2Op difference: %e' % norm)
