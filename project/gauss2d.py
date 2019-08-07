import numpy as np
import scipy.stats as ss
import matplotlib.pyplot as pl
avg = np.array([0, 0])
cov = np.array([[1, 0], [0, 1]])
nnn = 25
xs = np.random.multivariate_normal(avg, cov, nnn)

s3 = max([cov[i, i] for i in range(cov.shape[0])]) * 3
xx = np.linspace(-s3, s3)
yy = np.linspace(-s3, s3)
xv, yv = np.meshgrid(xx, yy, sparse=False, indexing='ij')

xy = np.array([xv.ravel(), yv.ravel()]).T
zz = ss.multivariate_normal.pdf(xy, mean=avg, cov=cov)
zz = np.reshape(zz, xv.shape)

#xm = np.mean(xs, axis=0)
xm = np.median(xs, axis=0)
pl.contour(xx, yy, zz)
pl.axis('equal')
pl.axis('off')
pl.plot(xs[:,0], xs[:,1], '.')
pl.plot(xm[0], xm[1], 'r+')
pl.plot([xm[0]-s3, xm[0]+s3], [xm[1]-s3, xm[1]-s3], 'r')
pl.plot([xm[0]-s3, xm[0]+s3], [xm[1]+s3, xm[1]+s3], 'r')
pl.plot([xm[0]-s3, xm[0]-s3], [xm[1]-s3, xm[1]+s3], 'r')
pl.plot([xm[0]+s3, xm[0]+s3], [xm[1]-s3, xm[1]+s3], 'r')