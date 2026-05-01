import numpy as np
from default import printarr

n = 5

### Standard Fourier matrix

w = np.exp(-2*np.pi*1j/n)

F = np.zeros((n,n), dtype=complex)
for i in range(n):
    for j in range(n):
        F[i,j] = np.exp(2*np.pi*1j*i*(j/n)) / np.sqrt(n)

printarr( F )

print( 'F = F.T: ', np.allclose(F, F.T) )
print( 'F=F^*:', np.allclose(F, np.matrix.conjugate(F)) )

orthogonality_check = True

print( 'Rows are unitary: ', np.allclose( np.linalg.norm( F, axis=0 ), np.ones(n) ) )
print( 'Columns are unitary: ', np.allclose( np.linalg.norm( F, axis=1 ), np.ones(n) ) ) # redundnant

for i in range(n):
    for j in range(i+1,n):
        orthogonality_check *= np.isclose( np.vdot(F[i],F[j]), 0 )

print( 'Orthogonality_check: ', orthogonality_check )

### Variable frequencies Fourier matrix
print('______________________________________________________________________')
print()

dom = np.arange(n)
# dom = np.sort( np.random.choice(2*n, size=n, replace=False) )
freq = np.arange(n)/n
# freq = arr = np.sort( np.random.choice(2*n, size=n, replace=False) )
# print(arr)

F2 = np.zeros((n,n), dtype=complex)
for i in range(n):
    for j in range(n):
        F2[i,j] = np.exp(2*np.pi*1j*dom[i]*(freq[j])) / np.sqrt(n)

printarr( F2 )

print( 'F2 = F2.T: ', np.allclose(F2, F2.T) )
print( 'F2=F2^*:', np.allclose(F2, np.matrix.conjugate(F2)) )

orthogonality_check = True

print( 'Rows are unitary: ', np.allclose( np.linalg.norm( F2, axis=0 ), np.ones(n) ) )
print( 'Columns are unitary: ', np.allclose( np.linalg.norm( F2, axis=1 ), np.ones(n) ) ) # redundnant

for i in range(n):
    for j in range(i+1,n):
        orthogonality_check *= np.isclose( np.vdot(F2[i],F2[j]), 0 )

print( 'Orthogonality_check: ', orthogonality_check )

