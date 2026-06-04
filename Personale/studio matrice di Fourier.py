import numpy as np

# %% Dimensione 1, tutto a mano

x = np.array([2,4,-1,5,5], dtype=float) # array di prova

x_fft = np.fft.fft(x) # invece di numpy si può usare my_fft

pi = np.pi
def my_fft(x): # definizione esplicita di fft
    M = len(x)
    x_fft = np.zeros( x.shape, dtype=complex )
    for k in range(M):
        x_fft[k] = np.sum( [ x[i] * ( np.cos( k * i/M * 2*pi ) - 1j*np.sin( k * i/M * 2*pi ) ) for i in range(M) ] ) # con il meno (complesso coniugato)!
    return x_fft
print(np.allclose(x_fft, my_fft(x)))
    

basis = np.zeros( (5,5), dtype=complex ) # le colonne di questa matrice saranno una base di Fourier per R^5

for i,k in np.ndindex(basis.shape):
    basis[i,k] = np.cos( i * k/5 * 2*pi ) + 1j*np.sin( i * k/5 * 2*pi )

s = np.zeros(5, dtype=complex) # qui andrò a mettere
for k in range(5):
    s += x_fft[k] * basis[k]

s = s/5 # la FFT definita così non è normalizzata

print(np.round(x, decimals=2))
print(np.round(s.real, decimals=2))

# %% Spiegazioni

### fftfreq e fftshift:

# sono le frequenze [0, 1/M, ..., (M-1)/M]
# ad esempio per M=5 [0, 0.2, 0.4, 0.6, 0.8],
# ma le frequenze magggiori di M/2 sono negativizzate per convenzione: [0, 0.2, 0.4, -0.4, -0.2]

# Questo mi sembra  naturale (anche "a mano") per fft 1-dimensionale
# Per fft2 in 2 dimensioni invece è più naturale mettere lo zero al centro. Per questo, fftshift riporta la frequenza "0" al centro

### Ortonormalizzazione

# La base di Fourier costrita con cos e sin/con l'esponenziale non è normale, per cui la sua inversa va normalizzata  dividendo per M (oppure si divide direttamente per sqrt(M)).

### Se il dominio di definizione è diverso?

# Implicitamente si suppone nei conti che x discretizzi valori in [0,1), infatti "i/M" varia da 0 a (M-1)/M.

# %% Dimensione 2, tutto a mano

x = np.random.rand(5,6)

x_fft = np.fft.fftn(x)

def my_fft2(x):
    M,N = x.shape
    x_fft = np.zeros( x.shape, dtype=complex )
    for k0,k1 in np.ndindex(x_fft.shape):
        x_fft[k0,k1] = np.sum( [ x[i,j] * ( np.cos( (k0*i/M+k1*j/N) * 2*pi ) - 1j*np.sin( (k0*i/M+k1*j/N) * 2*pi ) ) for i,j in np.ndindex(x.shape) ] )
    return x_fft

print(np.allclose( x_fft, my_fft2(x) ))

basis = np.zeros( (5,6,5,6), dtype=complex )
pi = np.pi

for k0,k1,i,j in np.ndindex(basis.shape):
    dot_product = k0*i/5 + k1*j/6
    basis[k0,k1,i,j] = np.cos( dot_product * 2*pi ) + 1j*np.sin( dot_product * 2*pi )

s =  np.zeros(x.shape, dtype=complex)

for k0,k1 in np.ndindex(x.shape):
    s += x_fft[k0,k1] * basis[k0,k1]

s = s/30

print(np.round(x, decimals=2))
print(np.round(s.real, decimals=2))

# %% Standard Fourier matrix

n = 5

w = np.exp(-2*np.pi*1j/n)

F = np.zeros((n,n), dtype=complex)
for i in range(n):
    for j in range(n):
        F[i,j] = np.exp(2*np.pi*1j*i*(j/n)) / np.sqrt(n)

print( np.round( F, decimals=2 ) )

print( 'F = F.T: ', np.allclose(F, F.T) )
print( 'F = F^*:', np.allclose(F, np.matrix.conjugate(F)) )
print( 'F@F*=id:', np.allclose( F @ np.matrix.conjugate(F), np.eye(n) ) )

print( 'Rows are unitary: ', np.allclose( np.linalg.norm( F, axis=0 ), np.ones(n) ) )
print( 'Columns are unitary: ', np.allclose( np.linalg.norm( F, axis=1 ), np.ones(n) ) ) # redundnant

orthogonality_check = True

for i in range(n):
    for j in range(i+1,n):
        orthogonality_check *= np.isclose( np.vdot(F[i],F[j]), 0 )

print( 'Orthogonality_check: ', orthogonality_check )

# %% Variable frequencies Fourier matrix
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

print( np.round( F2, decimals=2 ) )

print( 'F2 = F2.T: ', np.allclose(F2, F2.T) )
print( 'F2=F2^*:', np.allclose(F2, np.matrix.conjugate(F2)) )

orthogonality_check = True

print( 'Rows are unitary: ', np.allclose( np.linalg.norm( F2, axis=0 ), np.ones(n) ) )
print( 'Columns are unitary: ', np.allclose( np.linalg.norm( F2, axis=1 ), np.ones(n) ) ) # redundnant

for i in range(n):
    for j in range(i+1,n):
        orthogonality_check *= np.isclose( np.vdot(F2[i],F2[j]), 0 )

print( 'Orthogonality_check: ', orthogonality_check )

