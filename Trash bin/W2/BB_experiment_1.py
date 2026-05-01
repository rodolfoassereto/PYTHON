import numpy as np
import scipy.sparse as sp
from differential_operators_sparse import create_partialplus_sparse, create_nabla_sparse

# definisco le shape da usare

rho_shape = (7,6,5) # (tempo, x, y)
xi_shape = (3,) + rho_shape # nella prima variabile metto i valori di rho, m_x, m_y
xi0 = np.zeros(xi_shape)

f, g = np.zeros(rho_shape[1:]), np.zeros(rho_shape[1:])
f[0,0], g[-1,-1] = 1, 1

# definisco gli operatori differenziali

partial_t, partial_0, partial_1 = [ create_partialplus_sparse(rho_shape, i) for i in range(3) ]
print( 'partial_t.shape: ', partial_t.shape ) # (210, 210)
print( 'partial_0.shape: ', partial_0.shape ) # (210, 210)
print( 'partial_1.shape: ', partial_1.shape ) # (210, 210)

diff_op_cont_eq = sp.hstack( [ partial_t.T, partial_0, partial_1 ] )
print( 'diff_op_cont_eq.shape: ', diff_op_cont_eq.shape) # (210, 630)

'''
Definisco la matrice b. La matrice b è un "undersampling" di xi. Seleziona:
- nella prima entrata (rho) la prima e ultima componente in tempo
- nella seconda entrata (m_x) i bordi "superiore e inferiore"
- nella terza entrata (m_y) i bordi "destro e sinistro"
'''

where_projection = np.zeros(xi0.shape)
where_projection[0,[0,-1],:,:] = 1
where_projection[1,:,:,[0,-1]] = 1
where_projection[2,:,[0,-1],:] = 1
where_projection = where_projection.ravel()
selected_indices = np.where(where_projection == 1)[0]  # Indices to keep
rows = np.arange(len(selected_indices))  # Row indices (0 to num_selected - 1)
cols = selected_indices  # Column indices match A_flat positions
data = np.ones(len(selected_indices))  # Values are all 1s
b = sp.csr_matrix((data, (rows, cols)), shape=(len(selected_indices), xi0.size))

print( 'b.shape: ', b.shape ) # (214, 630)

'''
Definisco l'array b0_tilde che contiene (f, g, 0, 0) e definisce il constraint per le condizioni iniziale e finale sulla curva rho, e le condizioni "zero al bordo" per m...
'''

xi_temp = np.zeros(xi0.shape)
xi_temp[0,0,:,:], xi_temp[0,-1,:,:] = f, g
b0_tilde = b @ xi_temp.ravel()

'''
...e definisco infine b0 come concatenzaione ( 0, b0_tilde), dove il primo zero è il constraint sulla divergenza, e b0_tilde è il constraint sulle condizioni di bordo
'''

b0 = np.zeros( np.prod(rho_shape) + len(b0_tilde) )
b0[ -len(b0_tilde): ] = b0_tilde

'''
Definisco ora la matrice K del sistema K @ xi = b0, dove:
- xi è la variabile contenente (rho, m_x, m_y)
- K @ xi = [ [div], [b] ] @ xi = ( d_t(rho) + d_0(m_x) + d_1(m_y) , b(xi) )
- b0 = ( 0 , b0_tilde ), dove la prima componente 0 è lunga come xi e la seconda componente b0_tilde
'''

K = sp.bmat( [ [ diff_op_cont_eq ],
               [ b ] ], format='csr' )

print( 'K.shape: ', K.shape ) # (424, 630)
print( 'rank(diff_op_cont_eq): ', np.linalg.matrix_rank( diff_op_cont_eq.toarray() ) ) # 209
print( 'rank(K): ', np.linalg.matrix_rank( K.toarray() ) )

' Definisco la proiezione su { K @ xi = b0 } '

def proj_K(xi):
    return ( xi.ravel() - K.T @ sp.linalg.spsolve( K @ K.T , K @ xi.ravel() - b0 ) ).reshape(xi.shape)

proj_K(np.random.rand(*xi_shape))
