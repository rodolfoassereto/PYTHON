import numpy as np

v = [2,4,6]
w = [0, 1, -1]


mat0 = np.array([[1,1,0],
                  [1,1,2],
                  [0,-2,-3]]) # 4x4

mat1 = np.array([[1,1,0,2],
                  [1,1,2,4],
                  [0,0,0,-1],
                  [0,0,-2,-3]]) # 4x4

mat2 = mat0[1:,1:]







# Old calculationms for the TV inequality

# zer6 = [0,0,0,0,0,0]
# zer2 = [0,0]
# A = np.array([zer6,
#               zer6,
#               zer6,
#               zer2+[0,.25]+zer2,
#               zer2+[.5,.5]+zer2,
#               zer2+[0.5,0.5]+zer2,
#               zer6,
#               zer6])

# A1 = np.array([zer6,
#               zer6,
#               zer2+[0.0625,0.0625]+zer2,
#               zer2+[0.0625,0.0625]+zer2,
#               zer2+[0.5,0.5]+zer2,
#               zer2+[0.5,0.5]+zer2,
#               zer6,
#               zer6])

  # B = np.array([[0,0,0],
#               [0,0.0625,0],
#               [0,0.5,0],
#               [0,0,0]])

# print(A)
# print(A1)
# print(B)
# print("TV(A)=",TV(A))
# print("TV(A1)=",TV(A1))
# print("2*TV(B)=",2*TV(B))