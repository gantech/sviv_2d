import numpy as np



indices = np.array(range(9))

indices_mat = np.copy(indices).reshape(3,3)

print(indices_mat)
print(indices_mat.reshape(-1))


mat = np.random.rand(3,3)
# mat = np.random.randint(0, 9, (3,3))

x = mat[0, :]
y = mat[1, :]
z = mat[2, :]

vv = mat.reshape(-1)

# Determinant of matrix
det = vv[0] * (vv[4]*vv[8] - vv[5]*vv[7]) \
      - vv[1] * (vv[3]*vv[8] - vv[5]*vv[6]) \
      + vv[2] * (vv[3]*vv[7] - vv[4]*vv[6]) \

vv_inv_det = np.vstack((np.cross(y, z), np.cross(z, x), np.cross(x, y))).T

mat_inv = (1/det)*vv_inv_det

res = mat_inv @ mat - np.eye(3)
res = np.max(np.abs(res).reshape(-1))

print(res)

# #### Verification checks
# 
# print('Matrices')
# print(mat)
# 
# print('\nDet:')
# # print(det)
# # print(np.linalg.det(mat))
# 
# print('\nAdjoint Mat:')
# print(vv_inv_det)
# print(np.linalg.inv(mat)*det)
# 
# print('\nIdentity check:')
# print(mat_inv @ mat)

