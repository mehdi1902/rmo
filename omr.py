from skimage.io import imread, imshow, imsave
from matplotlib.pyplot import plot
from matplotlib.pylab import show
from skimage import img_as_bool
import matplotlib.pyplot as plt
import numpy as np
from skimage.color import rgb2gray
from skimage.filter import rank
from skimage.morphology import disk, erosion, dilation
from skimage.filter import canny, sobel
import skimage.morphology
from scipy.ndimage.morphology import binary_fill_holes
import pywt


'''
Notes:
- If sheet is empty of answers
- Sparse answers
'''

POINTS = 3
INTERVAL = 5
NUM = 10
FILTER_SIZE = 4



def separate_columns(p, arr, row=True):
    i = 0
    arr = np.array(arr)
    # if not row:
    #     arr = (arr > np.max(arr)/10.) * arr
    res = []
    while i < len(arr):
        if arr[i]==0:
            if sum(arr[i:])>0:
                idx = np.where(arr[i:]>0)[0][0]
            else:
                idx = len(arr)
            if row:
                thresh = len(arr)/250.
            else:
                thresh = len(arr)/100.
            if idx> thresh or row:
                res.append(i)
                res.append(i + idx)
            i += idx
        else:
            i += 1
#        print i
    idxs = []
    for i in range(1, len(res)-1, 2):
        idxs.append([res[i], res[i+1]])
    if row:
        rows = []
        for idx in idxs:
            rows.append(p[idx[0]:idx[1], :])
        return rows, idxs
    else:
        cols = []
        for idx in idxs:
            cols.append(p[:, idx[0]:idx[1]])
        return cols, idxs


img = rgb2gray(imread('1102.jpg'))[177:960, 14:706]
# p = rgb2gray(imread('1142_2.jpg'))[177:960, 14:706]
# p = rgb2gray(imread('1208(D).jpg'))[200:1435, :]
##################
# Separate answers
tmp = dilation(erosion(~img, disk(FILTER_SIZE)), disk(FILTER_SIZE))
p = tmp > 10
# p = img_as_bool(tmp)
FILTER_SIZE = p.shape[0]/400.

#########################
# Separate border of test
m, n = p.shape
#rows_diff = [0]*m
cols_diff = [0]*n

for i in range(m-1):
    for j in range(n-1):
#        rows_diff[i] += abs(p[i, j] - p[i, j+1])
        cols_diff[j] += abs(p[i, j] - p[i+1, j])


cols, col_idx = separate_columns(p, cols_diff, row=False)
#rows, row_idx = separate_columns(rows_diff, row=True)

#n_cols = 0
#min_bound = rows[0].shape[1]
#max_bound = 0


rows_diffs = []
rows = []
rows_idx = []
for c in cols:
    rows_diff = [0]*m
    for i in range(c.shape[0]-1):
        for j in range(c.shape[1]-1):
            rows_diff[i] += abs(c[i, j] - c[i, j+1])
    r, r_idx = separate_columns(c, rows_diff, row=True)
#    rows.append(r)
    rows_idx.append(r_idx)
    rows_diffs.append(rows_diff)
    
H = []
for i in range(len(rows_idx)):
    H.append(np.mean([j[1]-j[0] for j in rows_idx[i]]))
height = np.mean(H)

for c in cols:
    c.reshape(())


n_cols = len(cols)
#n_rows = len(rows)

####################
# Separate questions
width = np.mean(np.array([c.shape[1] for c in cols]))
#height = np.mean(np.array([r.shape[0] for r in rows]))



# q = []
# for r in rows:
#     for i in range(n_cols):
#         q.append(r[:, int(i*width):int((i+1)*width)])

#questions = []
#for c in cols:
#    for idx in row_idx:
#        questions.append(c[idx[0]:idx[1], :])
#
#
#predict = []
#for q in questions:
#    max_area = 0
#    max_idx = -1
#    l = q.shape[1]
#    for i in range(4):
#        area = np.sum(q[:, i*l/4:(i+1)*l/4])
#        if area > max_area:
#            max_area = area
#            max_idx = i + 1
#    predict.append(max_idx)




#imshow(p)
#show()

#==============================================================================
# m, n = p.shape
# w = n/NUM
# h = m/NUM
# 
# windows = []
# 
# for i in range(NUM):
#     windows.append(p[i*h:(i+1)*h, i*w:(i+1)*w])
#==============================================================================



#==============================================================================
#  edge = sobel(p)
#  imshow(binary_fill_holes(edge > .2, disk(4)))
# 
# titles = ['Approximation', ' Horizontal detail',
#           'Vertical detail', 'Diagonal detail']
# 
# coeffs2 = pywt.dwt2(p, 'rbio4.4')
# LL, (LH, HL, HH) = coeffs2
# 
# rows = np.sum(HH, axis=1)
# cols = np.sum(HH, axis=0)
# 
# dif_row = np.max(rows) - np.min(rows)
# dif_col = np.max(cols) - np.min(cols)
# 
# if dif_row > dif_col:
#     points = find_border(rows)
#     questions = p[min(points):max(points), :]
# else:
#     points = find_border(cols)
#     questions = p[:, min(points):max(points)]
# imshow(questions)
#==============================================================================

