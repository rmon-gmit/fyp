import numpy as np

stars = "*************"


mylist = [1, 2, 3]    # normal list
myarray = np.array(mylist)    # list casted to a numpy array
print(stars, " NORMAL LIST ", stars)
print(mylist)
print(stars, " NUMPY ARRAY ", stars)
print(myarray)

print(stars, " ARRANGED ARRAY ", stars)
myarrange = np.arange(0, 10, 3)
print(myarrange)

print(stars, " ZEROS 2D ARRAY ", stars)
myzeros = np.zeros(shape=(4, 4))     # 2d array of 0's. shape=(rows, columns)
print(myzeros)

print(stars, " ONES 2D ARRAY ", stars)
myones = np.ones(shape=(4, 4))     # 2d array of 1's. shape=(rows, columns)
print(myones)

print(stars, " RANDOM NUMBERS ", stars)
np.random.seed(101)     # seed allows you to create the same random numbers, seed dumber is arbitrary
arr = np.random.randint(0, 100, 10)     # randint(range from, range to, number of values to print)
print("Random array 1: ", arr)

arr2 = np.random.randint(0, 100, 10)     # randint(range from, range to, number of values to print)
print("Random array 2: ", arr2)

print(stars, " ARRAY MAX VALUE OF ARRAY 1 ", stars)
print(arr.max())

print(stars, " ARRAY MIN VALUE OF ARRAY 2 ", stars)
print(arr2.min())

print(stars, " INDEX LOCATION OF ARRAY 1 MAX VALUE ", stars)
print(arr.argmax())

print(stars, " INDEX LOCATION OF ARRAY 2 MIN VALUE ", stars)
print(arr2.argmin())

print(stars, " MEAN VALUE OF ARRAY 2 ", stars)
print(arr2.mean())

print(stars, " RESHAPING ARRAY 1 TO BE 2 ROWS AND 5 COLUMNS ", stars)
print(arr2.reshape(2, 5))   # wont work when array reshape size doesnt make sense with array size

print(stars, " 10X10 MATRIX ", stars)
matrix = np.arange(0, 100).reshape(10, 10)
print(matrix)

print(stars, " PICKING OUT INTERSECTION OF ROW 3 COLUMN 5 ", stars)
print(matrix[3, 5])

print(stars, " SLICING ALL VALUES OF COLUMN 1 ", stars)
print(matrix[1, :])

print(stars, " RESHAPING IT TO A COLUMN ", stars)
print(matrix[1, :].reshape(10, 1))

print(stars, " SLICING A 3X3 MATRIX FROM ROW AND COLUMN 0 TO ROW AND COLUMN 3 ", stars)
print(matrix[0:3, 0:3])     # grabs up to but not including 3!

print(stars, " REASSIGNING SAME AREA OF MATRIX TO 0 ", stars)
matrix[0:3, 0:3] = 0
print(matrix)




















