# Gauss-Jordan Matrix Inversion
Rachel Tan 

## Implementation

`inverse.cu` reads an NxN matrix $A$ of type `int` from a `.txt` file stored in the local directory. It uses Gauss-Jordan Elimination for inversion and stores the result in `./output`.

This assumes that: 
- $A$ is square and non-singular 
- there exists an Identity matrix $I$ such that $A A^{-1} = A^{-1} A = I$


In order to find the inverse of the matrix following steps need to be followed:  

1. Form the augmented matrix $(A \Vert I)$ 
2. Perform the row reduction operation on this augmented matrix to generate a row reduced echelon form of the matrix.
3. The following row operations are performed on augmented matrix when required: 
   - Interchange any two row.
   - Multiply each element of row by a non-zero integer.
   - Replace a row by the sum of itself and a constant multiple of another row of the matrix.

### Algorithm
See [1]

```
Read matrix
Initialize n to size of matrix
Initialize j to 0

while j<n {
    Find k where matrix[k][j] != 0
    
    for all threads i of n in parallel {
        matrix[j][i] = matrix[j][i] + matrix[k][i]
    }

    for all threads i of n in parallel {
        matrix[j][i] = matrix[j][i] / matrix[j][j]
    }

    for all r blocks of n {
        for all threads i of n in parallel {
            matrix[i][r] = matrix[i][r] - matrix[i][j] * matrix[j][r]
        }
    }

    j++
}

```

## References
[1] Sharma, Girish & Agarwala, Abhishek & Bhattacharya, Baidurya. (2013). A fast parallel Gauss Jordan algorithm for matrix inversion using CUDA. Computers & Structures. 128. 31–37. 10.1016/j.compstruc.2013.06.015. 
