# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch

@torch.jit.script
def unravel_index(index: int, shape: torch.Tensor):
    """
    Unravel the index input to fit the given shape.
    This function is needed for torch.jit.script compatibility.

    Args:
        index (int): The index to unravel.
        shape (Tesnor): The shape to unravel the index to.

    Returns:
        Tensor: The unraveled index.
    """
    out = []
    shape = torch.flip(shape, dims=(0,))
    for dim in shape:
        out.append(index % dim)
        index = index // dim
    out = torch.tensor([ int(x.item()) for x in out ])
    return torch.flip(out, dims=(0,))

@torch.jit.script
class LinearSumAssignmentSolver(object):
    """
    Solver for the linear sum assignment problem. Created for torch.jit.script compatibility in NeMo.
    The linear sum assignment (LSA) problem is also referred to as bipartite matching problem. An LSA
    problem is described by a matrix `cost_mat`, where each cost_mat[i,j] is the cost of matching 
    vertex i of the first partite set (e.g. a "worker") and vertex j of the second set (e.g. a "job"). 
    Thus, the goal of LSA-solver is to find a complete assignment of column element to row element with 
    the minimal cost. Note that the solution may not be unique and there could be multiple solutions that
    yield the same minimal cost.

    This implementation is based on the LAP solver from scipy: 
        https://github.com/scipy/scipy/blob/v0.18.1/scipy/optimize/_hungarian.py.

    References
        1. http://csclab.murraystate.edu/bob.pilgrim/445/munkres.html
        2. https://en.wikipedia.org/wiki/Hungarian_algorithm
        3. https://github.com/scipy/scipy/blob/v0.18.1/scipy/optimize/_hungarian.py

    Attributes:
        cost_mat : 2D matrix containing cost matrix. Number of columns must be larger than number of rows.
        row_uncovered : 1D matrix containing boolean values indicating whether a row is covered.
        col_uncovered : 1D matrix containing boolean values indicating whether a column is covered.
        zero_row : 1D matrix containing the row index of the last zero found.
        zero_col : 1D matrix containing the column index of the last zero found.
        path : 2D matrix containing the path taken through the matrix.
        marked : 2D matrix containing the marked zeros.
    """
    def __init__(self, cost_matrix: torch.Tensor):
        """
        Initialize the solver with the given cost matrix.

        Args:
            cost_matrix (Tensor): 2D matrix containing cost matrix. 
                                  Number of columns must be larger than number of rows.
        """
        self.cost_mat = cost_matrix
        row_len, col_len = self.cost_mat.shape
        self.zero_row = torch.tensor(0, dtype=torch.long).to(cost_matrix.device)
        self.zero_col = torch.tensor(0, dtype=torch.long).to(cost_matrix.device)
        self.row_uncovered = torch.ones(row_len, dtype=torch.bool).to(cost_matrix.device)
        self.col_uncovered = torch.ones(col_len, dtype=torch.bool).to(cost_matrix.device)
        self.path = torch.zeros((row_len+col_len, 2), dtype=torch.long).to(cost_matrix.device)
        self.marked = torch.zeros((row_len, col_len), dtype=torch.long).to(cost_matrix.device)

    def _reset_uncovered_mat(self):
        """
        Reset all uncovered matrix elements. Assign `True` to all uncovered elements.
        """
        self.row_uncovered[:] = True
        self.col_uncovered[:] = True

    def _step1(self):
        """
        Step 1

        For each row of the matrix, find the smallest element and
        subtract it from every element in its row.
        """
        self.cost_mat -= torch.min(self.cost_mat, dim=1)[0].unsqueeze(1)

    def _step2(self):
        """
        Step 2

        - Find a zero in the resulting matrix. 
        - If there is no marked zero in  its row or column, 
          mark the zero. Repeat for each element in the matrix.
        """
        ind_out = torch.where(self.cost_mat == 0)
        ind, val = list(ind_out[0]), list(ind_out[1])
        for i, j in zip(ind, val):
            if self.col_uncovered[j] and self.row_uncovered[i]:
                self.marked[i, j] = 1
                self.col_uncovered[j] = False
                self.row_uncovered[i] = False

        self._reset_uncovered_mat()

    def _sub_n_find_zeros(self) -> int:
        """
        This function is performing step 1 and step 2 in the following wikipedia article:
        https://en.wikipedia.org/wiki/Hungarian_algorithm

        """
        self._step1()
        self._step2()
        return 3

    def _step3(self) -> int:
        """
        Step 3

        - Cover each column containing a marked zero. 
        - If n columns are covered,
            the marked zeros describe a complete set of unique assignments.
            In this case, Go to DONE (0 state)
        - Otherwise, Go to Step 4.
        """
        marked = (self.marked == 1)
        self.col_uncovered[torch.any(marked, dim=0)] = False
        if marked.sum() < self.cost_mat.shape[0]:
            return 4
        else:
            return 0

    def _step4(self) -> int:
        """
        Step 4

        Find a noncovered zero and prime it. If there is no marked zero
        in the row containing this primed zero, Go to Step 5. Otherwise,
        cover this row and uncover the column containing the marked
        zero. Continue in this manner until there are no uncovered zeros
        left. Save the smallest uncovered value and Go to Step 6.
        """
        # We convert to int as numpy operations are faster on int
        cost_mat = (self.cost_mat == 0).int()
        covered_cost_mat = cost_mat * self.row_uncovered.unsqueeze(1)
        covered_cost_mat *= self.col_uncovered.long()
        n = self.cost_mat.shape[0]
        m = self.cost_mat.shape[1]

        while True:
            urv = unravel_index(torch.argmax(covered_cost_mat).item(), torch.tensor([m, n]))
            row, col = int(urv[0].item()), int(urv[1].item())
            if covered_cost_mat[row, col] == 0:
                return 6
            else:
                self.marked[row, col] = 2
                # Find the first marked element in the row
                mark_col = torch.argmax((self.marked[row] == 1).int())
                if self.marked[row, mark_col] != 1:
                    # Could not find one
                    self.zero_row = torch.tensor(row)
                    self.zero_col = torch.tensor(col)
                    return 5
                else:
                    col = mark_col
                    self.row_uncovered[row] = False
                    self.col_uncovered[col] = True
                    covered_cost_mat[:, col] = C[:, col] * self.row_uncovered
                    covered_cost_mat[row] = 0
        return 0

    def _step5(self) -> int:
        """
        Step 5

        Construct a series of alternating primed and marked zeros as follows.
        - Let Z0 represent the uncovered primed zero found in Step 4.
        - Let Z1 denote the marked zero in the column of Z0 (if any).
        - Let Z2 denote the primed zero in the row of Z1 (there will always be one).
        - Continue until the series terminates at a primed zero that has no marked
        zero in its column. Unmark each marked zero of the series, mark each
        primed zero of the series, erase all primes and uncover every line in the
        matrix. Return to Step 3
        """
        count = torch.tensor(0)
        path = self.path
        path[count, 0] = self.zero_row.long()
        path[count, 1] = self.zero_col.long()

        while True:
            # Find the first marked element in the col defined by the path.
            row = torch.argmax((self.marked[:, path[count, 1]] == 1).int())

            if self.marked[row, path[count, 1]] != 1:
                # Could not find one
                break
            else:
                count += 1
                path[count, 0] = row
                path[count, 1] = path[count - 1, 1]

            # Find the first prime element in the row defined by the first path step
            col = int(torch.argmax((self.marked[path[count, 0]] == 2).int()))
            if self.marked[row, col] != 2:
                col = -1
            count += 1
            path[count, 0] = path[count - 1, 0]
            path[count, 1] = col

        # Convert paths
        for i in range(int(count.item())+ 1):
            if self.marked[path[i, 0], path[i, 1]] == 1:
                self.marked[path[i, 0], path[i, 1]] = 0
            else:
                self.marked[path[i, 0], path[i, 1]] = 1

        self._reset_uncovered_mat()
        
        # Remove all prime markings in marked matrix
        self.marked[self.marked == 2] = 0
        return 3

    def _step6(self) -> int:
        """
        Step 6

        - Add the value found in Step 4 to every element of each covered row.
        - Subtract it from every element of each uncovered column.
        - Return to Step 4 without altering any marks, primes, or covered lines.
        """
        # the smallest uncovered value in the matrix
        if torch.any(self.row_uncovered) and torch.any(self.col_uncovered):
            row_minval = torch.min(self.cost_mat[self.row_uncovered], dim=0)[0]
            minval = torch.min(row_minval[self.col_uncovered])
            self.cost_mat[~self.row_uncovered] += minval
            self.cost_mat[:, self.col_uncovered] -= minval
        return 4

@torch.jit.script
def linear_sum_assignment__(cost_matrix):
    """
    Launcher function for the linear sum assignment problem solver.

    Args:
        cost_matrix (Tensor): The cost matrix of shape (N, M) 
    """

    cost_matrix = cost_matrix.clone().detach()

    if len(cost_matrix.shape) != 2:
        raise ValueError("2-d tensor is expected but got a {cost_matrix.shape} tensor" )

    # The algorithm expects more columns than rows in the cost matrix.
    if cost_matrix.shape[1] < cost_matrix.shape[0]:
        cost_matrix = cost_matrix.T
        transposed = True
    else:
        transposed = False

    lap_solver = LinearSumAssignmentSolver(cost_matrix)
    f_str: int = 0  if 0 in cost_matrix.shape else 1

    import ipdb; ipdb.set_trace() 
    # while step is not None:
    while f_str != 0:
        if f_str == 1:
            f_str = lap_solver._step1()
        elif f_str == 2:
            f_str = lap_solver._step2()
        elif f_str == 3:
            f_str = lap_solver._step3()
        elif f_str == 4:
            f_str = lap_solver._step4()
        elif f_str == 5:
            f_str = lap_solver._step5()
        elif f_str == 6:
            f_str = lap_solver._step6()
    
    if transposed:
        marked = lap_solver.marked.T
    else:
        marked = lap_solver.marked
    return torch.where(marked == 1)




@torch.jit.script
def linear_sum_assignment(cost_matrix):
    """
    Launcher function for the linear sum assignment problem solver.

    Args:
        cost_matrix (Tensor): The cost matrix of shape (N, M) 
    """

    cost_matrix = cost_matrix.clone().detach()

    if len(cost_matrix.shape) != 2:
        raise ValueError("2-d tensor is expected but got a {cost_matrix.shape} tensor" )

    # The algorithm expects more columns than rows in the cost matrix.
    if cost_matrix.shape[1] < cost_matrix.shape[0]:
        cost_matrix = cost_matrix.T
        transposed = True
    else:
        transposed = False

    lap_solver = LinearSumAssignmentSolver(cost_matrix)
    f_str: int = 0  if 0 in cost_matrix.shape else 1

    # while step is not None:
    while f_str != 0:
        if f_str == 1:
            f_str = lap_solver._step1()
        elif f_str == 2:
            f_str = lap_solver._step2()
        elif f_str == 3:
            f_str = lap_solver._step3()
        elif f_str == 4:
            f_str = lap_solver._step4()
        elif f_str == 5:
            f_str = lap_solver._step5()
        elif f_str == 6:
            f_str = lap_solver._step6()
    
    if transposed:
        marked = lap_solver.marked.T
    else:
        marked = lap_solver.marked
    return torch.where(marked == 1)


