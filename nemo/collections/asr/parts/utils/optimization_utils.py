# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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

from scipy.optimize import linear_sum_assignment as scipy_linear_sum_assignment

@torch.jit.script
def unravel_index(index: int, shape):
    """
    Unravel the index input to fit the given shape.
    """
    out = []
    shape = torch.flip(shape, dims=(0,))
    for dim in shape:
        out.append(index % dim)
        index = index // dim
    out = torch.tensor([ int(x.item()) for x in out ])
    return torch.flip(out, dims=(0,))

@torch.jit.script
class LAPsolver(object):
    """
    State of the Hungarian algorithm.

    Parameters
    ----------
    cost_matrix : 2D matrix
            The cost matrix. Must have shape[1] >= shape[0].
    """

    def __init__(self, cost_matrix):
        self.C = cost_matrix

        n, m = self.C.shape
        self.row_uncovered = torch.ones(n, dtype=torch.bool).to(cost_matrix.device)
        self.col_uncovered = torch.ones(m, dtype=torch.bool).to(cost_matrix.device)
        self.Z0_r = torch.tensor(0, dtype=torch.long).to(cost_matrix.device)
        self.Z0_c = torch.tensor(0, dtype=torch.long).to(cost_matrix.device)
        self.path = torch.zeros((n+m, 2), dtype=torch.long).to(cost_matrix.device)
        self.marked = torch.zeros((n, m), dtype=torch.long).to(cost_matrix.device)

    def _reset_uncovered_mat(self):
        """
        Clear all covered matrix cells and assign `True` to all uncovered elements.

        """
        self.row_uncovered[:] = True
        self.col_uncovered[:] = True

    def _sub_n_find_zeros(self) -> int:
        """Steps 1 and 2 in the Wikipedia page.

        Step 1: For each row of the matrix, find the smallest element and
        subtract it from every element in its row.

        Step 2: Find a zero (Z) in the resulting matrix. If there is no
        starred zero in its row or column, star Z. Repeat for each element
        in the matrix.
        """
        self.C -= torch.min(self.C, dim=1)[0].unsqueeze(1)
        ind_out = torch.where(self.C == 0)
        ind, val = list(ind_out[0]), list(ind_out[1])
        for i, j in zip(ind, val):
            if self.col_uncovered[j] and self.row_uncovered[i]:
                self.marked[i, j] = 1
                self.col_uncovered[j] = False
                self.row_uncovered[i] = False

        self._reset_uncovered_mat()
        return 3

    def _step3(self) -> int:
        """
        Step 3

        - Cover each column containing a starred zero. 
        - If n columns are covered,
            the starred zeros describe a complete set of unique assignments.
            A
            In this case, Go to DONE (0 state)
        - Otherwise, Go to Step 4.
        """
        marked = (self.marked == 1)
        self.col_uncovered[torch.any(marked, dim=0)] = False
        if marked.sum() < self.C.shape[0]:
            return 4
        else:
            return 0

    def _step4(self) -> int:
        """
        Step 4

        Find a noncovered zero and prime it. If there is no starred zero
        in the row containing this primed zero, Go to Step 5. Otherwise,
        cover this row and uncover the column containing the starred
        zero. Continue in this manner until there are no uncovered zeros
        left. Save the smallest uncovered value and Go to Step 6.
        """
        # We convert to int as numpy operations are faster on int
        C = (self.C == 0).int()
        covered_C = C * self.row_uncovered.unsqueeze(1)
        covered_C *= self.col_uncovered.long()
        n = self.C.shape[0]
        m = self.C.shape[1]

        while True:
            urv = unravel_index(torch.argmax(covered_C).item(), torch.tensor([m, n]))
            row, col = int(urv[0].item()), int(urv[1].item())
            if covered_C[row, col] == 0:
                return 6
            else:
                self.marked[row, col] = 2
                # Find the first starred element in the row
                star_col = torch.argmax((self.marked[row] == 1).int())
                if self.marked[row, star_col] != 1:
                    # Could not find one
                    self.Z0_r = torch.tensor(row)
                    self.Z0_c = torch.tensor(col)
                    return 5
                else:
                    col = star_col
                    self.row_uncovered[row] = False
                    self.col_uncovered[col] = True
                    covered_C[:, col] = C[:, col] * self.row_uncovered
                    covered_C[row] = 0
        return 0

    def _step5(self) -> int:
        """
        Step 5

        Construct a series of alternating primed and starred zeros as follows.
            - Let Z0 represent the uncovered primed zero found in Step 4.
            - Let Z1 denote the starred zero in the column of Z0 (if any).
            - Let Z2 denote the primed zero in the row of Z1 (there will always be one).
        Continue until the series terminates at a primed zero that has no starred
        zero in its column. Unstar each starred zero of the series, star each
        primed zero of the series, erase all primes and uncover every line in the
        matrix. Return to Step 3
        """
        count = torch.tensor(0)
        path = self.path
        path[count, 0] = self.Z0_r.long()
        path[count, 1] = self.Z0_c.long()

        while True:
            # Find the first starred element in the col defined by the path.
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
        - Return to Step 4 without altering any stars, primes, or covered lines.
        """
        # the smallest uncovered value in the matrix
        if torch.any(self.row_uncovered) and torch.any(self.col_uncovered):
            row_minval = torch.min(self.C[self.row_uncovered], dim=0)[0]
            minval = torch.min(row_minval[self.col_uncovered])
            self.C[~self.row_uncovered] += minval
            self.C[:, self.col_uncovered] -= minval
        return 4

@torch.jit.script
def linear_sum_assignment(cost_matrix):
    cost_matrix = cost_matrix.clone().detach()

    if len(cost_matrix.shape) != 2:
        raise ValueError("2-d tensor is expected but got a {cost_matrix.shape} tensor" )

    # The algorithm expects more columns than rows in the cost matrix.
    if cost_matrix.shape[1] < cost_matrix.shape[0]:
        cost_matrix = cost_matrix.T
        transposed = True
    else:
        transposed = False

    lap_solver = LAPsolver(cost_matrix)
    f_str: int = 0  if 0 in cost_matrix.shape else 1

    # while step is not None:
    while f_str != 0:
        if f_str == 1:
            f_str = lap_solver._sub_n_find_zeros()
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


