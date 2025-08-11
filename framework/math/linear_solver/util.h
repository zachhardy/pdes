#pragma once
#include "framework/math/matrix.h"

namespace pdes
{
  namespace internal
  {
    /**
     * Computes the inverse of diagonal entries from a square matrix.
     *
     * This function is used in iterative solvers to construct simple
     * preconditioners based on diagonal scaling. Throws if the matrix is not
     * square or contains a zero on the diagonal.
     *
     * @tparam MatrixType The matrix type being used..
     * @param matrix A square matrix from which to extract the diagonal.
     * @param label Descriptive label used in exception messages.
     * @return Vector containing 1 / diagonal(i) for each row.
     */
    template<typename MatrixType>
    std::vector<typename MatrixType::value_type>
    extract_inv_diagonal(const MatrixType& matrix, const std::string& label)
    {
      using value_type = typename MatrixType::value_type;

      if (matrix.m() != matrix.n())
        throw std::invalid_argument(label + ": Matrix must be square.");

      std::vector<value_type> result;
      for (size_t i = 0; i < matrix.m(); ++i)
      {
        const auto diag = matrix.el(i, i);
        if (diag == value_type(0))
          throw std::runtime_error(label + ": Zero diagonal in row " + std::to_string(i) + ".");
        result.push_back(value_type(1) / diag);
      }
      return result;
    }
  }
}
