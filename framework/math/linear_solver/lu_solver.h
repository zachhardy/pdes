#pragma once
#include "framework/math/matrix.h"
#include <string>

namespace pdes
{
  /**
    * @brief LU factorization solver with partial pivoting.
    *
    * This solver performs LU decomposition of a square matrix A such that A = LU,
    * where L is lower triangular with unit diagonal and U is upper triangular.
    * It supports in-place solving with optional factor caching.
    *
    * @tparam MatrixType The matrix type to factor (default: Matrix<>).
    */
  template<typename MatrixType = Matrix<>>
  class LUSolver
  {
  public:
    using value_type = typename MatrixType::value_type;

    /// Constructs an empty LU solver.
    LUSolver() = default;

    /// Performs LU factorization on matrix A.
    void factor(const MatrixType& A);

    /**
     * Solves Ax = b by factoring A and performing forward/backward substitution.
     *
     * @param A The system matrix.
     * @param b The right-hand side.
     * @param x The resulting solution vector.
     */
    template<typename VectorType>
    void solve(const MatrixType& A, const VectorType& b, VectorType& x);

    /**
     * Solves LUx = b using pre-factored matrix.
     *
     * @param b The right-hand side.
     * @param x The resulting solution vector.
     */
    template<typename VectorType>
    void solve(const VectorType& b, VectorType& x) const;

    /// Returns the name of the solver.
    static std::string name() { return "LUSolver"; }

  private:
    MatrixType L_, U_;
    std::vector<size_t> pivot_;
    bool factored_ = false;
  };

  /*-------------------- inline functions --------------------*/

  template<typename MatrixType>
  void
  LUSolver<MatrixType>::factor(const MatrixType& A)
  {
    const size_t n = A.m();
    if (A.m() != A.n())
      throw std::invalid_argument(name() + ": matrix must be square.");

    U_ = A;
    L_ = MatrixType(n, n, value_type(0));
    pivot_.resize(n);
    std::iota(pivot_.begin(), pivot_.end(), 0);

    for (size_t k = 0; k < n; ++k)
    {
      // Find pivot row
      auto max_row = k;
      auto max_val = std::abs(U_(pivot_[k], k));
      for (size_t i = k + 1; i < n; ++i)
      {
        if (auto val = std::abs(U_(pivot_[i], k)); val > max_val)
        {
          max_val = val;
          max_row = i;
        }
      }

      if (max_val == value_type(0))
        throw std::runtime_error(name() + ": singular matrix in LU factorization.");

      // Swap rows in pivot vector
      std::swap(pivot_[k], pivot_[max_row]);

      // Eliminate below pivot
      for (size_t i = k + 1; i < n; ++i)
      {
        const size_t row_i = pivot_[i];
        const size_t row_k = pivot_[k];

        const auto factor = U_(row_i, k) / U_(row_k, k);
        L_(row_i, k) = factor;

        for (size_t j = k; j < n; ++j)
          U_(row_i, j) -= factor * U_(row_k, j);
      }
    }

    // Fill diagonal of L
    for (size_t i = 0; i < n; ++i)
      L_(i, i) = 1.0;

    factored_ = true;
  }

  template<typename MatrixType>
  template<typename VectorType>
  void
  LUSolver<MatrixType>::solve(const MatrixType& A, const VectorType& b, VectorType& x)
  {
    factor(A);
    solve(b, x);
  }

  template<typename MatrixType>
  template<typename VectorType>
  void
  LUSolver<MatrixType>::solve(const VectorType& b, VectorType& x) const
  {
    if (not factored_)
      throw std::runtime_error(name() + ": matrix not factorized");
    if (b.size() != L_.m())
      throw std::invalid_argument(name() + ": size of b does not match factorized system");

    const size_t n = L_.m();

    // Step 1: Permute right-hand side
    VectorType b_perm(n);
    for (size_t i = 0; i < n; ++i)
      b_perm(i) = b(pivot_[i]);

    // Step 2: Forward substitution Ly = b_perm
    VectorType y(n);
    for (size_t i = 0; i < n; ++i)
    {
      y(i) = b_perm(i);
      for (size_t j = 0; j < i; ++j)
        y(i) -= L_(i, j) * y(j);
    }

    // Step 3: Backward substitution Ux = y
    x.resize(n, 0.0);
    for (ssize_t i = static_cast<ssize_t>(n) - 1; i >= 0; --i)
    {
      x(i) = y(i);
      for (size_t j = i + 1; j < n; ++j)
        x(i) -= U_(i, j) * x(j);
      x(i) /= U_(i, i);
    }
  }
}
