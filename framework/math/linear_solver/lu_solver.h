#pragma once
#include "framework/math/matrix.h"
#include <string>

namespace pdes
{
  /**
    * LU factorization solver with partial pivoting.
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

    /// Solves Ax = b by factoring A and performing forward/backward substitution.
    template<typename VectorType>
    void solve(const MatrixType& A, const VectorType& b, VectorType& x);

    /// Solves LUx = b using pre-factored matrix.
    template<typename VectorType>
    void solve(const VectorType& b, VectorType& x) const;

    /// Returns the name of the solver.
    static std::string name() { return "LUSolver"; }

  private:
    MatrixType LU_;
    std::vector<size_t> pivot_;
    bool factored_ = false;
  };

  /*-------------------- inline functions --------------------*/

  template<typename MatrixType>
  void
  LUSolver<MatrixType>::factor(const MatrixType& A)
  {
    const auto n = A.m();
    if (A.m() != A.n())
      throw std::invalid_argument(name() + ": matrix must be square");

    LU_ = A;
    pivot_.resize(n);
    std::iota(pivot_.begin(), pivot_.end(), 0);

    for (size_t k = 0; k < n; ++k)
    {
      // Find pivot row
      auto pivot_row = k;
      auto max_val = std::abs(LU_(pivot_[k], k));
      for (size_t i = k + 1; i < n; ++i)
      {
        auto val = std::abs(LU_(pivot_[i], k));
        if (val > max_val)
        {
          max_val = val;
          pivot_row = i;
        }
      }

      if (max_val == value_type(0))
        throw std::runtime_error(name() + ": singular matrix in LU factorization");

      std::swap(pivot_[k], pivot_[pivot_row]);

      const auto pk = pivot_[k];
      for (size_t i = k + 1; i < n; ++i)
      {
        const auto pi = pivot_[i];
        LU_(pi, k) /= LU_(pk, k);
        for (size_t j = k + 1; j < n; ++j)
          LU_(pi, j) -= LU_(pi, k) * LU_(pk, j);
      }
    }

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
      throw std::logic_error(name() + ": solve() called before factor()");

    const auto n = LU_.m();
    if (b.size() != n)
      throw std::invalid_argument(name() + ": dimension mismatch");

    // Permute b
    VectorType b_perm(n);
    for (size_t i = 0; i < n; ++i)
      b_perm(i) = b(pivot_[i]);

    // Forward substitution: solve Ly = Pb
    VectorType y(n);
    for (size_t i = 0; i < n; ++i)
    {
      y(i) = b_perm(i);
      for (size_t j = 0; j < i; ++j)
        y(i) -= LU_(pivot_[i], j) * y(j);
    }

    // Backward substitution: solve Ux = y
    VectorType x_pivoted(n);
    for (int i = static_cast<int>(n) - 1; i >= 0; --i)
    {
      x_pivoted(i) = y(i);
      for (size_t j = i + 1; j < n; ++j)
        x_pivoted(i) -= LU_(pivot_[i], j) * x_pivoted(j);
      x_pivoted(i) /= LU_(pivot_[i], i);
    }

    // Undo pivoting to get solution x in original order
    x.reinit(n);
    for (size_t i = 0; i < n; ++i)
      x(pivot_[i]) = x_pivoted(i);
  }
}
