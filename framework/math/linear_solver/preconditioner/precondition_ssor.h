#pragma once
#include "framework/math/linear_solver/preconditioner/preconditioner.h"
#include "framework/math/matrix.h"
#include "framework/math/linear_solver/util.h"

namespace pdes
{
  /**
   * Symmetric Successive Over-Relaxation (SSOR) preconditioner.
   *
   * Applies one forward and one backward sweep using a relaxation factor \f$ \omega \in (0, 2) \f$.
   * Approximates the inverse of A for use in iterative solvers.
   *
   * @tparam MatrixType Type of matrix to precondition (default: Matrix<>).
   */
  template<typename MatrixType = Matrix<>>
  class PreconditionSSOR final : public Preconditioner<MatrixType>
  {
  public:
    using value_type = typename Preconditioner<MatrixType>::value_type;
    using VectorType = typename Preconditioner<MatrixType>::VectorType;

    /// Constructs the preconditioner using matrix A and relaxation factor omega.
    explicit PreconditionSSOR(value_type omega = 1.3);

    void build(const MatrixType* A) override;

    /// Applies the preconditioner: z = Ainv r.
    void vmult(const VectorType& src, VectorType& dst) const override;

    /// Returns the name of the preconditioner.
    std::string name() const override { return "PreconditionSSOR"; }

  private:
    const MatrixType* A_ = nullptr;
    std::vector<value_type> inv_diag_;
    value_type omega_;
  };

  /*-------------------- member functions --------------------*/

  template<typename MatrixType>
  PreconditionSSOR<MatrixType>::PreconditionSSOR(value_type omega)
    : omega_(omega)
  {
    if (omega_ <= 0.0 or omega_ >= 2.0)
      throw std::invalid_argument(name() + ": SSOR omega must be in (0, 2)");
  }


  template<typename MatrixType>
  void
  PreconditionSSOR<MatrixType>::build(const MatrixType* A)
  {
    A_ = A;
    if (not A_->is_square())
      throw std::invalid_argument(name() + ": matrix must be square.");
    inv_diag_ = internal::extract_inv_diagonal(*A_, name());
  }

  template<typename MatrixType>
  void
  PreconditionSSOR<MatrixType>::vmult(const VectorType& src, VectorType& dst) const
  {
    if (not A_)
      throw std::runtime_error(name() + " not initialized");
    if (src.size() != inv_diag_.size())
      throw std::runtime_error(name() + ": size mismatch");

    const auto n = src.size();
    Vector<value_type> y(n, value_type(0));
    dst.reinit(n, value_type(0));

    // Forward solve: (D + \omega L) y = r
    for (size_t i = 0; i < n; ++i)
    {
      value_type sum = 0;
      for (const auto& [j, aij]: A_->row_entries(i))
        sum += aij * y(j); // lower triangle only

      y(i) = (src(i) - omega_ * sum) * inv_diag_[i];
    }

    // Backward solve: (D + \omega L)^T z = D y
    for (size_t i = n; i-- > 0;)
    {
      value_type sum = 0;
      for (const auto& [j, aij]: A_->row_entries(i))
        sum += aij * dst(j); // upper triangle of transpose(L)

      const auto rhs = inv_diag_[i] * y(i);
      dst(i) = (rhs - omega_ * sum) * inv_diag_[i];
    }
  }
}
