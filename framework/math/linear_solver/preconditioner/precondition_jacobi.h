#pragma once
#include "framework/math/matrix.h"
#include "framework/math/linear_solver/util.h"


namespace pdes
{
  /**
   * @brief Jacobi (diagonal) preconditioner.
   *
   * Applies an inverse diagonal scaling to the input vector. Requires the matrix
   * to be square and non-singular.
   *
   * @tparam MatrixType Type of matrix to precondition (default: Matrix<>).
   */
  template<typename MatrixType = Matrix<>>
  class PreconditionJacobi final : public Preconditioner<MatrixType>
  {
  public:
    using value_type = typename Preconditioner<MatrixType>::value_type;
    using VectorType = typename Preconditioner<MatrixType>::VectorType;

    void build(const MatrixType* A) override;

    /// Applies the preconditioner: z = Ainv r.
    void vmult(const VectorType& src, VectorType& dst) const override;

    /// Returns the name of the preconditioner.
    std::string name() const override { return "PreconditionJacobi"; }

  private:
    const MatrixType* A_ = nullptr;
    std::vector<value_type> inv_diag_;
  };

  /*-------------------- member functions --------------------*/

  template<typename MatrixType>
  void
  PreconditionJacobi<MatrixType>::build(const MatrixType* A)
  {
    A_ = A;
    if (not A_->is_square())
      throw std::invalid_argument(name() + ": matrix must be square.");
    inv_diag_ = internal::extract_inv_diagonal(*A_, name());
  }

  template<typename MatrixType>
  void
  PreconditionJacobi<MatrixType>::vmult(const VectorType& src, VectorType& dst) const
  {
    if (not A_)
      throw std::runtime_error(name() + " not initialized");
    if (inv_diag_.size() != src.size())
      throw std::runtime_error(name() + ": dimension mismatch error");

    dst.reinit(src.size());
    for (size_t i = 0; i < src.size(); ++i)
      dst(i) = inv_diag_[i] * src(i);
  }
}
