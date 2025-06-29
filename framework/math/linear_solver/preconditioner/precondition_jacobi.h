#pragma once
#include "framework/math/linear_solver/preconditioner/preconditioner.h"
#include "framework/math/matrix.h"
#include "framework/math/linear_solver/util.h"


namespace pdes
{
  template<typename Number = double>
  class PreconditionJacobi final : public Preconditioner<Number>
  {
  public:
    PreconditionJacobi() = default;
    explicit PreconditionJacobi(const Matrix<>* A);

    void vmult(const Vector<Number>& src, Vector<Number>& dst) const override;

    std::string name() const override { return "PreconditionJacobi"; }

  private:
    const Matrix<Number>* A_;
    Vector<Number> inv_diag_;
  };

  /*-------------------- inline functions --------------------*/

  template<typename Number>
  PreconditionJacobi<Number>::PreconditionJacobi(const Matrix<>* A)
    : A_(A)
  {
    if (not A_->is_square())
      throw std::invalid_argument(name() + ": matrix must be square.");
    inv_diag_ = internal::extract_inv_diagonal(*A_, name());
  }

  template<typename Number>
  void
  PreconditionJacobi<Number>::vmult(const Vector<Number>& src, Vector<Number>& dst) const
  {
    if (not A_)
      throw std::runtime_error(name() + " not initialized");
    if (inv_diag_.size() != src.size())
      throw std::runtime_error(name() + ": dimension mismatch error");

    dst.resize(src.size());
    for (size_t i = 0; i < src.size(); ++i)
      dst(i) = inv_diag_[i] * src(i);
  }
}
