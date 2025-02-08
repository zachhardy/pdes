#pragma once

#include "framework/math/ndarray.h"

namespace pdes
{
  template<typename Number = double>
  class Vector : public NDArray<1, Number>
  {
  public:
    /// Constructs a Vector with the given @p size set to @p value.
    explicit Vector(const size_t n, const Number value = 0.0)
      : NDArray<1, Number>({n}, value) {}

    /// Constructs a size @p n vector and set the entries with raw data.
    explicit Vector(const size_t n, const Number* ptr)
      : NDArray<1, Number>({n}, ptr) {}

    /// Set all entries of the Vector to the given @p value.
    Vector& operator=(Number value);

    /// Resizes the Vector to the given @p size and set all entries to zero.
    void resize(const size_t size) { resize(size, 0.0); }

    /// Resizes the Vector to the given @p size and set all entries to @p value.
    void resize(size_t size, Number value);

    /// Returns a reference to the entry @p i.
    Number& operator[](const size_t i) { return this->at(i); }
    /// Returns the value of entry @p i.
    Number operator[](const size_t i) const { return this->at(i); }

    /// Returns a reference to the entry @p i.
    Number& operator()(const size_t i) { return this->at(i); }
    /// Returns the value of entry @p i.
    Number operator()(const size_t i) const { return this->at(i); }

    /// Sets entry @p i to @p value.
    void set(const size_t i, const Number value) { this->at(i) = value; }
    /// Sets the entries with the given @p indices to the given @p values.
    void set(const std::vector<size_t>& indices,
             const std::vector<Number>& values);
    /// Sets @p n entries at the given @p indices to the given @p values.
    void set(size_t n, const size_t* indices, const Number* values);

    /// Adds the given @p value to all entries.
    void add(Number value);
    /// Adds the given @p value to entry @p i.
    void add(size_t i, Number value) { this->at(i) += value; }

    /// Adds the given @p values to the entries at the given @p indices.
    void add(const std::vector<size_t>& indices,
             const std::vector<Number>& values);
    /// Adds the given @p n @p values to the given @p indices.
    void add(size_t n, const size_t* indices, const Number* values);

    /// Returns the minimum-valued entry.
    Number min() const;
    /// Returns the maximum-valued entry.
    Number max() const;
    /// Returns the sum of the entries.
    Number sum() const;
    /// Returns the mean of the entries.
    double mean() const;

    /// Returns the @f$ \ell_\infty @f$-norm of the Vector.
    Number linfty_norm() const;
    /// Returns the @f$ \ell_1 @f$-norm of the Vector.
    Number l1_norm() const;
    /// Returns the @f$ \ell_2 @f$-norm of the Vector.
    double l2_norm() const;
    /// Returns the @f$ \ell_p @f$-norm of the Vector.
    double lp_norm(double p) const;

    /// Returns the @f$ \ell_2 @f$-norm squared of the Vector.
    double norm_sqr() const;

    /// Returns the dot product with another Vector.
    Number dot(const Vector& other) const;

    /// Prints the Vector to an output stream
    void print(std::ostream& os = std::cout,
               unsigned int precision = 3,
               bool scientific = true,
               bool across = true) const;
  };

  /* -------------------- inline functions --------------------*/

  template<typename Number>
  Vector<Number>&
  Vector<Number>::operator=(Number value)
  {
    this->set(value);
    return *this;
  }

  template<typename Number>
  void
  Vector<Number>::resize(const size_t size, const Number value)
  {
    this->reshape(size);
    this->set(value);
  }

  template<typename Number>
  void
  Vector<Number>::set(const std::vector<size_t>& indices,
                      const std::vector<Number>& values)
  {
    if (indices.size() != values.size())
      throw std::invalid_argument("Vector size mismatch (indices, values).");

    this->set(indices.size(), indices.data(), values.data());
  }

  template<typename Number>
  void
  Vector<Number>::set(const size_t n,
                      const size_t* indices,
                      const Number* values)
  {
    for (size_t i = 0; i < n; ++i)
      this->set(indices[i], values[i]);
  }

  template<typename Number>
  void
  Vector<Number>::add(const Number value)
  {
    auto func = [value](Number x) { return value + x; };
    std::transform(this->cbegin(), this->cend(), this->cbegin(), func);
  }

  template<typename Number>
  void
  Vector<Number>::add(const std::vector<size_t>& indices,
                      const std::vector<Number>& values)
  {
    if (indices.size() != values.size())
      throw std::invalid_argument("Incompatible indices and values sizes.");
    this->add(indices.size(), indices.data(), values.data());
  }

  template<typename Number>
  void
  Vector<Number>::add(const size_t n,
                      const size_t* indices,
                      const Number* values)
  {
    for (size_t i = 0; i < n; ++i)
      this->add(indices[i], values[i]);
  }

  template<typename Number>
  Number
  Vector<Number>::min() const
  {
    return *std::min_element(this->cbegin(), this->cend());
  }

  template<typename Number>
  Number
  min(const Vector<Number>& vec)
  {
    return vec.min();
  }

  template<typename Number>
  Number
  Vector<Number>::max() const
  {
    return *std::max_element(this->cbegin(), this->cend());
  }

  template<typename Number>
  Number
  max(const Vector<Number>& vec)
  {
    return vec.max();
  }

  template<typename Number>
  Number
  Vector<Number>::sum() const
  {
    auto func = std::plus<Number>();
    return std::accumulate(this->cbegin(), this->cend(), Number(0), func);
  }

  template<typename Number>
  Number
  sum(const Vector<Number>& vec)
  {
    return vec.sum();
  }

  template<typename Number>
  double
  Vector<Number>::mean() const
  {
    return this->sum() / static_cast<double>(this->size());
  }

  template<typename Number>
  double
  mean(const Vector<Number>& vec)
  {
    return vec.mean();
  }

  template<typename Number>
  Number
  Vector<Number>::linfty_norm() const
  {
    auto func = [](Number max, Number x) {
      return std::fabs(max) < std::fabs(x);
    };
    return *std::max_element(this->cbegin(), this->cend(), func);
  }

  template<typename Number>
  Number
  linfty_norm(const Vector<Number>& vec)
  {
    return vec.linfty_norm();
  }

  template<typename Number>
  Number
  Vector<Number>::l1_norm() const
  {
    auto func = [](Number norm, Number x) { return norm + std::fabs(x); };
    return std::accumulate(this->cbegin(), this->cend(), Number(0), func);
  }

  template<typename Number>
  Number
  l1_norm(const Vector<Number>& vec)
  {
    return vec.l1_norm();
  }

  template<typename Number>
  double
  Vector<Number>::l2_norm() const
  {
    return std::sqrt(this->norm_sqr());
  }

  template<typename Number>
  double
  l2_norm(const Vector<Number>& vec)
  {
    return vec.l2_norm();
  }

  template<typename Number>
  double
  Vector<Number>::lp_norm(const double p) const
  {
    auto func = [p](Number norm, Number x) {
      return norm + std::pow(std::fabs(x), p);
    };
    const auto tmp = std::accumulate(this->cbegin(), this->cend(), Number(0),
                                     func);
    return std::pow(tmp, 1.0 / p);
  }

  template<typename Number>
  double
  lp_norm(const Vector<Number>& vec, const double p)
  {
    return vec.lp_norm(p);
  }

  template<typename Number>
  double
  Vector<Number>::norm_sqr() const
  {
    return std::inner_product(this->cbegin(), this->cend(), this->cbegin(),
                              Number(0), Number(0));
  }

  template<typename Number>
  double
  norm_sqr(const Vector<Number>& vec)
  {
    return vec.norm_sqr();
  }

  template<typename Number>
  Number
  Vector<Number>::dot(const Vector& other) const
  {
    if (this->size() != other.size())
      throw std::invalid_argument("Vector size mismatch.");

    return std::inner_product(this->cbegin(),
                              this->cend(),
                              other.cbegin(),
                              Number(0));
  }

  template<typename Number>
  Number
  dot(const Vector<Number>& vec1, const Vector<Number>& vec2)
  {
    return vec1.dot(vec2);
  }

  template<typename Number>
  void
  Vector<Number>::print(std::ostream& os,
                        const unsigned int precision,
                        const bool scientific,
                        const bool across) const
  {
    const auto old_flags = os.flags();
    const auto old_precision = os.precision(precision);

    os.precision(precision);
    if (scientific)
      os.setf(std::ios::scientific, std::ios::floatfield);
    else
      os.setf(std::ios::fixed, std::ios::floatfield);

    if (across)
      for (size_t i = 0; i < this->size(); ++i)
        os << (i == 0 ? "[" : "")
            << this->at(i)
            << (i == this->size() - 1 ? "]" : " ");
    else
      for (unsigned int i = 0; i < this->size(); ++i)
        os << (i == 0 ? "[" : "")
            << this->at(i)
            << (i == this->size() - 1 ? "]" : "")
            << std::endl;
    os << std::endl;

    os.flags(old_flags);
    os.precision(old_precision);
  }
}
