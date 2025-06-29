#pragma once
#include "framework/types.h"
#include "framework/math/ndarray.h"
#include <numeric>
#include <string>
#include <sstream>
#include <iomanip>


namespace pdes
{
  template<typename Number = types::real>
  class Vector
  {
  public:
    using value_type = typename NDArray<1, Number>::value_type;

    /// Constructs an empty Vector.
    Vector() = default;

    /// Constructs a Vector with the given @p size set to @p value.
    explicit Vector(const size_t n, const Number value = 0)
      : entries_({n}, value)
    {}

    /// Constructs a size @p n Vector and set the entries with raw data.
    explicit Vector(const size_t n, const Number* ptr)
      : entries_({n}, ptr) {}

    Vector(const Vector& other) = default;
    Vector(Vector&& other) noexcept = default;

    Vector& operator=(const Vector& other) = default;
    Vector& operator=(Vector&& other) noexcept = default;

    /// Sets the Vector to a value.
    Vector& operator=(Number value);

    /// Returns the size of the Vector.
    size_t size() const { return entries_.size(); }

    /// Returns whether the Vector is all zero.
    bool is_zero() const { return entries_.is_zero(); }
    /// Returns whether the Vector is all non-negative.
    bool is_nonnegative() const { return entries_.is_nonnegative(); }
    /// Returns whether the Vector is empty.
    bool empty() const noexcept { return size() == 0; }

    /// Resizes the Vector to the given @p size and set all entries to zero.
    void resize(const size_t size) { resize(size, Number(0)); }
    /// Resizes the Vector to the given @p size and set all entries to @p value.
    void resize(size_t size, Number value);

    /// Returns a reference to the entry @p i.
    Number& operator[](const size_t i) { return entries_.at(i); }
    /// Returns the value of entry @p i.
    Number operator[](const size_t i) const { return entries_.at(i); }

    /// Returns a reference to the entry @p i.
    Number& operator()(const size_t i) { return entries_.at(i); }
    /// Returns the value of entry @p i.
    Number operator()(const size_t i) const { return entries_.at(i); }

    /// Returns an iterator to the start of the Vector.
    Number* begin() { return entries_.begin(); }
    /// Returns a constant iterator to the start of the Vector.
    const Number* begin() const { return entries_.begin(); }

    /// Returns an iterator to the end of the Vector.
    Number* end() { return entries_.end(); }
    /// Returns a constant iterator to the end of the Vector.
    const Number* end() const { return entries_.end(); }

    /// Sets the entire Vector to @p value.
    void set(const Number value) { entries_.set(value); }
    /// Sets entry @p i to @p value.
    void set(const size_t i, const Number value) { entries_.at(i) = value; }
    /// Sets the entries with the given @p indices to the given @p values.
    void set(const std::vector<size_t>& indices,
             const std::vector<Number>& values);
    /// Sets @p n entries at the given @p indices to the given @p values.
    void set(size_t n, const size_t* indices, const Number* values);

    /// Multiplies by a scalar value.
    void scale(Number a);

    /// Multiplies by a scalar value.
    Vector& operator *=(Number a);
    /// Divides by a non-zero scalar value.
    Vector& operator/=(Number a);

    /// Adds the given @p value to all entries.
    void add(Number value);
    /// Adds the given @p value to entry @p i.
    void add(const size_t i, Number value) { entries_.at(i) += value; }

    /// Adds the given @p values to the entries at the given @p indices.
    void add(const std::vector<size_t>& indices,
             const std::vector<Number>& values);
    /// Adds the given @p n @p values to the given @p indices.
    void add(size_t n, const size_t* indices, const Number* values);

    /// Addition of another Vector.
    void add(const Vector& other) { add(Number(1), other); }
    /// Addition of a scaled Vector.
    void add(const Number b, const Vector& other) { sadd(Number(1), b, other); }
    /// Scaling and addition of another Vector.
    void sadd(const Number a, const Vector& other) { sadd(Number(1), a, other); }
    /// Scaling and addition of a scaled Vector.
    void sadd(Number a, Number b, const Vector& other);

    /// Addition of another Vector.
    Vector& operator+=(const Vector& other);
    /// Subtraction of another Vector.
    Vector& operator-=(const Vector& other);

    /// Returns the minimum-valued entry.
    Number min() const;
    /// Returns the maximum-valued entry.
    Number max() const;
    /// Returns the sum of the entries.
    Number sum() const;
    /// Returns the mean of the entries.
    types::real mean() const;

    /// Returns the @f$ \ell_\infty @f$-norm of the Vector.
    Number linfty_norm() const;
    /// Returns the @f$ \ell_1 @f$-norm of the Vector.
    Number l1_norm() const;
    /// Returns the @f$ \ell_2 @f$-norm of the Vector.
    types::real l2_norm() const;
    /// Returns the @f$ \ell_p @f$-norm of the Vector.
    types::real lp_norm(types::real p) const;

    /// Returns the @f$ \ell_2 @f$-norm squared of the Vector.
    types::real norm_sqr() const;

    /// Returns the dot product with another Vector.
    Number dot(const Vector& other) const;

    /// Write the vector as a string.
    std::string to_string(unsigned int precision = 3,
                          bool scientific = false,
                          bool newline = true) const;

  private
  :
    NDArray<1, Number> entries_;
  };

  /* -------------------- inline functions --------------------*/

  template<typename Number>
  Vector<Number>&
  Vector<Number>::operator=(const Number value)
  {
    entries_ = value;
    return *this;
  }


  template<typename Number>
  void
  Vector<Number>::resize(const size_t size, const Number value)
  {
    entries_.reshape({size});
    entries_.set(value);
  }

  template<typename Number>
  void
  Vector<Number>::set(const std::vector<size_t>& indices,
                      const std::vector<Number>& values)
  {
    if (indices.size() != values.size())
      throw std::invalid_argument("Vector size mismatch (indices, values).");

    set(indices.size(), indices.data(), values.data());
  }

  template<typename Number>
  void
  Vector<Number>::set(const size_t n,
                      const size_t* indices,
                      const Number* values)
  {
    for (size_t i = 0; i < n; ++i)
      set(indices[i], values[i]);
  }

  template<typename Number>
  void
  Vector<Number>::scale(const Number a)
  {
    auto func = [a](Number x) { return a * x; };
    std::transform(begin(), end(), begin(), func);
  }

  template<typename Number>
  Vector<Number>&
  Vector<Number>::operator*=(Number a)
  {
    scale(a);
    return *this;
  }

  template<typename Number>
  Vector<Number>&
  Vector<Number>::operator/=(Number a)
  {
    if (a == Number(0))
      throw std::invalid_argument("Division by zero error.");

    scale(Number(1) / a);
    return *this;
  }

  template<typename Number>
  void
  Vector<Number>::add(const Number value)
  {
    auto func = [value](Number x) { return value + x; };
    std::transform(begin(), end(), begin(), func);
  }

  template<typename Number>
  void
  Vector<Number>::add(const std::vector<size_t>& indices,
                      const std::vector<Number>& values)
  {
    if (indices.size() != values.size())
      throw std::invalid_argument("Incompatible indices and values sizes.");
    add(indices.size(), indices.data(), values.data());
  }

  template<typename Number>
  void
  Vector<Number>::add(const size_t n,
                      const size_t* indices,
                      const Number* values)
  {
    for (size_t i = 0; i < n; ++i)
      add(indices[i], values[i]);
  }

  template<typename Number>
  void
  Vector<Number>::sadd(Number a, Number b, const Vector& other)
  {
    if (size() != other.size())
      throw std::invalid_argument("Incompatible Vector sizes.");

    auto func = [a, b](Number x, Number y) { return a * x + b * y; };
    std::transform(begin(), end(), other.begin(), begin(), func);
  }

  template<typename Number>
  Vector<Number>&
  Vector<Number>::operator+=(const Vector& other)
  {
    sadd(Number(1), other);
    return *this;
  }

  template<typename Number>
  Vector<Number>&
  Vector<Number>::operator-=(const Vector& other)
  {
    add(Number(-1), other);
    return *this;
  }


  template<typename Number>
  Number
  Vector<Number>::min() const
  {
    return *std::min_element(begin(), end());
  }

  template<typename Number>
  Number
  Vector<Number>::max() const
  {
    return *std::max_element(begin(), end());
  }

  template<typename Number>
  Number
  Vector<Number>::sum() const
  {
    auto func = std::plus<Number>();
    return std::accumulate(begin(), end(), Number(0), func);
  }

  template<typename Number>
  types::real
  Vector<Number>::mean() const
  {
    return sum() / static_cast<types::real>(size());
  }

  template<typename Number>
  Number
  Vector<Number>::linfty_norm() const
  {
    auto func = [](Number max, Number x) {
      return std::abs(max) < std::abs(x);
    };
    return *std::max_element(begin(), end(), func);
  }

  template<typename Number>
  Number
  Vector<Number>::l1_norm() const
  {
    auto func = [](Number norm, Number x) { return norm + std::fabs(x); };
    return std::accumulate(begin(), end(), Number(0), func);
  }

  template<typename Number>
  types::real
  Vector<Number>::l2_norm() const
  {
    return std::sqrt(norm_sqr());
  }

  template<typename Number>
  types::real
  Vector<Number>::lp_norm(const types::real p) const
  {
    auto func = [p](Number norm, Number x) {
      return norm + std::pow(std::fabs(x), p);
    };
    const auto tmp = std::accumulate(begin(), end(), Number(0), func);
    return std::pow(tmp, Number(1) / p);
  }

  template<typename Number>
  types::real
  Vector<Number>::norm_sqr() const
  {
    return std::inner_product(begin(), end(), begin(), Number(0));
  }

  template<typename Number>
  Number
  Vector<Number>::dot(const Vector& other) const
  {
    if (size() != other.size())
      throw std::invalid_argument("Vector size mismatch.");

    return std::inner_product(begin(),
                              end(),
                              other.begin(),
                              Number(0));
  }

  template<typename Number>
  std::string
  Vector<Number>::to_string(const unsigned int precision,
                            const bool scientific,
                            const bool newline) const
  {
    std::ostringstream oss;
    oss << std::setprecision(precision);
    if (scientific)
      oss << std::scientific;
    else
      oss << std::fixed;

    oss << "[";
    for (size_t i = 0; i < size(); ++i)
    {
      oss << (*this)(i);
      if (i < size() - 1)
        oss << " ";
    }
    oss << "]";
    if (newline)
      oss << '\n';

    return oss.str();
  }

  /* -------------------- free functions --------------------*/

  template<typename Number>
  Vector<Number>
  operator*(Number c, const Vector<Number>& x)
  {
    return x * c;
  }

  template<typename Number>
  Vector<Number>
  operator-(const Vector<Number>& x)
  {
    Vector vec(x);
    vec *= Number(-1);
    return vec;
  }

  template<typename Number>
  Vector<Number>
  operator+(const Vector<Number>& x, const Vector<Number>& y)
  {
    Vector vec(x);
    vec += y;
    return vec;
  }

  template<typename Number>
  Vector<Number>
  operator-(const Vector<Number>& x, const Vector<Number>& y)
  {
    Vector vec(x);
    vec -= y;
    return vec;
  }

  template<typename Number>
  Number
  min(const Vector<Number>& x)
  {
    return x.min();
  }

  template<typename Number>
  Number
  max(const Vector<Number>& x)
  {
    return x.max();
  }

  template<typename Number>
  Number
  sum(const Vector<Number>& x)
  {
    return x.sum();
  }

  template<typename Number>
  types::real
  mean(const Vector<Number>& x)
  {
    return x.mean();
  }

  template<typename Number>
  Number
  linfty_norm(const Vector<Number>& x)
  {
    return x.linfty_norm();
  }

  template<typename Number>
  Number
  l1_norm(const Vector<Number>& x)
  {
    return x.l1_norm();
  }

  template<typename Number>
  types::real
  l2_norm(const Vector<Number>& x)
  {
    return x.l2_norm();
  }

  template<typename Number>
  types::real
  lp_norm(const Vector<Number>& x, const types::real p)
  {
    return x.lp_norm(p);
  }

  template<typename Number>
  types::real
  norm_sqr(const Vector<Number>& x)
  {
    return x.norm_sqr();
  }

  template<typename Number>
  types::real
  dot(const Vector<Number>& x, const Vector<Number>& y)
  {
    return x.dot(y);
  }

  template<typename Number>
  Vector<Number>
  operator*(const Vector<Number>& x, Number c)
  {
    Vector vec(x);
    vec *= c;
    return vec;
  }

  template<typename Number>
  std::ostream&
  operator<<(std::ostream& os, const Vector<Number>& vec)
  {
    return os << vec.to_string(3, false, false);
  }
}
