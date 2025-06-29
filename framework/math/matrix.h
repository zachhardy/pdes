#pragma once
#include "framework/types.h"
#include "framework/math/vector.h"


namespace pdes
{
  template<typename Number = types::real>
  class Matrix
  {
  public:
    using value_type = typename NDArray<2, Number>::value_type;

    Matrix() = default;

    /// Constructs a square @p m by @p m matrix set to zero.
    explicit Matrix(const size_t m)
      : Matrix(m, m, Number(0)) {}

    /// Constructs an @p m by @p n matrix set to zero.
    explicit Matrix(const size_t m, const size_t n)
      : Matrix(m, n, Number(0)) {}

    /// Constructs an @p m by @p n matrix set to @p value.
    explicit Matrix(const size_t m, const size_t n, const Number value)
      : entries_({m, n}, value) {}

    /// Constructs an @p m by @p m matrix and set the entries with raw data.
    explicit Matrix(const size_t m, const Number* ptr)
      : Matrix(m, m, ptr) {}

    /// Constructs an @p m by @p n matrix and set the entries with raw data.
    explicit Matrix(const size_t m, const size_t n, const Number* ptr)
      : entries_({m, n}, ptr) {}

    /// Returns the number of rows.
    size_t m() const noexcept { return entries_.shape()[0]; }
    /// Returns the number of columns.
    size_t n() const noexcept { return entries_.shape()[1]; }
    /// Returns the size of the Matrix
    size_t size() const { return entries_.size(); }

    /// Returns whether the Matrix is all zero.
    bool is_zero() const { return entries_.is_zero(); }
    /// Returns whether the Matrix is all non-negative.
    bool is_nonnegative() const { return entries_.is_nonnegative(); }
    /// Returns whether the Matrix is square.
    bool is_square() const { return n() == m(); }
    /// Returns whether the Matrix is empty.
    bool empty() const noexcept { return size() == 0; }

    /// Returns a reference to the entry at row @p i, column @p j.
    Number& operator()(const size_t i, const size_t j) { return entries_.at(i, j); }
    /// Returns the value of the entry at row @p i, column @p j.
    Number operator()(const size_t i, const size_t j) const { return entries_.at(i, j); }

    /// Returns an iterator to the start of row @p i.
    Number* begin(const size_t i) { return entries_.data() + i * n(); }
    /// Returns a constant iterator to the start of row @p i.
    const Number* begin(const size_t i) const { return entries_.data() + i * n(); }

    /// Returns an iterator to the end of row @p i.
    Number* end(const size_t i) { return entries_.data() + (i + 1) * n(); }
    /// Returns a constant iterator to the end of row @p i.
    const Number* end(const size_t i) const { return entries_.data() + (i + 1) * n(); }

    /// Returns a pointer to the raw matrix data.
    Number* data() { return entries_.data(); }
    /// Returns a constant pointer to the raw matrix data.
    const Number* data() const { return entries_.data(); }

    /// Sets the entry at row @p i, column @p j to @p value.
    void set(size_t i, size_t j, Number value) { entries_.at(i, j) = value; }
    /**
     * Collectively sets several matrix entries.
     *
     * @param rows The row indices of each set operation
     * @param cols The column indices of each set operation
     * @param values The values for each set operation.
     */
    void set(const std::vector<size_t>& rows,
             const std::vector<size_t>& cols,
             const std::vector<Number>& values);
    /**
     * Collectively sets several matrix entries.
     *
     * @param n The number of set operations to perform
     * @param rows A pointer to the first row index of the set operations
     * @param cols A pointer to the first column index of the set operations
     * @param values A pointer to the first value of the set operations
     */
    void set(size_t n,
             const size_t* rows,
             const size_t* cols,
             const Number* values);

    /// Multiplies by a scalar value.
    void scale(Number a);

    /// Multiplies by a scalar value.
    Matrix& operator *=(Number a);
    /// Divides by a non-zero scalar value.
    Matrix& operator/=(Number a);

    /// Adds the given @p value to the row @p i and column @p j.
    void add(size_t i, size_t j, Number value) { entries_.at(i, j) += value; }
    /**
     * Collectively adds to several matrix elements.
     *
     * @param rows The row indices of each add operation
     * @param cols The column indices of each add operation
     * @param values The values for each add operation.
     */
    void add(const std::vector<size_t>& rows,
             const std::vector<size_t>& cols,
             const std::vector<Number>& values);
    /**
     * Collectively adds to several matrix entries.
     *
     * @param n The number of add operations to perform
     * @param rows A pointer to the first row index of the add operations
     * @param cols A pointer to the first column index of the add operations
     * @param values A pointer to the first value of the add operations
     */
    void add(size_t n,
             const size_t* rows,
             const size_t* cols,
             const Number* values);

    /// Adds another Matrix.
    void add(const Matrix& other) { add(Number(1), other); }
    /// Adds a scaled Matrix.
    void add(const Number b, const Matrix& other) { sadd(Number(1), b, other); }
    /// Scales and adds another Matrix.
    void sadd(const Number a, const Matrix& other) { sadd(a, Number(1), other); }
    /// Scales and adds a scaled Matrix.
    void sadd(Number a, Number b, const Matrix& other);

    /// Adds of another Matrix.
    Matrix& operator+=(const Matrix& other);
    /// Subtracts of another Matrix.
    Matrix& operator-=(const Matrix& other);

    /**
     * Multiplies the Matrix by a Vector, optionally adding the result into
     * the given destination vector.
    */
    void vmult(const Vector<Number>& x,
               Vector<Number>& b,
               bool add = false) const;
    /// Adds a Matrix-Vector product to a destination Vector.
    void vmult_add(const Vector<Number>& x, Vector<Number>& b) const { vmult(x, b, true); }
    /// Returns a Matrix-Vector product.
    Vector<Number> vmult(const Vector<Number>& x) const;
    /// Returns a Matrix-Vector product.
    Vector<Number> operator*(const Vector<Number>& x) const;

    /// Returns a Matrix-Matrix product.
    Matrix mmult(const Matrix& B) const;
    /// Returns a Matrix-Matrix product.
    Matrix operator*(const Matrix& B) const;

    /// Compute the residual r = Ax - b.
    void residual(const Vector<Number>& x,
                  const Vector<Number>& b,
                  Vector<Number>& r) const;

    /// Returns the residual r = Ax - b.
    Vector<Number> residual(const Vector<Number>& x,
                            const Vector<Number>& b) const;

    /// Compute the residual norm.
    types::real residual_norm(const Vector<Number>& x,
                              const Vector<Number>& b) const;

    /// Write the matrix as a string.
    std::string to_string(unsigned int precision = 3,
                          bool scientific = false,
                          bool newline = true) const;

  private:
    NDArray<2, Number> entries_;
  };

  /* -------------------- member functions --------------------*/

  template<typename Number>
  void
  Matrix<Number>::set(const std::vector<size_t>& rows,
                      const std::vector<size_t>& cols,
                      const std::vector<Number>& values)
  {
    if (rows.size() != cols.size())
      throw std::invalid_argument("Vector size mismatch (rows, cols).");
    if (values.size() != rows.size())
      throw std::invalid_argument("Vector size mismatch (rows, values).");

    set(rows.size(), rows.data(), cols.data(), values.data());
  }

  template<typename Number>
  void
  Matrix<Number>::set(const size_t n,
                      const size_t* rows,
                      const size_t* cols,
                      const Number* values)
  {
    for (size_t i = 0; i < n; ++i)
      set(rows[i], cols[i], values[i]);
  }

  template<typename Number>
  void
  Matrix<Number>::scale(const Number a)
  {
    auto func = [a](Number x) { return a * x; };
    std::transform(entries_.begin(), entries_.end(), entries_.begin(), func);
  }

  template<typename Number>
  Matrix<Number>&
  Matrix<Number>::operator*=(const Number a)
  {
    scale(a);
    return *this;
  }

  template<typename Number>
  Matrix<Number>&
  Matrix<Number>::operator/=(Number a)
  {
    if (a == Number(0))
      throw std::invalid_argument("Division by zero error.");

    scale(Number(1) / a);
    return *this;
  }

  template<typename Number>
  void
  Matrix<Number>::add(const std::vector<size_t>& rows,
                      const std::vector<size_t>& cols,
                      const std::vector<Number>& values)
  {
    if (rows.size() != cols.size())
      throw std::invalid_argument("Vector size mismatch (rows, cols).");
    if (values.size() != rows.size())
      throw std::invalid_argument("Vector size mismatch (rows, values).");

    add(rows.size(), rows.data(), cols.data(), values.data());
  }

  template<typename Number>
  void
  Matrix<Number>::add(const size_t n,
                      const size_t* rows,
                      const size_t* cols,
                      const Number* values)
  {
    for (size_t i = 0; i < n; ++i)
      add(rows[i], cols[i], values[i]);
  }

  template<typename Number>
  void
  Matrix<Number>::sadd(Number a, Number b, const Matrix& other)
  {
    if (m() != other.m() or n() != other.n())
      throw std::invalid_argument("Matrix dimension mismatch.");

    auto func = [a, b](Number x, Number y) { return a * x + b * y; };
    std::transform(entries_.begin(),
                   entries_.end(),
                   other.entries_.begin(),
                   entries_.begin(),
                   func);
  }

  template<typename Number>
  Matrix<Number>&
  Matrix<Number>::operator+=(const Matrix& other)
  {
    add(other);
    return *this;
  }

  template<typename Number>
  Matrix<Number>&
  Matrix<Number>::operator-=(const Matrix& other)
  {
    add(Number(-1), other);
    return *this;
  }

  template<typename Number>
  void
  Matrix<Number>::vmult(const Vector<Number>& x,
                        Vector<Number>& b,
                        const bool add) const
  {
    if (x.size() != n())
      throw std::invalid_argument("Matrix-vector(vec) dimension mismatch.");
    if (b.size() != m())
      throw std::invalid_argument("Matrix-vector(dst) dimension mismatch.");

#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (size_t i = 0; i < m(); ++i)
    {
      Number val = add ? b(i) : Number(0);
      const Number* row = begin(i);
      for (size_t j = 0; j < n(); ++j)
        val += row[j] * x(j);
      b(i) = val;
    }
  }

  template<typename Number>
  Vector<Number>
  Matrix<Number>::vmult(const Vector<Number>& x) const
  {
    Vector dst(m(), Number(0));
    vmult(x, dst);
    return dst;
  }

  template<typename Number>
  Vector<Number>
  Matrix<Number>::operator*(const Vector<Number>& x) const
  {
    return vmult(x);
  }

  template<typename Number>
  Matrix<Number>
  Matrix<Number>::mmult(const Matrix& B) const
  {
    if (n() != B.m())
      throw std::invalid_argument(
        "Dimension mismatch error for matrix-matrix multiplication.");

#ifdef _OPENMP
#pragma omp parallel for
#endif
    Matrix C(m(), B.n(), Number(0));
    for (size_t i = 0; i < m(); ++i)
    {
      for (size_t j = 0; j < B.n(); ++j)
        for (size_t k = 0; k < n(); ++k)
          C(i, j) += (*this)(i, k) * B(k, j);
    }
    return C;
  }

  template<typename Number>
  Matrix<Number>
  Matrix<Number>::operator*(const Matrix& B) const
  {
    return mmult(B);
  }


  template<typename Number>
  void
  Matrix<Number>::residual(const Vector<Number>& x,
                           const Vector<Number>& b,
                           Vector<Number>& r) const
  {
    r = Number(0);

#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (size_t i = 0; i < m(); ++i)
    {
      Number sum = 0;
      const Number* row = begin(i);
      for (size_t j = 0; j < n(); ++j)
        sum += row[j] * x(j);
      r(i) = sum - b(i);
    }
  }

  template<typename Number>
  Vector<Number>
  Matrix<Number>::residual(const Vector<Number>& x, const Vector<Number>& b) const
  {
    Vector<Number> r(b.size(), Number(0));
    residual(x, b, r);
    return r;
  }


  template<typename Number>
  types::real
  Matrix<Number>::residual_norm(const Vector<Number>& x,
                                const Vector<Number>& b) const
  {
    Number norm_sqr = 0;

#ifdef _OPENMP
#pragma omp parallel for reduction(+:norm_sqr)
#endif
    for (size_t i = 0; i < m(); ++i)
    {
      Number sum = 0.0;
      const Number* row = begin(i);
      for (size_t j = 0; j < n(); ++j)
        sum += row[j] * x(j);

      const auto ri = sum - b(i);
      norm_sqr += ri * ri;
    }
    return std::sqrt(norm_sqr);
  }

  template<typename Number>
  std::string
  Matrix<Number>::to_string(const unsigned int precision,
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
    for (size_t i = 0; i < m(); ++i)
    {
      oss << (i == 0 ? "[" : " [");
      for (size_t j = 0; j < n(); ++j)
      {
        oss << (*this)(i, j);
        if (j < n() - 1)
          oss << " ";
      }
      oss << (i < m() - 1 ? "]\n" : "]]");
    }

    if (newline)
      oss << '\n';

    return oss.str();
  }

  /* -------------------- free functions --------------------*/

  template<typename Number>
  Matrix<Number>
  operator*(const Matrix<Number>& A, const Number c)
  {
    Matrix mat(A);
    mat.scale(c);
    return mat;
  }

  template<typename Number>
  Matrix<Number>
  operator*(const Number c, const Matrix<Number>& A)
  {
    return A * c;
  }

  template<typename Number>
  Matrix<Number>
  operator-(const Matrix<Number>& A)
  {
    Matrix mat(A);
    mat *= Number(-1);
    return mat;
  }

  template<typename Number>
  Matrix<Number>
  operator+(const Matrix<Number>& A, const Matrix<Number>& B)
  {
    Matrix mat(A);
    mat += B;
    return mat;
  }

  template<typename Number>
  Matrix<Number>
  operator-(const Matrix<Number>& A, const Matrix<Number>& B)
  {
    Matrix mat(A);
    mat -= B;
    return mat;
  }

  template<typename Number>
  Vector<Number>
  vmult(const Matrix<Number>& A, const Vector<Number>& x)
  {
    return A.vmult(x);
  }

  template<typename Number>
  Matrix<Number>
  mmult(const Matrix<Number>& A, const Matrix<Number>& B)
  {
    return A.mmult(B);
  }

  template<typename Number>
  Vector<Number>
  operator*(const Matrix<Number>& A, const Vector<Number>& x)
  {
    return A.vmult(x);
  }

  template<typename Number>
  Matrix<Number>
  operator*(const Matrix<Number>& A, const Matrix<Number>& B)
  {
    return A.mmult(B);
  }

  template<typename Number>
  std::ostream&
  operator<<(std::ostream& os, const Matrix<Number>& mat)
  {
    return os << mat.to_string(3, false, false);
  }
}
