#pragma once
#include "framework/types.h"
#include "framework/math/vector.h"

namespace pdes
{
  template<typename Number = types::real>
  class Matrix : public NDArray<2, Number>
  {
  public:
    /// Constructs a square @p m by @p m matrix set to @p value.
    explicit Matrix(const size_t m, const Number value = 0)
      : Matrix(m, m, value) {}

    /// Constructs an @p m by @p n matrix set to @p value.
    explicit Matrix(const size_t m, const size_t n, const Number value = 0)
      : NDArray<2, Number>({m, n}, value) {}

    /// Constructs an @p m by @p m matrix and set the entries with raw data.
    explicit Matrix(const size_t m, const Number* ptr)
      : Matrix(m, m, ptr) {}

    /// Constructs an @p m by @p n matrix and set the entries with raw data.
    explicit Matrix(const size_t m, const size_t n, const Number* ptr)
      : NDArray<2, Number>({m, n}, ptr) {}

    /// Returns the number of rows.
    size_t m() const noexcept { return this->shape_[0]; }
    /// Returns the number of columns.
    size_t n() const noexcept { return this->shape_[1]; }

    /// Returns a reference to the entry at row @p i, column @p j.
    Number& operator()(size_t i, size_t j);
    /// Returns the value of the entry at row @p i, column @p j.
    Number operator()(size_t i, size_t j) const;

    /// Sets the entry at row @p i, column @p j to @p value.
    void set(size_t i, size_t j, Number value);
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

    /// Adds the given @p value to the row @p i and column @p j.
    void add(size_t i, size_t j, Number value);
    /**
     * Collectively add to several matrix elements.
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

    /**
     * Multiply the Matrix by a Vector, optionally adding the result into
     * the given destination vector.
    */
    void vmult(const Vector<Number>& x,
               Vector<Number>& b,
               bool add = false) const;
    /// Add a matrix-vector product to a destination Vector.
    void vmult_add(const Vector<Number>& x,
                   Vector<Number>& b) const;
    /// Returns a matrix-vector product.
    Vector<Number> vmult(const Vector<Number>& x) const;

    /// Multiply the Matrix by another Matrix.
    Matrix mmult(const Matrix& B) const;

    /// Prints the Matrix to an output stream
    void print(std::ostream& os = std::cout,
               unsigned int precision = 3,
               bool scientific = true) const;
  };

  /* -------------------- inline functions --------------------*/

  template<typename Number>
  Number&
  Matrix<Number>::operator()(size_t i, size_t j)
  {
    return this->at(i, j);
  }

  template<typename Number>
  Number
  Matrix<Number>::operator()(size_t i, size_t j) const
  {
    return this->at(i, j);
  }

  template<typename Number>
  void
  Matrix<Number>::set(const size_t i, const size_t j, const Number value)
  {
    this->at(i, j) = value;
  }

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

    this->set(rows.size(), rows.data(), cols.data(), values.data());
  }

  template<typename Number>
  void
  Matrix<Number>::set(const size_t n,
                      const size_t* rows,
                      const size_t* cols,
                      const Number* values)
  {
    for (size_t i = 0; i < n; ++i)
      this->set(rows[i], cols[i], values[i]);
  }

  template<typename Number>
  void
  Matrix<Number>::add(const size_t i, const size_t j, const Number value)
  {
    this->at(i, j) += value;
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

    this->add(rows.size(), rows.data(), cols.data(), values.data());
  }

  template<typename Number>
  void
  Matrix<Number>::add(const size_t n,
                      const size_t* rows,
                      const size_t* cols,
                      const Number* values)
  {
    for (size_t i = 0; i < n; ++i)
      this->add(rows[i], cols[i], values[i]);
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

    for (size_t i = 0; i < m(); ++i)
    {
      Number val = add ? b(i) : Number(0);
      for (size_t j = 0; j < n(); ++j)
        val += this->at(i, j) * Number(x(j));
      b(i) = val;
    }
  }

  template<typename Number>
  void
  Matrix<Number>::vmult_add(const Vector<Number>& x,
                            Vector<Number>& b) const
  {
    vmult(x, b, true);
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
  Matrix<Number>
  Matrix<Number>::mmult(const Matrix& B) const
  {
    if (n() != B.m())
      throw std::invalid_argument(
        "Dimension mismatch error for matrix-matrix multiplication.");

    Matrix C(m(), B.n(), Number(0));
    for (size_t i = 0; i < m(); ++i)
      for (size_t j = 0; j < B.n(); ++j)
        for (size_t k = 0; k < n(); ++k)
          C(i, j) += this->at(i, k) * B(k, j);
    return C;
  }

  template<typename Number>
  void
  Matrix<Number>::print(std::ostream& os,
                        const unsigned int precision,
                        const bool scientific) const
  {
    const auto old_flags = os.flags();
    const auto old_precision = os.precision(precision);

    os.precision(precision);
    if (scientific)
      os.setf(std::ios::scientific, std::ios::floatfield);
    else
      os.setf(std::ios::fixed, std::ios::floatfield);

    for (size_t i = 0; i < m(); ++i)
    {
      os << (i == 0 ? "[[" : " [");
      for (size_t j = 0; j < n(); ++j)
        os << this->at(i, j)
            << (j == n() - 1 ? "]" : " ");
      os << (i == m() - 1 ? "]" : "");
      os << std::endl;
    }

    os.flags(old_flags);
    os.precision(old_precision);
  }

  /* -------------------- free functions --------------------*/

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
  operator*(const Matrix<Number>& A, Number c)
  {
    Matrix mat(A);
    mat.scale(c);
    return mat;
  }

  template<typename Number>
  Matrix<Number>
  operator*(Number c, const Matrix<Number>& A)
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
}
