#pragma once
#include "framework/math/vector.h"
#include <vector>
#include <map>

namespace pdes
{
  /**
   * List-of-Lists (LIL) sparse matrix with compressible CSR backend.
   *
   * This matrix supports flexible construction via row-wise or triplet-style
   * insertion using an internal list-of-lists format. Once construction is
   * complete, the matrix can be compressed into Compressed Sparse Row (CSR)
   * format for fast access and multiplication.
   *
   * - Before compression, users may freely use `add()` and `set()` to modify values.
   * - After compression, the matrix becomes immutable and efficient for read-only access.
   *
   * Common usage:
   *   1. Construct the matrix and insert values using `add()` or `set()`.
   *   2. Call `compress()` (not shown in this snippet).
   *   3. Perform matrix-vector operations (e.g., `vmult()`).
   *
   * This class does not support structural changes (e.g., inserting new nonzeros)
   * after compression.
   *
   * @tparam Number The numeric type stored in the matrix (default: types::real).
   */
  template<typename Number = types::real>
  class SparseMatrix
  {
  public:
    struct RowEntry
    {
      size_t col;
      Number value;
    };

    using value_type = Number;
    using vector_type = Vector<Number>;

    /// Default constructor.
    SparseMatrix() = default;

    /// Constructs a square @p n x @p n matrix..
    explicit SparseMatrix(size_t n) : SparseMatrix(n, n) {}

    /// Constructs a matrix with given @p m rows and @p n columns.
    SparseMatrix(size_t m, size_t n);

    /**
     * Clears all matrix data.
     *
     * Resets both assembled and compressed data, returning the matrix to an
     * uninitialized state.
     */
    void clear() noexcept;

    /// Returns the number of rows in the matrix.
    size_t m() const { return m_; }

    /// Returns the number of columns in the matrix.
    size_t n() const { return n_; }

    /// Returns the size of the matrix.
    size_t size() const { return m_ * n_; }

    /// Returns true if the matrix has been compressed.
    bool is_compressed() const noexcept { return compressed_; }

    /// Returns true if the matrix is empty (has zero rows or columns).
    bool empty() const noexcept { return m_ == 0 or n_ == 0; }

    /// Reinitializes the sparse matrix with @p m rows and @n columns, clearing existing data.
    void reinit(size_t m, size_t n);

    /**
     * Compresses the LIL-format matrix into CSR format.
     *
     * Converts the list-of-lists representation into CSR by flattening each
     * row's map into a contiguous format with row pointers, column indices,
     * and non-zero values. After compression, the matrix is immutable.
     */
    void compress();

    /**
     * Returns a reference to the value at (@p i, @p j) or throws if not present.
     *
     * Unlike el(), this accessor throws an exception if the (@p i, @p j) entry does not exist.
     */
    Number& operator()(size_t i, size_t j);

    /**
     * Returns the value at (@p i, @p j) or throws if not present.
     *
     * Unlike el(), this accessor throws an exception if the (@p i, @p j) entry does not exist.
     */
    Number operator()(size_t i, size_t j) const;

    /**
     * Returns the value at (@p i, @p j), or zero if not present.
     *
     * Safe read-only accessor for numerical operations.
     */
    Number el(size_t i, size_t j) const;

    /**
     * Iterator over a single row of the matrix.
     *
     * Provides access to (column, value) pairs in that row.
     */
    std::vector<RowEntry> row_entries(size_t i) const;

    /// Sets the @p value at entry (@p i, @p j), overwriting any existing value.
    void set(size_t i, size_t j, Number value);

    /**
     * Sets a dense block of values using row and column index arrays.
     *
     * The size of `values` must match size of `rows` and `cols`.
     */
    void set(const std::vector<size_t>& rows,
             const std::vector<size_t>& cols,
             const std::vector<Number>& values);

    /// Sets values[k] to (rows[k], cols[k]) for k in [0, n).
    void set(size_t n,
             const size_t* rows,
             const size_t* cols,
             const Number* values);

    /// Adds the @p value at entry (@p i, @p j).
    void add(size_t i, size_t j, Number value);

    /**
     * Adds a dense block of values using row and column index arrays.
     *
     * The size of `values` must match the size of `rows` and  `cols`.
     */
    void add(const std::vector<size_t>& rows,
             const std::vector<size_t>& cols,
             const std::vector<std::vector<Number>>& values);

    //// Assigns values[k] to (rows[k], cols[k]) for k in [0, n).
    void add(size_t n,
             const size_t* rows,
             const size_t* cols,
             const Number* values);

    /// Scales all values in the matrix by a given @p factor.
    void scale(Number factor);

    /// Scales all entries by @p factor in-place.
    SparseMatrix& operator*=(Number factor);

    /// Divides all entries by non-zero @p factor in-place.
    SparseMatrix& operator/=(Number factor);

    /**
     * Computes matrix-vector product b = Ax.
     * If add = true, result is added into existing values in @p b.
     */
    void vmult(const Vector<Number>& x, Vector<Number>& b, bool add = false) const;

    /// Adds a matrix-vector product to the destination via b += Ax.
    void vmult_add(const Vector<Number>& x, Vector<Number>& b) const { vmult(x, b, true); }

    /// Returns the matrix-vector product b = Ax.
    Vector<Number> vmult(const Vector<Number>& x) const;

    /// Returns the matrix-vector product b = Ax.
    Vector<Number> operator*(const Vector<Number>& x) const;

    /// Computes the residual r = Ax - b.
    void residual(const Vector<Number>& x,
                  const Vector<Number>& b,
                  Vector<Number>& r) const;

    /// Returns the residual vector r = Ax - b.
    Vector<Number> residual(const Vector<Number>& x,
                            const Vector<Number>& b) const;

    /// Returns the Euclidean norm of the residual r = Ax - b.
    types::real residual_norm(const Vector<Number>& x,
                              const Vector<Number>& b) const;

    std::string to_string(unsigned int precision = 3,
                          bool scientific = false,
                          bool newline = true) const;

  private:
    size_t m_ = 0;
    size_t n_ = 0;

    /// The starting index for each row.
    std::vector<size_t> row_ptr_;
    /// The non-zero column indices on each row.
    std::vector<size_t> cols_;
    /// The values for each non-zero column on each row
    std::vector<Number> values_;

    /// A flag for the matrix being compressed. Once compressed, the
    /// matrix is immutable.
    bool compressed_ = false;

    /// An LIL-formatted cache before compression
    std::vector<std::map<size_t, Number>> cache_;
  };

  /*-------------------- member functions --------------------*/

  template<typename Number>
  SparseMatrix<Number>::SparseMatrix(const size_t m, const size_t n)
    : m_(m),
      n_(n),
      cache_(m)
  {}

  template<typename Number>
  void
  SparseMatrix<Number>::clear() noexcept
  {
    m_ = 0;
    n_ = 0;
    compressed_ = false;

    row_ptr_.clear();
    cols_.clear();
    values_.clear();
    cache_.clear();
  }

  template<typename Number>
  void
  SparseMatrix<Number>::reinit(const size_t m, const size_t n)
  {
    m_ = m;
    n_ = n;

    row_ptr_.clear();
    cols_.clear();
    values_.clear();

    cache_.resize(m);
  }


  template<typename Number>
  void
  SparseMatrix<Number>::compress()
  {
    if (compressed_)
      throw std::logic_error("SparseMatrix::compress(): already compressed");

    size_t count = 0;
    row_ptr_.resize(m_ + 1, 0);
    for (size_t i = 0; i < m_; ++i)
    {
      row_ptr_[i] = count;
      for (const auto& [j, val]: cache_[i])
      {
        cols_.push_back(j);
        values_.push_back(val);
        ++count;
      }
    }
    row_ptr_[m_] = count;
    compressed_ = true;
    cache_.clear();
  }

  template<typename Number>
  Number&
  SparseMatrix<Number>::operator()(const size_t i, const size_t j)
  {
    if (i >= m_ or j >= n_)
      throw std::out_of_range("SparseMatrix::operator(): index out of bounds");

    // Find value in CSR format, if compressed
    if (compressed_)
    {
      for (size_t k = row_ptr_[i]; k < row_ptr_[i + 1]; ++k)
      {
        if (cols_[k] == j)
          return values_[k];
      }
      throw std::runtime_error("SparseMatrix::operator(): entry not found.");
    }

    // Otherwise, find value in LIL format
    const auto& row_map = cache_[i];
    auto it = row_map.find(j);
    if (it == row_map.end())
      throw std::runtime_error("SparseMatrix::operator(): entry not found.");
    return it->second;
  }


  template<typename Number>
  Number
  SparseMatrix<Number>::operator()(const size_t i, const size_t j) const
  {
    if (i >= m_ or j >= n_)
      throw std::out_of_range("SparseMatrix::operator(): index out of bounds");

    // Find value in CSR format, if compressed
    if (compressed_)
    {
      for (size_t k = row_ptr_[i]; k < row_ptr_[i + 1]; ++k)
      {
        if (cols_[k] == j)
          return values_[k];
      }
      throw std::runtime_error("SparseMatrix::operator(): entry not found.");
    }

    // Otherwise, find value in LIL format
    const auto& row_map = cache_[i];
    auto it = row_map.find(j);
    if (it == row_map.end())
      throw std::runtime_error("SparseMatrix::operator(): entry not found.");
    return it->second;
  }

  template<typename Number>
  Number
  SparseMatrix<Number>::el(const size_t i, const size_t j) const
  {
    if (i >= m_ or j >= n_)
      throw std::out_of_range("SparseMatrix::operator(): index out of bounds");

    // Find value in CSR format, if compressed
    if (compressed_)
    {
      for (size_t k = row_ptr_[i]; k < row_ptr_[i + 1]; ++k)
      {
        if (cols_[k] == j)
          return values_[k];
      }
      return Number(0);
    }

    // Otherwise, find value in LIL format
    const auto& row_map = cache_[i];
    auto it = row_map.find(j);
    return it != row_map.end() ? it->second : Number(0);
  }

  template<typename Number>
  std::vector<typename SparseMatrix<Number>::RowEntry>
  SparseMatrix<Number>::row_entries(const size_t i) const
  {
    if (i >= m_)
      throw std::out_of_range("SparseMatrix::row_entries(): row index out of bounds");

    std::vector<RowEntry> pairs;
    if (compressed_)
    {
      for (size_t k = row_ptr_[i]; k < row_ptr_[i + 1]; ++k)
        pairs.emplace({cols_[k], values_[k]});
    }
    else
    {
      for (const auto& [col, val]: cache_[i])
        pairs.emplace({col, val});
    }
    return pairs;
  }

  template<typename Number>
  void
  SparseMatrix<Number>::set(const size_t i,
                            const size_t j,
                            const Number value)
  {
    if (compressed_)
      throw std::logic_error("Cannot add to compressed matrix.");
    if (i >= m_ or j >= n_)
      throw std::out_of_range("SparseMatrix::add() index out of bounds");
    cache_[i][j] = value;
  }

  template<typename Number>
  void
  SparseMatrix<Number>::set(const std::vector<size_t>& rows,
                            const std::vector<size_t>& cols,
                            const std::vector<Number>& values)
  {
    if (compressed_)
      throw std::logic_error("Cannot set block in compressed matrix.");
    if (rows.size() != cols.size() or cols.size() != values.size())
      throw std::invalid_argument("Mismatched sizes in SparseMatrix::set triplets");

    for (size_t i = 0; i < rows.size(); ++i)
      set(rows[i], cols[i], values[i]);
  }

  template<typename Number>
  void
  SparseMatrix<Number>::set(const size_t n,
                            const size_t* rows,
                            const size_t* cols,
                            const Number* values)
  {
    if (compressed_)
      throw std::logic_error("Cannot set block in compressed matrix.");
    for (size_t i = 0; i < n; ++i)
      set(rows[i], cols[i], values[i]);
  }

  template<typename Number>
  void
  SparseMatrix<Number>::add(const size_t i,
                            const size_t j,
                            const Number value)
  {
    if (compressed_)
      throw std::logic_error("Cannot add to compressed matrix.");
    if (i >= m_ or j >= n_)
      throw std::out_of_range("SparseMatrix::add() index out of bounds");
    cache_[i][j] += value;
  }

  template<typename Number>
  void
  SparseMatrix<Number>::add(const std::vector<size_t>& rows,
                            const std::vector<size_t>& cols,
                            const std::vector<std::vector<Number>>& values)
  {
    if (compressed_)
      throw std::logic_error("Cannot set block in compressed matrix.");
    if (rows.size() != cols.size() or cols.size() != values.size())
      throw std::invalid_argument("Mismatched sizes in SparseMatrix::set triplets");

    for (size_t i = 0; i < rows.size(); ++i)
      add(rows[i], cols[i], values[i]);
  }

  template<typename Number>
  void
  SparseMatrix<Number>::add(const size_t n,
                            const size_t* rows,
                            const size_t* cols,
                            const Number* values)
  {
    if (compressed_)
      throw std::logic_error("Cannot set block in compressed matrix.");
    for (size_t i = 0; i < n; ++i)
      add(rows[i], cols[i], values[i]);
  }

  template<typename Number>
  void
  SparseMatrix<Number>::scale(const Number factor)
  {
    if (not compressed_)
      throw std::runtime_error("SparseMatrix::scale(): matrix must be compressed");

    auto func = [factor](const Number x) { return x * factor; };
    std::transform(values_.begin(), values_.end(), values_.begin(), func);
  }

  template<typename Number>
  SparseMatrix<Number>&
  SparseMatrix<Number>::operator*=(const Number factor)
  {
    scale(factor);
    return *this;
  }

  template<typename Number>
  SparseMatrix<Number>&
  SparseMatrix<Number>::operator/=(Number factor)
  {
    if (factor == Number(0))
      throw std::invalid_argument("Division by zero error.");

    scale(Number(1) / factor);
    return *this;
  }

  template<typename Number>
  void
  SparseMatrix<Number>::vmult(const Vector<Number>& x,
                              Vector<Number>& b,
                              const bool add) const
  {
    if (not compressed_)
      throw std::logic_error("SparseMatrix::vmult(): matrix must be compressed");
    if (x.size() != n_ or b.size() != m_)
      throw std::invalid_argument("SparseMatrix::vmult(): input vector size mismatch");

#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (size_t i = 0; i < m_; ++i)
    {
      Number sum = add ? b(i) : Number(0);
      for (size_t k = row_ptr_[i]; k < row_ptr_[i + 1]; ++k)
        sum += values_[k] * x[cols_[k]];
      b(i) = sum;
    }
  }

  template<typename Number>
  Vector<Number>
  SparseMatrix<Number>::vmult(const Vector<Number>& x) const
  {
    Vector<Number> b(m_, Number(0));
    vmult(x, b);
    return b;
  }

  template<typename Number>
  Vector<Number>
  SparseMatrix<Number>::operator*(const Vector<Number>& x) const
  {
    return vmult(x);
  }

  template<typename Number>
  void
  SparseMatrix<Number>::residual(const Vector<Number>& x,
                                 const Vector<Number>& b,
                                 Vector<Number>& r) const
  {
    if (not compressed_)
      throw std::logic_error("SparseMatrix::vmult(): matrix must be compressed");
    if (x.size() != n_ or b.size() != m_)
      throw std::invalid_argument("SparseMatrix::vmult(): input vector size mismatch");

#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (size_t i = 0; i < m(); ++i)
    {
      Number sum = 0;
      for (size_t k = row_ptr_[i]; k < row_ptr_[i + 1]; ++k)
        sum += values_[k] * x(k);
      r(i) = sum - b(i);
    }
  }

  template<typename Number>
  Vector<Number>
  SparseMatrix<Number>::residual(const Vector<Number>& x,
                                 const Vector<Number>& b) const
  {
    Vector<Number> r(m_, Number(0));
    residual(x, b, r);
    return r;
  }

  template<typename Number>
  types::real
  SparseMatrix<Number>::residual_norm(const Vector<Number>& x,
                                      const Vector<Number>& b) const
  {
    Number norm_sqr = 0;

#ifdef _OPENMP
#pragma omp parallel for reduction(+:norm_sqr)
#endif
    for (size_t i = 0; i < m(); ++i)
    {
      Number sum = 0.0;
      for (size_t k = row_ptr_[i]; k < row_ptr_[i + 1]; ++k)
        sum += values_[k] * x(k);

      const auto ri = sum - b(i);
      norm_sqr += ri * ri;
    }
    return std::sqrt(norm_sqr);
  }

  template<typename Number>
  std::string
  SparseMatrix<Number>::to_string(const unsigned int precision,
                                  const bool scientific,
                                  const bool newline) const
  {
    if (!compressed_)
      throw std::logic_error("SparseMatrix::to_string(): matrix must be compressed");

    std::ostringstream oss;
    oss << std::setprecision(precision);
    if (scientific)
      oss << std::scientific;
    else
      oss << std::fixed;

    oss << "[";
    for (size_t i = 0; i < m_; ++i)
    {
      oss << (i == 0 ? "[" : " [");
      bool first = true;
      for (size_t k = row_ptr_[i]; k < row_ptr_[i + 1]; ++k)
      {
        if (!first)
          oss << " ";
        oss << "(" << cols_[k] << ", " << values_[k] << ")";
        first = false;
      }
      oss << (i < m_ - 1 ? "]\n" : "]]");
    }

    if (newline)
      oss << '\n';

    return oss.str();
  }

  /* -------------------- free functions --------------------*/

  template<typename Number>
  SparseMatrix<Number>
  operator*(const SparseMatrix<Number>& A, const Number c)
  {
    SparseMatrix<Number> mat(A);
    mat *= c;
    return mat;
  }

  template<typename Number>
  SparseMatrix<Number>
  operator*(const Number c, const SparseMatrix<Number>& A)
  {
    return A * c;
  }

  template<typename Number>
  SparseMatrix<Number>
  operator/(const SparseMatrix<Number>& A, const Number c)
  {
    SparseMatrix<Number> mat(A);
    mat /= c;
    return mat;
  }

  template<typename Number>
  Vector<Number>
  vmult(const SparseMatrix<Number>& A, const Vector<Number>& x)
  {
    return A.vmult(x);
  }

  template<typename Number>
  Vector<Number>
  operator*(const SparseMatrix<Number>& A, const Vector<Number>& x)
  {
    return A.vmult(x);
  }
}
