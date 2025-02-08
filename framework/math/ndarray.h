#pragma once

#include <array>
#include <iostream>
#include <numeric>
#include <vector>

namespace pdes
{
  /**
   * Implementation of an arbitrary rank array.
   *
   * @tparam rank The number of rankensions in the NDArray.
   * @tparam Number The underlying datatype of the NDArray data.
   */
  template<int rank, typename Number = double>
  class NDArray
  {
  public:
    /// Constructs an empty NDArray.
    NDArray() noexcept;

    /// Constructs an NDArray from a shape.
    template<typename... Index>
    explicit NDArray(Index... shape) : NDArray({shape...}) {}

    /// Constructs an NDArray from a shape.
    NDArray(std::initializer_list<size_t> shape);

    /// Constructs an NDArray from a shape ad set all entries to @p value.
    explicit NDArray(std::initializer_list<size_t> shape, Number value);

    /// Constructs an NDArray from a shape and set the entries with raw data.
    explicit NDArray(std::initializer_list<size_t> shape, const Number* ptr);

    /// Constructs an NDArray by copying another.
    template<typename OtherNumber>
    explicit NDArray(const NDArray<rank, OtherNumber>& other);

    /// Constructs an NDArray by moving another.
    template<typename OtherNumber>
    explicit NDArray(NDArray<rank, OtherNumber>&& other) noexcept;

    /// Copies another NDArray into this one.
    template<typename OtherNumber>
    NDArray& operator=(const NDArray<rank, OtherNumber>& other);

    /// Moves another NDArray into this one.
    template<typename OtherNumber>
    NDArray& operator=(NDArray<rank, OtherNumber>&& other) noexcept;

    /// Reshapes an NDArray clearing existing data.
    template<typename... Index>
    void reshape(Index... shape) { reshape({shape...}); }

    /// Reshapes an NDArray clearing existing data.
    void reshape(std::initializer_list<size_t> shape);

    /// Sets all entries to the given value.
    void set(Number value);

    /// Clear the contents of the NDArray.
    void clear();

    /// Returns the number of entries in the NDArray.
    size_t size() const noexcept { return size_; }

    /// Returns whether the NDArray is all zero.
    bool is_zero() const;
    /// Returns whether the NDArray is all non-negative.
    bool is_nonnegative() const;
    /// Returns whether the NDArray is empty.
    bool empty() const noexcept { return size_ == 0; }

    /// Returns the shape of the NDArray.
    const std::array<size_t, rank>& shape() const noexcept { return shape_; }

    /// Returns a reference to the specified entry.
    template<typename... Index>
    Number& operator()(Index... indices) noexcept;
    /// Returns the value of the specified entry.
    template<typename... Index>
    Number operator()(Index... indices) const noexcept;

    /// Returns a reference to the specified entry with bounds checking.
    template<typename... Index>
    Number& at(Index... indices);
    /// Returns the value of the specified entry with bounds checking.
    template<typename... Index>
    Number at(Index... indices) const;

    /// Returns an iterator to the first entry of the NDArray.
    Number* begin() noexcept { return entries_.get(); }
    /// Returns a constant iterator to the first entry of the NDArray.
    const Number* cbegin() const noexcept { return entries_.get(); }

    /// Returns an iterator to the end of the NDArray.
    Number* end() noexcept { return entries_.get() + size_; }
    /// Returns a constant iterator to the end of the NDArray.
    const Number* cend() const noexcept { return entries_.get() + size_; }

    /// Returns a pointer to the raw underlying data.
    Number* data() noexcept { return entries_.get(); }
    /// Returns a constant pointer to the raw underlying data.
    const Number* data() const noexcept { return entries_.get(); }

    /// Scales all entries by a scalar.
    void scale(Number value);
    /// Multiplies all entries by a scalar.
    NDArray& operator*=(Number value);
    /// Divides all entries by a non-zero scalar.
    NDArray& operator/=(Number value);

    /// Adds another NDArray.
    void add(const NDArray<rank, Number>& other);
    /// Adds another NDArray scaled by @p b.
    void add(Number b, const NDArray<rank, Number>& other);
    /// Scales this NDArray by @p a and adds another.
    void sadd(Number a, const NDArray<rank, Number>& other);
    /// Scales this NDArray by @p a and adds another scaled by @p b.
    void sadd(Number a,
              Number b,
              const NDArray<rank, Number>& other);

    /// Adds another NDArray.
    NDArray& operator+=(const NDArray<rank, Number>& other);
    /// Subtracts another NDArray.
    NDArray& operator-=(const NDArray<rank, Number>& other);

  private:
    /// Compute the linear index from a set of indices.
    template<typename... Index>
    size_t compute_index(Index... indices) const;

  protected:
    size_t size_;
    std::array<size_t, rank> shape_;
    std::array<size_t, rank> strides_;
    std::unique_ptr<Number[]> entries_;
  };

  /* -------------------- inline functions --------------------*/

  template<int rank, typename Number>
  NDArray<rank, Number>::NDArray() noexcept
    : size_(0),
      shape_{},
      strides_{},
      entries_(nullptr)
  {
    static_assert(std::is_floating_point_v<Number>,
                  "NDArray must have a floating point type.");
  }

  template<int rank, typename Number>
  NDArray<rank, Number>::NDArray(const std::initializer_list<size_t> shape)
    : NDArray()
  {
    reshape(shape);
  }

  template<int rank, typename Number>
  NDArray<rank, Number>::NDArray(const std::initializer_list<size_t> shape,
                                 const Number value)
    : NDArray()
  {
    reshape(shape);
    set(value);
  }

  template<int rank, typename Number>
  NDArray<rank, Number>::NDArray(const std::initializer_list<size_t> shape,
                                 const Number* ptr)
    : NDArray()
  {
    reshape(shape);
    std::copy(ptr, ptr + size_, entries_.get());
  }


  template<int rank, typename Number>
  template<typename OtherNumber>
  NDArray<rank, Number>::NDArray(const NDArray<rank, OtherNumber>& other)
    : size_(other.size_),
      shape_(other.shape_),
      strides_(other.strides_),
      entries_(std::make_unique<Number[]>(size_))
  {
    std::copy(other.entries_.get(),
              other.entries_.get() + size_,
              entries_.get());
  }

  template<int rank, typename Number>
  template<typename OtherNumber>
  NDArray<rank, Number>::NDArray(NDArray<rank, OtherNumber>&& other) noexcept
    : size_(std::move(other.size_)),
      shape_(std::move(other.shape_)),
      strides_(std::move(other.strides_)),
      entries_(std::move(other.entries_))
  {
  }

  template<int rank, typename Number>
  template<typename OtherNumber>
  NDArray<rank, Number>&
  NDArray<rank, Number>::operator=(const NDArray<rank, OtherNumber>& other)
  {
    if (this != &other)
    {
      size_ = other.size_;
      shape_ = other.shape_;
      strides_ = other.strides_;
      entries_ = std::make_unique<Number[]>(size_);
      std::copy(other.entries_.get(),
                other.entries_.get() + size_,
                entries_.get());
    }
    return *this;
  }

  template<int rank, typename Number>
  template<typename OtherNumber>
  NDArray<rank, Number>&
  NDArray<rank, Number>::operator=(NDArray<rank, OtherNumber>&& other) noexcept
  {
    if (this != &other)
    {
      size_ = std::move(other.size_);
      shape_ = std::move(other.shape_);
      strides_ = std::move(other.strides_);
      entries_ = std::move(other.entries_);
    }
    return *this;
  }

  template<int rank, typename Number>
  void
  NDArray<rank, Number>::reshape(const std::initializer_list<size_t> shape)
  {
    if (shape.size() != rank)
      throw std::invalid_argument("Incompatible shape for given rank.");

    // Copy the rankension sizes into the shape array
    std::copy(shape.begin(), shape.end(), shape_.begin());

    // The size of the NDArray is the product of the size of each rankension
    constexpr auto func = std::multiplies<size_t>();
    size_ = std::accumulate(shape_.begin(), shape_.end(), 1, func);

    // The stride is the number of entries to get to the next entry
    // of the same rankension. The NDArray is ordered from the last
    // rankension to the first, so the stride of a rankension is the
    // product of all higher rankensions.
    strides_[rank - 1] = 1;
    for (int i = rank - 1; i > 0; --i)
      strides_[i - 1] = strides_[i] * shape_[i];

    // Resize the data array and set to zero
    entries_ = std::make_unique<Number[]>(size_);
  }

  template<int rank, typename Number>
  void
  NDArray<rank, Number>::set(Number value)
  {
    std::fill_n(entries_.get(), size_, static_cast<Number>(value));
  }

  template<int rank, typename Number>
  void
  NDArray<rank, Number>::clear()
  {
    size_ = 0;
    shape_ = {};
    strides_ = {};
    entries_ = nullptr;
  }

  template<int rank, typename Number>
  bool
  NDArray<rank, Number>::is_zero() const
  {
    auto func = [](Number x) { return x == Number(0); };
    return std::all_of(entries_.get(), entries_.get() + size_, func);
  }

  template<int rank, typename Number>
  bool
  NDArray<rank, Number>::is_nonnegative() const
  {
    auto func = [](Number x) { return x >= Number(0); };
    return std::all_of(entries_.get(), entries_.get() + size_, func);
  }

  template<int rank, typename Number>
  template<typename... Index>
  Number&
  NDArray<rank, Number>::operator()(Index... indices) noexcept
  {
    return entries_[compute_index(indices...)];
  }

  template<int rank, typename Number>
  template<typename... Index>
  Number
  NDArray<rank, Number>::operator()(Index... indices) const noexcept
  {
    return entries_[compute_index(indices...)];
  }

  template<int rank, typename Number>
  template<typename... Index>
  Number&
  NDArray<rank, Number>::at(Index... indices)
  {
    size_t idx[]{static_cast<size_t>(indices)...};
    for (int i = 0; i < rank; ++i)
    {
      if (idx[i] >= shape_[i])
        throw std::out_of_range("Index " + std::to_string(idx[i])
                                + " is out of bounds for rankension "
                                + std::to_string(i) + " with size "
                                + std::to_string(shape_[i]) + ".");
    }
    return entries_[compute_index(indices...)];
  }

  template<int rank, typename Number>
  template<typename... Index>
  Number
  NDArray<rank, Number>::at(Index... indices) const
  {
    size_t idx[]{static_cast<size_t>(indices)...};
    for (int i = 0; i < rank; ++i)
    {
      if (idx[i] >= shape_[i])
        throw std::out_of_range("Index " + std::to_string(idx[i])
                                + " is out of bounds for rankension "
                                + std::to_string(i) + " with size "
                                + std::to_string(shape_[i]) + ".");
    }
    return entries_[compute_index(indices...)];
  }

  template<int rank, typename Number>
  void
  NDArray<rank, Number>::scale(const Number value)
  {
    auto func = [value](Number x) { return Number(value) * x; };
    std::transform(this->cbegin(), this->cend(), this->begin(), func);
  }

  template<int rank, typename Number>
  NDArray<rank, Number>&
  NDArray<rank, Number>::operator*=(const Number value)
  {
    this->scale(value);
    return *this;
  }

  template<int rank, typename Number>
  NDArray<rank, Number>
  operator*(const NDArray<rank, Number>& ndarray, const Number value)
  {
    NDArray result(ndarray);
    result.scale(value);
    return result;
  }

  template<int rank, typename Number>
  NDArray<rank, Number>
  operator*(const Number value, const NDArray<rank, Number>& ndarray)
  {
    return ndarray * value;
  }

  template<int rank, typename Number>
  NDArray<rank, Number>
  operator-(const NDArray<rank, Number>& ndarray)
  {
    NDArray<rank, Number> result(ndarray);
    result.scale(Number(-1));
    return result;
  }

  template<int rank, typename Number>
  NDArray<rank, Number>&
  NDArray<rank, Number>::operator/=(const Number value)
  {
    if (value == Number(0))
      throw std::invalid_argument("Divide by zero error.");

    this->scale(Number(1 / value));
    return *this;
  }

  template<int rank, typename Number>
  NDArray<rank, Number>
  operator/(const NDArray<rank, Number>& ndarray, const Number value)
  {
    NDArray result(ndarray);
    result.scale(Number(1 / value));
    return result;
  }

  template<int rank, typename Number>
  void
  NDArray<rank, Number>::add(const NDArray& other)
  {
    this->add(Number(1), other);
  }

  template<int rank, typename Number>
  void
  NDArray<rank, Number>::add(const Number b,
                             const NDArray& other)
  {
    this->sadd(Number(1), Number(b), other);
  }

  template<int rank, typename Number>
  void
  NDArray<rank, Number>::sadd(const Number a,
                              const NDArray& other)
  {
    this->sadd(Number(a), Number(1), other);
  }

  template<int rank, typename Number>
  void
  NDArray<rank, Number>::sadd(const Number a,
                              const Number b,
                              const NDArray& other)
  {
    if (other.shape_ != this->shape)
      throw std::invalid_argument(
        "Incompatible NDArray rank for add operations.");

    auto func = [a, b](Number x, Number y) { return a * x + b * y; };
    std::transform(this->cbegin(),
                   this->cend(),
                   other.cbegin(),
                   this->begin(),
                   func);
  }

  template<int rank, typename Number>
  NDArray<rank, Number>&
  NDArray<rank, Number>::operator+=(const NDArray& other)
  {
    this->add(other);
    return *this;
  }

  template<int rank, typename Number>
  NDArray<rank, Number>
  operator+(const NDArray<rank, Number>& ndarray1,
            const NDArray<rank, Number>& ndarray2)


  {
    NDArray result(ndarray1);
    result += ndarray2;
    return result;
  }

  template<int rank, typename Number>
  NDArray<rank, Number>&
  NDArray<rank, Number>::operator-=(const NDArray& other)
  {
    this->add(Number(-1), other);
    return *this;
  }

  template<int rank, typename Number>
  NDArray<rank, Number>
  operator-(const NDArray<rank, Number>& ndarray1,
            const NDArray<rank, Number>& ndarray2)
  {
    NDArray result(ndarray1);
    result -= ndarray2;
    return result;
  }

  template<int rank, typename Number>
  template<typename... Index>
  size_t
  NDArray<rank, Number>::compute_index(Index... indices) const
  {
    const auto n = sizeof...(indices);
    size_t idx[]{static_cast<size_t>(indices)...};

    if (n == 1)
      return idx[0] * strides_[0];
    if (n == 2)
      return idx[0] * strides_[0] + idx[1] + strides_[1];
    if (n == 3)
      return idx[0] * strides_[0] + idx[1] * strides_[1]
             + idx[2] * strides_[2];
    if (n == 4)
      return idx[0] * strides_[0] + idx[1] * strides_[1]
             + idx[2] * strides_[2] + idx[3] * strides_[3];

    size_t index = 0;
    for (int i = 0; i < rank; ++i)
      index += idx[i] * strides_[i];
    return index;
  }
}
