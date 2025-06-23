#pragma once

#include "framework/types.h"
#include <array>
#include <iostream>
#include <numeric>
#include <vector>

namespace pdes
{
  namespace internal
  {
    template<size_t rank>
    std::array<size_t, rank>
    to_array(const std::vector<size_t>& shape)
    {
      if (shape.size() != rank)
        throw std::invalid_argument("Shape size does not match NDArray rank.");
      std::array<size_t, rank> result;
      std::copy_n(shape.begin(), rank, result.begin());
      return result;
    }

    template<size_t rank>
    std::array<size_t, rank>
    to_array(std::initializer_list<size_t> shape)
    {
      if (shape.size() != rank)
        throw std::invalid_argument("Shape size does not match NDArray rank.");
      std::array<size_t, rank> result;
      std::copy(shape.begin(), shape.end(), result.begin());
      return result;
    }
  }

  /**
   * Implementation of an arbitrary rank array.
   *
   * @tparam rank The number of dimensions in the NDArray.
   * @tparam Number The underlying datatype of the NDArray data.
   */
  template<int rank, typename Number = types::real>
  class NDArray
  {
  public:
    /// Constructs an empty NDArray.
    NDArray() noexcept;

    /// Constructs an NDArray from a shape.
    explicit NDArray(const std::array<size_t, rank>& shape);
    explicit NDArray(const std::vector<size_t>& shape);
    NDArray(std::initializer_list<size_t> shape);

    /// Constructs an NDArray from a shape ad set all entries to @p value.
    explicit NDArray(const std::array<size_t, rank>& shape, Number value);
    explicit NDArray(const std::vector<size_t>& shape, Number value);
    explicit NDArray(std::initializer_list<size_t> shape, Number value);

    /// Constructs an NDArray from a shape and set the entries with raw data.
    explicit NDArray(const std::array<size_t, rank>& shape, const Number* ptr);
    explicit NDArray(const std::vector<size_t>& shape, const Number* ptr);
    explicit NDArray(std::initializer_list<size_t> shape, const Number* ptr);

    /// Constructs an NDArray by copying another.
    explicit NDArray(const NDArray& other);
    /// Constructs an NDArray by moving another.
    explicit NDArray(NDArray&& other) noexcept;

    /// Copies another NDArray into this one.
    NDArray& operator=(const NDArray& other);
    /// Moves another NDArray into this one.
    NDArray& operator=(NDArray&& other) noexcept;

    /// Sets the entire NDArray to a number.
    NDArray& operator=(Number value);

    /// Reshapes an NDArray clearing existing data.
    void reshape(const std::array<size_t, rank>& shape);
    void reshape(const std::vector<size_t>& shape);
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
    const Number* begin() const noexcept { return entries_.get(); }

    /// Returns an iterator to the end of the NDArray.
    Number* end() noexcept { return entries_.get() + size_; }
    /// Returns a constant iterator to the end of the NDArray.
    const Number* end() const noexcept { return entries_.get() + size_; }

    /// Returns a pointer to the raw underlying data.
    Number* data() noexcept { return entries_.get(); }
    /// Returns a constant pointer to the raw underlying data.
    const Number* data() const noexcept { return entries_.get(); }

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
  NDArray<rank, Number>::NDArray(const std::array<size_t, rank>& shape)
    : NDArray()
  {
    reshape(shape);
  }

  template<int rank, typename Number>
  NDArray<rank, Number>::NDArray(const std::vector<size_t>& shape)
    : NDArray(internal::to_array<rank>(shape)) {}

  template<int rank, typename Number>
  NDArray<rank, Number>::NDArray(const std::initializer_list<size_t> shape)
    : NDArray(internal::to_array<rank>(shape)) {}

  template<int rank, typename Number>
  NDArray<rank, Number>::NDArray(const std::array<size_t, rank>& shape,
                                 const Number value)
    : NDArray(shape)
  {
    set(value);
  }

  template<int rank, typename Number>
  NDArray<rank, Number>::NDArray(const std::vector<size_t>& shape,
                                 const Number value)
    : NDArray(internal::to_array<rank>(shape), value) {}

  template<int rank, typename Number>
  NDArray<rank, Number>::NDArray(const std::initializer_list<size_t> shape,
                                 const Number value)
    : NDArray(internal::to_array<rank>(shape), value) {}

  template<int rank, typename Number>
  NDArray<rank, Number>::NDArray(const std::array<size_t, rank>& shape,
                                 const Number* ptr)
    : NDArray(shape)
  {
    std::copy(ptr, ptr + size_, entries_.get());
  }

  template<int rank, typename Number>
  NDArray<rank, Number>::NDArray(const std::vector<size_t>& shape,
                                 const Number* ptr)
    : NDArray(internal::to_array<rank>(shape), ptr) {}

  template<int rank, typename Number>
  NDArray<rank, Number>::NDArray(const std::initializer_list<size_t> shape,
                                 const Number* ptr)
    : NDArray(internal::to_array<rank>(shape), ptr) {}

  template<int rank, typename Number>
  NDArray<rank, Number>::NDArray(const NDArray& other)
    : size_(other.size_),
      shape_(other.shape_),
      strides_(other.strides_),
      entries_(std::make_unique<Number[]>(other.size_))
  {
    std::copy(other.entries_.get(),
              other.entries_.get() + size_,
              entries_.get());
  }

  template<int rank, typename Number>
  NDArray<rank, Number>::NDArray(NDArray&& other) noexcept
    : size_(other.size_),
      shape_(std::move(other.shape_)),
      strides_(std::move(other.strides_)),
      entries_(std::move(other.entries_))
  {
    other.size_ = 0;
  }

  template<int rank, typename Number>
  NDArray<rank, Number>&
  NDArray<rank, Number>::operator=(const NDArray& other)
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
  NDArray<rank, Number>&
  NDArray<rank, Number>::operator=(NDArray&& other) noexcept
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
  NDArray<rank, Number>&
  NDArray<rank, Number>::operator=(Number value)
  {
    set(value);
    return *this;
  }

  template<int rank, typename Number>
  void
  NDArray<rank, Number>::reshape(const std::array<size_t, rank>& shape)
  {
    // Copy the dimension sizes into the shape array
    std::copy(shape.begin(), shape.end(), shape_.begin());

    // The size of the NDArray is the product of the size of each dimension
    constexpr auto func = std::multiplies<size_t>();
    size_ = std::accumulate(shape_.begin(), shape_.end(), 1, func);

    // The stride is the number of entries to get to the next entry of the
    // same dimension. The NDArray is ordered from the last dimension to the
    // first, so the stride of a dimension is the product of all higher
    // dimensions.
    strides_[rank - 1] = 1;
    for (int i = rank - 1; i > 0; --i)
      strides_[i - 1] = strides_[i] * shape_[i];

    // Resize the data array and set to zero
    entries_ = std::make_unique<Number[]>(size_);
  }

  template<int rank, typename Number>
  void
  NDArray<rank, Number>::reshape(const std::vector<size_t>& shape)
  {
    reshape(internal::to_array<rank>(shape));
  }

  template<int rank, typename Number>
  void
  NDArray<rank, Number>::reshape(const std::initializer_list<size_t> shape)
  {
    reshape(internal::to_array<rank>(shape));
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
                                + " is out of bounds for dimension "
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
                                + " is out of bounds for dimension "
                                + std::to_string(i) + " with size "
                                + std::to_string(shape_[i]) + ".");
    }
    return entries_[compute_index(indices...)];
  }

  template<int rank, typename Number>
  template<typename... Index>
  size_t
  NDArray<rank, Number>::compute_index(Index... indices) const
  {
    static_assert(sizeof...(indices) == rank,
                  "Incorrect number of indices for NDArray rank.");
    std::array<size_t, rank> idx{static_cast<size_t>(indices)...};

    size_t index = 0;
    for (int i = 0; i < rank; ++i)
      index += idx[i] * strides_[i];
    return index;
  }
}
