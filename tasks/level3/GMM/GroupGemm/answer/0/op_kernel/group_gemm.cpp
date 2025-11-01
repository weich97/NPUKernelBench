#include <cstdint>
#include <cstring>
#include <iostream>
#include <string>
#include <tuple>
#include <type_traits>
#include <vector>

#include "lib/matmul_intf.h"
#include <kernel_operator.h>

#define __TLA_REQUIRES(...) typename std::enable_if<(__VA_ARGS__)>::type * = nullptr

template <bool VALUE, class... Args>
constexpr bool DEPENDENT_BOOL_VALUE = VALUE;

template <class... Args>
constexpr bool DEPENDENT_FALSE = DEPENDENT_BOOL_VALUE<false, Args...>;

template <uint32_t ALIGN, typename T>
__forceinline__ [aicore]
constexpr T RoundUp(const T &val)
{
    static_assert(ALIGN != 0, "ALIGN must not be 0");
    return (val + ALIGN - 1) / ALIGN * ALIGN;
}

template <class T>
__forceinline__ [aicore]
constexpr T RoundUp(const T &val, const T align)
{
    return (val + align - 1) / align * align;
}

template <uint32_t ALIGN, typename T>
__forceinline__ [aicore]
constexpr T RoundDown(const T val)
{
    static_assert(ALIGN != 0, "ALIGN must not be 0");
    return val / ALIGN * ALIGN;
}

template <class T>
__forceinline__ [aicore]
constexpr T RoundDown(const T val, const T align)
{
    return val / align * align;
}

template <uint32_t DIVISOP, typename T>
__forceinline__ [aicore]
constexpr T CeilDiv(const T dividend)
{
    static_assert(DIVISOP != 0, "DIVISOP must not be 0");
    return (dividend + DIVISOP - 1) / DIVISOP;
}

template <class T>
__forceinline__ [aicore]
constexpr T CeilDiv(const T dividend, const T divisor)
{
    return (dividend + divisor - 1) / divisor;
}

namespace Catlass {

constexpr uint32_t BYTE_PER_C0 = 32;
constexpr uint32_t BYTE_PER_C2 = 64;
constexpr uint32_t C0_NUM_PER_FRACTAL = 16;
constexpr uint32_t BYTE_PER_FRACTAL = BYTE_PER_C0 * C0_NUM_PER_FRACTAL;

constexpr uint32_t BYTE_PER_BLK = 32;
constexpr uint32_t BLK_NUM_PER_VECTOR_FRACTAL = 8;
constexpr uint32_t BYTE_PER_VECTOR_FRACTAL = BYTE_PER_BLK * BLK_NUM_PER_VECTOR_FRACTAL;

constexpr uint64_t L2_OFFSET = 0;
constexpr uint32_t STRIDE_LIMIT = 65536;

template <int RANK_, class Index_ = uint32_t, class LongIndex_ = int64_t>
struct Coord {
public:
    // Number of elements in Coord
    static const int RANK = RANK_;

    // Index typen used to store elements
    using Index = Index_;

    // Type used to represent linear offsets
    using LongIndex = LongIndex_;

    // Default ctor initializes uniformly
    __forceinline__ [aicore] constexpr
    explicit Coord(Index value = Index(0))
    {
        for (int i = 0; i < RANK; ++i) {
            idx[i] = value;
        }
    }

    // Constructs from an array of integers
    __forceinline__ [aicore] constexpr
    Coord(Index const (&idx_)[RANK])
    {
        for (int i = 0; i < RANK; ++i) {
            idx[i] = idx_[i];
        }
    }

    // Constructs frrom an array of integers
    __forceinline__ [aicore]
    int Argmin() const
    {
        int i = 0;
        for (int j = 1; j < RANK; ++j) {
            if (idx[j] < idx[i]) {
                i = j;
            }
        }
        return i;
    }

    // Returns the index of the dimension with greatest value
    __forceinline__ [aicore]
    int Argmax() const
    {
        int i = 0;
        for (int j = 1; j < RANK; ++j) {
            if (idx[j] > idx[i]) {
                i = j;
            }
        }
        return i;
    }

    // Returns true if Coord is non-zero
    __forceinline__ [aicore]
    explicit operator bool() const
    {
        for (int i = 0; i < RANK; ++i) {
            if (idx[i]) {
                return true;
            }
        }
        return false;
    }

    // Return true if Coord is uniformly zero.
    __forceinline__ [aicore]
    bool operator!() const
    {
        for (int i = 0; i < RANK; ++i) {
            if (idx[i]) {
                return false;
            }
        }
        return true;
    }

    // Element-wise addition
    __forceinline__ [aicore]
    Coord operator+(Coord const &b) const
    {
        Coord c;
        for (int i = 0; i < RANK; ++i) {
            c.idx[i] = idx[i] + b.idx[i];
        }
        return c;
    }

    // Add a scalar to each element
    __forceinline__ [aicore]
    Coord operator+(const Index val) const
    {
        Coord c;
        for (int i = 0; i < RANK; ++i) {
            c.idx[i] = idx[i] + val;
        }
        return c;
    }

    // Element-wise subtraction
    __forceinline__ [aicore]
    Coord operator-(Coord const &b) const
    {
        Coord c;
        for (int i = 0; i < RANK; i++) {
            c.idx[i] = idx[i] - b.idx[i];
        }
        return c;
    }

    // Subtract a scalar from each element
    __forceinline__ [aicore]
    Coord operator-(Index const val) const
    {
        Coord c;
        for (int i = 0; i < RANK; ++i) {
            c.idx[i] = idx[i] - val;
        }
        return c;
    }

    // Element-wise multiply
    __forceinline__ [aicore]
    Coord operator*(Coord const &b) const
    {
        Coord c;
        for (int i = 0; i < RANK; i++) {
            c.idx[i] = idx[i] * b.idx[i];
        }
        return c;
    }

    // Element-wise division
    __forceinline__ [aicore]
    Coord operator/(Coord const &b) const
    {
        Coord c;
        for (int i = 0; i < RANK; i++) {
            c.idx[i] = idx[i] / b.idx[i];
        }
        return c;
    }

    // Element-wise mod
    __forceinline__ [aicore]
    Coord operator%(Coord const &b) const
    {
        Coord c;
        for (int i = 0; i < RANK; i++) {
            c.idx[i] = idx[i] % b.idx[i];
        }
        return c;
    }

    // In-place addition
    __forceinline__ [aicore]
    Coord &operator+=(Coord const &b)
    {
        for (int i = 0; i < RANK; ++i) {
            idx[i] += b.idx[i];
        }
        return *this;
    }

    // In-place equal
    __forceinline__ [aicore]
    bool operator==(Coord const &b) const
    {
        for (int i = 0; i < RANK; ++i) {
            if (idx[i] != b.idx[i]) {
                return false;
            }
        }
        return true;
    }

    // In-place equal
    __forceinline__ [aicore]
    bool operator==(Index const val) const
    {
        for (int i = 0; i < RANK; ++i) {
            if (idx[i] != val) {
                return false;
            }
        }
        return true;
    }

    // Member acces operator
    __forceinline__ [aicore]
    Index &operator[](int dim)
    {
        return idx[dim];
    }

    // Member access operator
    __forceinline__ [aicore]
    Index const &operator[](int dim) const
    {
        return idx[dim];
    }

    // Gets the index of a given Coord element
    template <int DIM>
    __forceinline__ [aicore]
    Index &At()
    {
        return idx[DIM];
    }

    // Access via index; may limit unrolling potential
    __forceinline__ [aicore]
    Index &At(int dim)
    {
        return idx[dim];
    }

    // Gets the index of a given Coord element
    template <int DIM>
    __forceinline__ [aicore]
    Index const &At() const
    {
        return idx[DIM];
    }

    // Access via index; may limit unrolling potential
    __forceinline__ [aicore]
    Index const &At(int dim) const
    {
        return idx[dim];
    }

    template <int... Is>
    __forceinline__ [aicore]
    auto GetCoordByAxis() const
    {
        Index idx_[sizeof...(Is)]{idx[Is]...};
        return Coord<sizeof...(Is), Index, LongIndex>{idx_};
    }

    __forceinline__ [aicore]
    static Coord Min(Coord const &a, Coord const &b)
    {
        Coord res;
        for (int i = 0; i < RANK; ++i) {
            res[i] = a[i] < b[i] ? a[i] : b[i];
        }
        return res;
    }

private:
  Index idx[RANK];
};

// Helper to make a 1-element coordinate
template <class T>
__forceinline__ [aicore] constexpr
Coord<1, T> MakeCoord(T dim0)
{
    T values[1] = {dim0};
    return Coord<1, T>(values);
}

/// Helper to make a 2-element coordinate
template <class T>
__forceinline__ [aicore] constexpr
Coord<2, T> MakeCoord(T dim0, T dim1)
{
    T values[2] = {dim0, dim1};
    return Coord<2, T>(values);
}

/// Helper to make a 3-element coordinate
template <class T>
__forceinline__ [aicore] constexpr
Coord<3, T> MakeCoord(T dim0, T dim1, T dim2)
{
    T values[3] = {dim0, dim1, dim2};
    return Coord<3, T>(values);
}

/// Helper to make a 4-element coordinate
template <class T>
__forceinline__ [aicore] constexpr
Coord<4, T> MakeCoord(T dim0, T dim1, T dim2, T dim3)
{
    T values[4] = {dim0, dim1, dim2, dim3};
    return Coord<4, T>(values);
}

}  // namespace Catlass


namespace Catlass::Epilogue::Tile {

template <
    /// Tag indicating architecture
    class ArchTag_,
    /// Compute data type
    class ComputeType_,
    /// Length of the compute buffer
    uint32_t COMPUTE_LENGTH_
>
struct TileElemWiseAdd {
    using ArchTag = ArchTag_;
    using ElementCompute = typename ComputeType_::Element;

    static constexpr uint32_t COMPUTE_LENGTH = COMPUTE_LENGTH_;

    __forceinline__ [aicore]
    TileElemWiseAdd() {}

    __forceinline__ [aicore]
    void operator()(
        AscendC::LocalTensor<ElementCompute> const &ubOut,
        AscendC::LocalTensor<ElementCompute> const &ubIn0,
        AscendC::LocalTensor<ElementCompute> const &ubIn1
    )
    {
        // Do the calculation
        AscendC::Add(ubOut, ubIn0, ubIn1, COMPUTE_LENGTH);
    }
};

} // namespace Catlass::Epilogue::Tile






namespace Catlass::Arch {

constexpr uint32_t MAX_REVERSE_DEPTH = 16;

using FlagID = uint16_t;
constexpr FlagID AIV_INTER_BLOCK_BARRIER = 8;
constexpr FlagID AIC_INTER_BLOCK_BARRIER = 9;
constexpr FlagID AIV_INTER_SUBBLOCK_BARRIER = 10;
constexpr FlagID FFTS_MAX_FLAG = 7;

struct CrossCoreFlag {
    __forceinline__ [aicore]
    CrossCoreFlag() : id(0) {}

    __forceinline__ [aicore]
    CrossCoreFlag(FlagID id) : id(id) {}

    FlagID id;
};

template <uint32_t REVERSE_DEPTH_ = MAX_REVERSE_DEPTH>
struct CrossCoreFlagWithReverse {
    __forceinline__ [aicore]
    CrossCoreFlagWithReverse() : id(0), reverseId(0) {}

    __forceinline__ [aicore]
    CrossCoreFlagWithReverse(FlagID id, FlagID reverseId) : id(id), reverseId(reverseId) {}

    FlagID id;
    FlagID reverseId;
    uint32_t count{ 0 };
};

template <uint8_t MODE, int32_t CORE_TYPE>
struct BarrierFlag {
    static_assert(MODE != MODE, "Unsupported cross core barrier flag, can not find the specialization.");
};

template <>
struct BarrierFlag<0x0, AscendC::AIV> {
    static constexpr FlagID ID = AIV_INTER_BLOCK_BARRIER;
};

template <>
struct BarrierFlag<0x0, AscendC::AIC> {
    static constexpr FlagID ID = AIC_INTER_BLOCK_BARRIER;
};

template <>
struct BarrierFlag<0x1, AscendC::AIV> {
    static constexpr FlagID ID = AIV_INTER_SUBBLOCK_BARRIER;
};

template <uint8_t MODE, pipe_t PIPE>
__forceinline__ [aicore]
void CrossCoreBarrier()
{
    constexpr FlagID flagId = BarrierFlag<MODE, g_coreType>::ID;
    AscendC::CrossCoreSetFlag<MODE, PIPE>(flagId);
    AscendC::CrossCoreWaitFlag(flagId);
}

template <uint8_t MODE, pipe_t PIPE>
__forceinline__ [aicore]
void CrossCoreSetFlag(CrossCoreFlag &flag)
{
    AscendC::CrossCoreSetFlag<MODE, PIPE>(flag.id);
}

__forceinline__ [aicore]
void CrossCoreWaitFlag(CrossCoreFlag &flag)
{
    AscendC::CrossCoreWaitFlag(flag.id);
}

template <uint8_t MODE, pipe_t PIPE, uint32_t REVERSE_DEPTH>
__forceinline__ [aicore]
void CrossCoreSetFlagWithReverse(CrossCoreFlagWithReverse<REVERSE_DEPTH> &flag)
{
    AscendC::CrossCoreSetFlag<MODE, PIPE>(flag.id);
    if (++flag.count >= REVERSE_DEPTH) {
        AscendC::CrossCoreWaitFlag(flag.reverseId);
        flag.count = 0;
    }
}

template <uint8_t MODE, pipe_t PIPE, uint32_t REVERSE_DEPTH>
__forceinline__ [aicore]
void CrossCoreWaitFlagWithReverse(CrossCoreFlagWithReverse<REVERSE_DEPTH> &flag)
{
    AscendC::CrossCoreWaitFlag(flag.id);
    if (++flag.count >= REVERSE_DEPTH) {
        AscendC::CrossCoreSetFlag<MODE, PIPE>(flag.reverseId);
        flag.count = 0;
    }
}

}  // namespace Catlass::Arch






namespace Catlass::Arch {

struct PositionGM {
    static constexpr AscendC::TPosition POSITION = AscendC::TPosition::GM;
};

struct PositionL1 {
    static constexpr AscendC::TPosition POSITION = AscendC::TPosition::A1;
};

struct PositionL0A {
    static constexpr AscendC::TPosition POSITION = AscendC::TPosition::A2;
};

struct PositionL0B {
    static constexpr AscendC::TPosition POSITION = AscendC::TPosition::B2;
};

struct PositionL0C {
    static constexpr AscendC::TPosition POSITION = AscendC::TPosition::CO1;
};

struct PositionUB {
    static constexpr AscendC::TPosition POSITION = AscendC::TPosition::VECCALC;
};

} // namespace Catlass::Arch






namespace tla {

//
// Common Operations
//

template <class T, class U,
          __TLA_REQUIRES(std::is_arithmetic<T>::value &&
                         std::is_arithmetic<U>::value)>
__forceinline__ [aicore] constexpr
auto
max(T const& t, U const& u) {
    return t < u ? u : t;
}

template <class T, class U,
          __TLA_REQUIRES(std::is_arithmetic<T>::value &&
                         std::is_arithmetic<U>::value)>
__forceinline__ [aicore] constexpr
auto
min(T const& t, U const& u) {
    return t < u ? t : u;
}

} // namespace tla






namespace Catlass::Epilogue::Tile {

template <
    /// Tag indicating architecture
    class ArchTag_,
    /// Compute data type
    class DstType_,
    class SrcType_,
    /// Length of the compute buffer
    class TileShape_
>
struct TileCast {
    using ArchTag = ArchTag_;
    using ElementDst = typename DstType_::Element;
    using ElementSrc = typename SrcType_::Element;
    using TileShape = TileShape_;

    __forceinline__ [aicore]
    TileCast() {}

    __forceinline__ [aicore]
    void operator()(
        AscendC::LocalTensor<ElementDst> const &ubOut,
        AscendC::LocalTensor<ElementSrc> const &ubIn
    )
    {
        AscendC::Cast(ubOut, ubIn, AscendC::RoundMode::CAST_RINT, TileShape::COUNT);
    }
};

} // namespace Catlass::Epilogue::Tile

namespace Catlass {

/// Shape of a matrix multiply-add operation
template <
    /// Rows of matrix product
    uint32_t M_ = 1,
    /// Columns of matrix product
    uint32_t N_ = 1,
    /// Inner dimension of matrix product
    uint32_t K_ = 1
>
struct GemmShape {
    static constexpr uint32_t M = M_;
    static constexpr uint32_t N = N_;
    static constexpr uint32_t K = K_;

    static constexpr int64_t MN = M * N;
    static constexpr int64_t MK = M * K;
    static constexpr int64_t KN = N * K;
    static constexpr int64_t MNK = M * N * K;

    static constexpr int64_t COUNT = MNK;

    /// Returns a Coord object
    __forceinline__ [aicore]
    static Coord<3> ToCoord()
    {
        return MakeCoord(M, N, K);
    }

    __forceinline__ [aicore]
    static Coord<2> ToCoordMN()
    {
        return MakeCoord(M, N);
    }

    __forceinline__ [aicore]
    static Coord<2> ToCoordMK()
    {
        return MakeCoord(M, K);
    }

    __forceinline__ [aicore]
    static Coord<2> ToCoordKN()
    {
        return MakeCoord(K, N);
    }
};

struct GemmCoord : public Coord<3, uint32_t> {
  using Index = uint32_t;
  using Base = Coord<3, Index>;
  static constexpr int M_INDEX = 0;
  static constexpr int N_INDEX = 1;

    /// Gemm K dimension - inner dimension of the Gemm problem
    static constexpr int K_INDEX = 2;

    /// Default ctor
    __forceinline__ [aicore]
    GemmCoord() {}

    /// Constructs from Coord<3> and a batch
    __forceinline__ [aicore]
    GemmCoord(Coord<3, Index> const &coord) : Base(coord) {}

    /// Helper to construct from a K, N, M, batch variables
    __forceinline__ [aicore]
    GemmCoord(Index m, Index n, Index k) : Base(MakeCoord(m, n, k)) {}

    /// Returns the Gemm M coordinate
    __forceinline__ [aicore]
    Index const &m() const
    {
        return this->At(M_INDEX);
    }

    /// Returns reference to the Gemm M coordinate
    __forceinline__ [aicore]
    Index &m()
    {
        return this->At(M_INDEX);
    }

    /// Returns the Gemm N coordinate
    __forceinline__ [aicore]
    Index const &n() const
    {
        return this->At(N_INDEX);
    }

    /// Returns reference to the Gemm N coordinate
    __forceinline__ [aicore]
    Index &n()
    {
        return this->At(N_INDEX);
    }

    /// Returns the Gemm K coordinate
    __forceinline__ [aicore]
    Index const &k() const
    {
        return this->At(K_INDEX);
    }

    /// Returns reference to the Gemm K coordinate
    __forceinline__ [aicore]
    Index &k()
    {
        return this->At(K_INDEX);
    }

    __forceinline__ [aicore]
    auto GetCoordMN() const
    {
        return this->GetCoordByAxis<M_INDEX, N_INDEX>();
    }

    __forceinline__ [aicore]
    auto GetCoordMK() const
    {
        return this->GetCoordByAxis<M_INDEX, K_INDEX>();
    }

    __forceinline__ [aicore]
    auto GetCoordKN() const
    {
        return this->GetCoordByAxis<K_INDEX, N_INDEX>();
    }
};

} // namespace Catlass

namespace Catlass {

template <
    uint32_t ROW_ = 1,
    uint32_t COLUMN_ = 1
>
struct MatrixShape {
    static constexpr uint32_t ROW = ROW_;
    static constexpr uint32_t COLUMN = COLUMN_;

    static constexpr int64_t COUNT = ROW * COLUMN;

    __forceinline__ [aicore]
    static Coord<2> ToCoord()
    {
        return MakeCoord(ROW, COLUMN);
    }
};

struct MatrixCoord : public Coord<2, uint32_t> {
  using Index = uint32_t;
  using Base = Coord<2, Index>;
  using LongIndex = typename Base::LongIndex;
  static constexpr uint32_t ROW_INDEX = 0;
  static constexpr uint32_t COLUMN_INDEX = 1;

    /// Default ctor
    __forceinline__ [aicore]
    MatrixCoord() {}

    /// Constructs from Coord<2>
    __forceinline__ [aicore]
    MatrixCoord(Coord<2, Index> const &coord) : Base(coord) {}

    /// Helper to construct from a row and column
    __forceinline__ [aicore]
    MatrixCoord(Index row, Index column) : Base(MakeCoord(row, column)) {}

    /// Helper to construct from a row and column, which are LongIndex based
    __forceinline__ [aicore]
    MatrixCoord(LongIndex row, LongIndex column) : Base(MakeCoord(Index(row), Index(column))) {}

    /// Returns the row of the coordinate
    __forceinline__ [aicore]
    Index const &row() const { return this->At(ROW_INDEX); }

    /// Returns the row of the coordinate
    __forceinline__ [aicore]
    Index &row() { return this->At(ROW_INDEX); }

    /// Returns the column of the coordinate
    __forceinline__ [aicore]
    Index const &column() const { return this->At(COLUMN_INDEX); }

    /// Returns the column of the coordinate
    __forceinline__ [aicore]
    Index &column() { return this->At(COLUMN_INDEX); }

    /// Element-wise addition
    __forceinline__ [aicore]
    MatrixCoord operator+(Base const &b) const
    {
        return MatrixCoord(Base::operator+(b));
    }

    /// In-place addition
    __forceinline__ [aicore]
    MatrixCoord &operator+=(Base const &b)
    {
        Base::operator+=(b);
        return *this;
    }
};
} // namespace Catlass

namespace Catlass::layout {

struct RowMajor {
public:
  static constexpr int RANK = 2;
  using Index = uint32_t;
  using LongIndex = int64_t;
  using Shape = Coord<RANK, Index>;
  using Stride = Coord<RANK, LongIndex>;

public:
  __forceinline__ [aicore]
  RowMajor(Index rows = 0, Index cols = 0)
      : shape_(MakeCoord(rows, cols)), stride_(MakeCoord(LongIndex(cols), LongIndex(1))) {}

  __forceinline__ [aicore]
  RowMajor(Index rows, Index cols, LongIndex ldm)
      : shape_(MakeCoord(rows, cols)), stride_(MakeCoord(ldm, LongIndex(1))) {}

    /// Ctor
    __forceinline__ [aicore]
    RowMajor(Shape shape, Stride stride) : shape_(shape), stride_(stride) {}

    template <class Element>
    __forceinline__ [aicore]
    static RowMajor MakeLayoutInUb(MatrixCoord const &shape)
    {
        return RowMajor(shape.row(), shape.column(), RoundUp<BYTE_PER_C0 / sizeof(Element)>(shape.column()));
    }

    /// Returns the offset of a coordinate in linear memory.
    /// Assumes coordinate has convention (row, column)
    __forceinline__ [aicore]
    LongIndex GetOffset(MatrixCoord const &coord) const
    {
        return LongIndex(coord.row()) * stride_[0] + LongIndex(coord.column());
    }

    /// Returns the layout of a tile.
    __forceinline__ [aicore]
    RowMajor GetTileLayout(MatrixCoord const &tileShape) const
    {
        return RowMajor(tileShape, stride());
    }

    /// Returns the shape of the layout
    __forceinline__ [aicore]
    Shape shape() const
    {
        return shape_;
    }

    /// Returns the shape of the layout
    __forceinline__ [aicore]
    Shape &shape()
    {
        return shape_;
    }

    /// Returns the shape of the layout
    __forceinline__ [aicore]
    typename Shape::Index shape(int idx) const
    {
        return shape_[idx];
    }

    /// Returns the shape of the layout
    __forceinline__ [aicore]
    typename Shape::Index &shape(int idx)
    {
        return shape_[idx];
    }

    /// Returns the stride of the layout
    __forceinline__ [aicore]
    Stride stride() const
    {
        return stride_;
    }

    /// Returns the stride of the layout
    __forceinline__ [aicore]
    Stride &stride()
    {
        return stride_;
    }

    /// Returns the stride of the layout
    __forceinline__ [aicore]
    typename Stride::Index stride(int idx) const
    {
        return stride_[idx];
    }

    /// Returns the stride of the layout
    __forceinline__ [aicore]
    typename Stride::Index &stride(int idx)
    {
        return stride_[idx];
    }

private:
  Shape shape_;
  Stride stride_;
};

struct ColumnMajor {
public:
  static constexpr int RANK = 2;
  using Index = uint32_t;
  using LongIndex = int64_t;
  using Shape = Coord<RANK, Index>;
  using Stride = Coord<RANK, LongIndex>;

public:
    // Methods

    /// Constructor
    __forceinline__ [aicore]
    ColumnMajor(Index rows = 0, Index cols = 0)
        : shape_(MakeCoord(rows, cols)), stride_(MakeCoord(LongIndex(1), LongIndex(rows))) {}

    /// Constructor
    __forceinline__ [aicore]
    ColumnMajor(Index rows, Index cols, LongIndex ldm)
        : shape_(MakeCoord(rows, cols)), stride_(MakeCoord(LongIndex(1), ldm)) {}

    /// Ctor
    __forceinline__ [aicore]
    ColumnMajor(Shape shape, Stride stride) : shape_(shape), stride_(stride) {}

    /// Returns the offset of a coordinate in linear memory.
    /// Assumes coordinate has convention (row, column)
    __forceinline__ [aicore]
    LongIndex GetOffset(MatrixCoord const &coord) const
    {
        return LongIndex(coord.row()) + LongIndex(coord.column()) * stride_[1];
    }

    /// Returns the layout of a tile.
    __forceinline__ [aicore]
    ColumnMajor GetTileLayout(MatrixCoord const &tileShape) const
    {
        return ColumnMajor(tileShape, stride());
    }

    /// Returns the shape of the layout
    __forceinline__ [aicore]
    Shape shape() const
    {
        return shape_;
    }

    /// Returns the shape of the layout
    __forceinline__ [aicore]
    Shape &shape()
    {
        return shape_;
    }

    /// Returns the shape of the layout
    __forceinline__ [aicore]
    typename Shape::Index shape(int idx) const
    {
        return shape_[idx];
    }

    /// Returns the shape of the layout
    __forceinline__ [aicore]
    typename Shape::Index &shape(int idx)
    {
        return shape_[idx];
    }

    /// Returns the stride of the layout
    __forceinline__ [aicore]
    Stride stride() const
    {
        return stride_;
    }

    /// Returns the stride of the layout
    __forceinline__ [aicore]
    Stride &stride()
    {
        return stride_;
    }

    /// Returns the stride of the layout
    __forceinline__ [aicore]
    typename Stride::Index stride(int idx) const
    {
        return stride_[idx];
    }

    /// Returns the stride of the layout
    __forceinline__ [aicore]
    typename Stride::Index &stride(int idx)
    {
        return stride_[idx];
    }

private:
  Shape shape_;
  Stride stride_;
};

struct nZ {
public:
  static constexpr int RANK = 4;
  using Index = uint32_t;
  using LongIndex = int64_t;
  static constexpr int ORG_SHAPE_RANK = 2;
  using OrgShape = Coord<ORG_SHAPE_RANK, Index>;
  using Shape = Coord<RANK, Index>;
  using Stride = Coord<RANK, LongIndex>;

public:
    // Methods

    /// Constructor
    __forceinline__ [aicore] constexpr
    nZ(Index orgRows = 0,                 /// Number of rows of origin matrices
       Index orgCols = 0,                 /// Number of cols of origin matrices
       Index rowsInFractal = 0,           /// Number of rows inside the fractal
       Index rowsByFractal = 0,           /// number of rows by the fractal
       Index colsInFractal = 0,           /// number of cols inside the fractal
       Index colsByFractal = 0,           /// number of cols by the fractal
       LongIndex strideRowsInFractal = 0, /// number of elements between adjacent rows inside the fractal
       LongIndex strideRowsByFractal = 0, /// number of elements between adjacent fractal rows
       LongIndex strideColsInFractal = 0, /// number of elements between adjacent cols inside the fractal
       LongIndex strideColsByFractal = 0) /// number of elements between adjacent fractal cols
        : orgShape_(MakeCoord(orgRows, orgCols)),
          shape_(MakeCoord(rowsInFractal, rowsByFractal, colsInFractal, colsByFractal)),
          stride_(MakeCoord(strideRowsInFractal, strideRowsByFractal, strideColsInFractal, strideColsByFractal)) {}

    /// Ctor
    __forceinline__ [aicore] constexpr
    nZ(OrgShape orgShape, Shape shape, Stride stride) : orgShape_(orgShape), shape_(shape), stride_(stride) {}

    /// Make the layout of a coordinate (row, column)
    template <class Element>
    __forceinline__ [aicore] constexpr
    static nZ MakeLayout(Index orgRows, Index orgCols)
    {
        constexpr uint32_t ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(Element);
        constexpr uint32_t ELE_NUM_PER_FRACTAL = BYTE_PER_FRACTAL / sizeof(Element);
        Index rowsRound = RoundUp<ELE_NUM_PER_C0>(orgRows);
        Index colsRound = RoundUp<C0_NUM_PER_FRACTAL>(orgCols);
        return nZ(orgRows,
                  orgCols,
                  ELE_NUM_PER_C0,
                  rowsRound / ELE_NUM_PER_C0,
                  C0_NUM_PER_FRACTAL,
                  colsRound / C0_NUM_PER_FRACTAL,
                  1,
                  colsRound * ELE_NUM_PER_C0,
                  ELE_NUM_PER_C0,
                  ELE_NUM_PER_FRACTAL);
    }

    /// Returns the offset of a coordinate in linear memory.
    /// Assumes coordinate has convention (row, column)
    __forceinline__ [aicore]
    LongIndex GetOffset(MatrixCoord const &coord) const
    {
        return LongIndex(coord.row()) / shape_[0] * stride_[1] + LongIndex(coord.column()) / shape_[2] * stride_[3] +
            (LongIndex(coord.row()) % shape_[0]) * stride_[0] + (LongIndex(coord.column()) % shape_[2]) * stride_[2];
    }

    /// Returns the layout of a tile.
    __forceinline__ [aicore]
    nZ GetTileLayout(MatrixCoord const &tileOriShape) const
    {
        auto tileShape = MakeCoord(
            shape(0), CeilDiv(tileOriShape.row(), shape(0)),
            shape(2), CeilDiv(tileOriShape.column(), shape(2))
        );
        return nZ(tileOriShape, tileShape, stride());
    }

    /// Returns the origin shape of the layout
    __forceinline__ [aicore]
    typename OrgShape::Index orgShape(int idx) const
    {
        return orgShape_[idx];
    }

    /// Returns the origin shape of the layout
    __forceinline__ [aicore]
    typename OrgShape::Index &orgShape(int idx)
    {
        return orgShape_[idx];
    }

    /// Returns the shape of the layout
    __forceinline__ [aicore]
    Shape shape() const
    {
        return shape_;
    }

    /// Returns the shape of the layout
    __forceinline__ [aicore]
    Shape &shape()
    {
        return shape_;
    }

    /// Returns the shape of the layout
    __forceinline__ [aicore]
    typename Shape::Index shape(int idx) const
    {
        return shape_[idx];
    }

    /// Returns the shape of the layout
    __forceinline__ [aicore]
    typename Shape::Index &shape(int idx)
    {
        return shape_[idx];
    }

    /// Returns the stride of the layout
    __forceinline__ [aicore]
    Stride stride() const
    {
        return stride_;
    }

    /// Returns the stride of the layout
    __forceinline__ [aicore]
    Stride &stride()
    {
        return stride_;
    }

    /// Returns the stride of the layout
    __forceinline__ [aicore]
    typename Stride::Index stride(int idx) const
    {
        return stride_[idx];
    }

    /// Returns the stride of the layout
    __forceinline__ [aicore]
    typename Stride::Index &stride(int idx)
    {
        return stride_[idx];
    }

private:
  OrgShape orgShape_;
  Shape shape_;
  Stride stride_;
};

struct zN {
public:
  static constexpr int RANK = 4;
  using Index = uint32_t;
  using LongIndex = int64_t;
  static constexpr int ORG_SHAPE_RANK = 2;
  using OrgShape = Coord<ORG_SHAPE_RANK, Index>;
  using Shape = Coord<RANK, Index>;
  using Stride = Coord<RANK, LongIndex>;

public:
    // Methods

    /// Constructor
    __forceinline__ [aicore] constexpr
    zN(Index orgRows = 0,                 /// Number of rows of origin matrices
       Index orgCols = 0,                 /// Number of cols of origin matrices
       Index rowsInFractal = 0,           /// Number of rows inside the fractal
       Index rowsByFractal = 0,           /// number of rows by the fractal
       Index colsInFractal = 0,           /// number of cols inside the fractal
       Index colsByFractal = 0,           /// number of cols by the fractal
       LongIndex strideRowsInFractal = 0, /// number of elements between adjacent rows inside the fractal
       LongIndex strideRowsByFractal = 0, /// number of elements between adjacent fractal rows
       LongIndex strideColsInFractal = 0, /// number of elements between adjacent cols inside the fractal
       LongIndex strideColsByFractal = 0) /// number of elements between adjacent fractal cols
        : orgShape_(MakeCoord(orgRows, orgCols)),
          shape_(MakeCoord(rowsInFractal, rowsByFractal, colsInFractal, colsByFractal)),
          stride_(MakeCoord(strideRowsInFractal, strideRowsByFractal, strideColsInFractal, strideColsByFractal)) {}

    /// Ctor
    __forceinline__ [aicore] constexpr
    zN(OrgShape orgShape, Shape shape, Stride stride) : orgShape_(orgShape), shape_(shape), stride_(stride) {}

    /// Make the layout of a coordinate (row, column)
    template <class Element>
    __forceinline__ [aicore] constexpr
    static zN MakeLayout(Index orgRows, Index orgCols)
    {
        constexpr uint32_t ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(Element);
        constexpr uint32_t ELE_NUM_PER_FRACTAL = BYTE_PER_FRACTAL / sizeof(Element);
        Index rowsRound = RoundUp<C0_NUM_PER_FRACTAL>(orgRows);
        Index colsRound = RoundUp<ELE_NUM_PER_C0>(orgCols);
        return zN(orgRows,
                  orgCols,
                  C0_NUM_PER_FRACTAL,
                  rowsRound / C0_NUM_PER_FRACTAL,
                  ELE_NUM_PER_C0,
                  colsRound / ELE_NUM_PER_C0,
                  ELE_NUM_PER_C0,
                  ELE_NUM_PER_FRACTAL,
                  1,
                  rowsRound * ELE_NUM_PER_C0);
    }

    __forceinline__ [aicore]
    static zN MakeLayoutInL0C(MatrixCoord const &shape)
    {
        return zN(shape.row(),
                  shape.column(),
                  C0_NUM_PER_FRACTAL,
                  CeilDiv<C0_NUM_PER_FRACTAL>(shape.row()),
                  C0_NUM_PER_FRACTAL,
                  CeilDiv<C0_NUM_PER_FRACTAL>(shape.column()),
                  C0_NUM_PER_FRACTAL,
                  C0_NUM_PER_FRACTAL * C0_NUM_PER_FRACTAL,
                  1,
                  RoundUp<C0_NUM_PER_FRACTAL>(shape.row()) * C0_NUM_PER_FRACTAL);
    }

    /// Returns the offset of a coordinate in linear memory.
    /// Assumes coordinate has convention (row, column)
    __forceinline__ [aicore]
    LongIndex GetOffset(MatrixCoord const &coord) const
    {
        return LongIndex(coord.row()) / shape_[0] * stride_[1] + LongIndex(coord.column()) / shape_[2] * stride_[3] +
            (LongIndex(coord.row()) % shape_[0]) * stride_[0] + (LongIndex(coord.column()) % shape_[2]) * stride_[2];
    }

    /// Returns the layout of a tile.
    __forceinline__ [aicore]
    zN GetTileLayout(MatrixCoord const &tileOriShape) const
    {
        auto tileShape = MakeCoord(
            shape(0), CeilDiv(tileOriShape.row(), shape(0)),
            shape(2), CeilDiv(tileOriShape.column(), shape(2))
        );
        return zN(tileOriShape, tileShape, stride());
    }

    /// Returns the origin shape of the layout
    __forceinline__ [aicore]
    typename OrgShape::Index orgShape(int idx) const
    {
        return orgShape_[idx];
    }

    /// Returns the origin shape of the layout
    __forceinline__ [aicore]
    typename OrgShape::Index &orgShape(int idx)
    {
        return orgShape_[idx];
    }

    /// Returns the shape of the layout
    __forceinline__ [aicore]
    Shape shape() const
    {
        return shape_;
    }

    /// Returns the shape of the layout
    __forceinline__ [aicore]
    Shape &shape()
    {
        return shape_;
    }

    /// Returns the shape of the layout
    __forceinline__ [aicore]
    typename Shape::Index shape(int idx) const
    {
        return shape_[idx];
    }

    /// Returns the shape of the layout
    __forceinline__ [aicore]
    typename Shape::Index &shape(int idx)
    {
        return shape_[idx];
    }

    /// Returns the stride of the layout
    __forceinline__ [aicore]
    Stride stride() const
    {
        return stride_;
    }

    /// Returns the stride of the layout
    __forceinline__ [aicore]
    Stride &stride()
    {
        return stride_;
    }

    /// Returns the stride of the layout
    __forceinline__ [aicore]
    typename Stride::Index stride(int idx) const
    {
        return stride_[idx];
    }

    /// Returns the stride of the layout
    __forceinline__ [aicore]
    typename Stride::Index &stride(int idx)
    {
        return stride_[idx];
    }

private:
  OrgShape orgShape_;
  Shape shape_;
  Stride stride_;
};

struct zZ {
public:
  static constexpr int RANK = 4;
  using Index = uint32_t;
  using LongIndex = int64_t;
  static constexpr int ORG_SHAPE_RANK = 2;
  using OrgShape = Coord<ORG_SHAPE_RANK, Index>;
  using Shape = Coord<RANK, Index>;
  using Stride = Coord<RANK, LongIndex>;

public:
    // Methods

    /// Constructor
    __forceinline__ [aicore] constexpr
    zZ(Index orgRows = 0,                 /// Number of rows of origin matrices
       Index orgCols = 0,                 /// Number of cols of origin matrices
       Index rowsInFractal = 0,           /// Number of rows inside the fractal
       Index rowsByFractal = 0,           /// number of rows by the fractal
       Index colsInFractal = 0,           /// number of cols inside the fractal
       Index colsByFractal = 0,           /// number of cols by the fractal
       LongIndex strideRowsInFractal = 0, /// number of elements between adjacent rows inside the fractal
       LongIndex strideRowsByFractal = 0, /// number of elements between adjacent fractal rows
       LongIndex strideColsInFractal = 0, /// number of elements between adjacent cols inside the fractal
       LongIndex strideColsByFractal = 0) /// number of elements between adjacent fractal cols
        : orgShape_(MakeCoord(orgRows, orgCols)),
          shape_(MakeCoord(rowsInFractal, rowsByFractal, colsInFractal, colsByFractal)),
          stride_(MakeCoord(strideRowsInFractal, strideRowsByFractal, strideColsInFractal, strideColsByFractal)) {}

    /// Ctor
    __forceinline__ [aicore] constexpr
    zZ(OrgShape orgShape, Shape shape, Stride stride) : orgShape_(orgShape), shape_(shape), stride_(stride) {}

    /// Make the layout of a coordinate (row, column)
    template <class Element>
    __forceinline__ [aicore] constexpr
    static zZ MakeLayout(Index orgRows, Index orgCols)
    {
        constexpr uint32_t ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(Element);
        constexpr uint32_t ELE_NUM_PER_FRACTAL = BYTE_PER_FRACTAL / sizeof(Element);
        Index rowsRound = RoundUp<C0_NUM_PER_FRACTAL>(orgRows);
        Index colsRound = RoundUp<ELE_NUM_PER_C0>(orgCols);
        return zZ(orgRows,
                  orgCols,
                  C0_NUM_PER_FRACTAL,
                  rowsRound / C0_NUM_PER_FRACTAL,
                  ELE_NUM_PER_C0,
                  colsRound / ELE_NUM_PER_C0,
                  ELE_NUM_PER_C0,
                  colsRound * C0_NUM_PER_FRACTAL,
                  1,
                  ELE_NUM_PER_FRACTAL);
    }

    /// Returns the offset of a coordinate in linear memory.
    /// Assumes coordinate has convention (row, column)
    __forceinline__ [aicore]
    LongIndex GetOffset(MatrixCoord const &coord) const
    {
        return LongIndex(coord.row()) / shape_[0] * stride_[1] + LongIndex(coord.column()) / shape_[2] * stride_[3];
    }

    /// Returns the origin shape of the layout
    __forceinline__ [aicore]
    typename OrgShape::Index orgShape(int idx) const
    {
        return orgShape_[idx];
    }

    /// Returns the origin shape of the layout
    __forceinline__ [aicore]
    typename OrgShape::Index &orgShape(int idx)
    {
        return orgShape_[idx];
    }

    /// Returns the shape of the layout
    __forceinline__ [aicore]
    Shape shape() const
    {
        return shape_;
    }

    /// Returns the shape of the layout
    __forceinline__ [aicore]
    Shape &shape()
    {
        return shape_;
    }

    /// Returns the shape of the layout
    __forceinline__ [aicore]
    typename Shape::Index shape(int idx) const
    {
        return shape_[idx];
    }

    /// Returns the shape of the layout
    __forceinline__ [aicore]
    typename Shape::Index &shape(int idx)
    {
        return shape_[idx];
    }

    /// Returns the stride of the layout
    __forceinline__ [aicore]
    Stride stride() const
    {
        return stride_;
    }

    /// Returns the stride of the layout
    __forceinline__ [aicore]
    Stride &stride()
    {
        return stride_;
    }

    /// Returns the stride of the layout
    __forceinline__ [aicore]
    typename Stride::Index stride(int idx) const
    {
        return stride_[idx];
    }

    /// Returns the stride of the layout
    __forceinline__ [aicore]
    typename Stride::Index &stride(int idx)
    {
        return stride_[idx];
    }

private:
    /// Origin Shape data member
    OrgShape orgShape_;

    /// Shape data member
    Shape shape_;

    /// Stride data member
    Stride stride_;
};

} // namespace Catlass::layout

namespace Catlass::Arch {

struct AtlasA2 {
  static constexpr uint32_t BIAS_SIZE = 1024;
  static constexpr uint32_t FIXBUF_SIZE = 7 * 1024;
  static constexpr uint32_t UB_SIZE = 192 * 1024;
  static constexpr uint32_t L1_SIZE = 512 * 1024;
  static constexpr uint32_t L0A_SIZE = 64 * 1024;
  static constexpr uint32_t L0B_SIZE = 64 * 1024;
  static constexpr uint32_t L0C_SIZE = 128 * 1024;
};

struct LocalTensorBufferBase {
public:
  template <class Element = half>
  __forceinline__ [aicore] AscendC::LocalTensor<Element>
  GetBufferByByte(const uint32_t offset) const {
    return tensor[offset].template ReinterpretCast<Element>();
  }

protected:
  __forceinline__ [aicore]
  LocalTensorBufferBase() = default;
  AscendC::LocalTensor<uint8_t> tensor;
};

template <class ArchTag, AscendC::TPosition Position> struct LocalTensorBuffer {
  static_assert(DEPENDENT_FALSE<ArchTag>,
                "Unsupported local tensor buffer, can not find the specialization.");
};

template <class ArchTag>
struct LocalTensorBuffer<ArchTag, AscendC::TPosition::A1> : LocalTensorBufferBase {
public:
  static constexpr AscendC::TPosition Position = AscendC::TPosition::A1;
  __forceinline__ [aicore]
  LocalTensorBuffer() {
    AscendC::TBuf<AscendC::TPosition::A1> tbufA1;
    GetTPipePtr()->InitBuffer(tbufA1, ArchTag::L1_SIZE);
    tensor = tbufA1.Get<uint8_t>();
  }
};

template <class ArchTag>
struct LocalTensorBuffer<ArchTag, AscendC::TPosition::A2> : LocalTensorBufferBase {
public:
  static constexpr AscendC::TPosition Position = AscendC::TPosition::A2;
  __forceinline__ [aicore]
  LocalTensorBuffer() {
    AscendC::TBuf<AscendC::TPosition::A2> tbufA2;
    GetTPipePtr()->InitBuffer(tbufA2, ArchTag::L0A_SIZE);
    tensor = tbufA2.Get<uint8_t>();
  }
};

template <class ArchTag>
struct LocalTensorBuffer<ArchTag, AscendC::TPosition::B2> : LocalTensorBufferBase {
public:
  static constexpr AscendC::TPosition Position = AscendC::TPosition::B2;
  __forceinline__ [aicore]
  LocalTensorBuffer() {
    AscendC::TBuf<AscendC::TPosition::B2> tbufB2;
    GetTPipePtr()->InitBuffer(tbufB2, ArchTag::L0B_SIZE);
    tensor = tbufB2.Get<uint8_t>();
  }
};

template <>
struct LocalTensorBuffer<Arch::AtlasA2, AscendC::TPosition::C2> : LocalTensorBufferBase {
public:
  using ArchTag = Arch::AtlasA2;
  static constexpr AscendC::TPosition Position = AscendC::TPosition::C2;
  __forceinline__ [aicore]
  LocalTensorBuffer() {
    AscendC::TBuf<AscendC::TPosition::C2> tbufC2;
    GetTPipePtr()->InitBuffer(tbufC2, ArchTag::BIAS_SIZE);
    tensor = tbufC2.Get<uint8_t>();
  }
};

template <class ArchTag>
struct LocalTensorBuffer<ArchTag, AscendC::TPosition::CO1> : LocalTensorBufferBase {
public:
  static constexpr AscendC::TPosition Position = AscendC::TPosition::CO1;
  __forceinline__ [aicore]
  LocalTensorBuffer() {
    AscendC::TBuf<AscendC::TPosition::CO1> tbufCO1;
    GetTPipePtr()->InitBuffer(tbufCO1, ArchTag::L0C_SIZE);
    tensor = tbufCO1.Get<uint8_t>();
  }
};

template <class ArchTag>
struct LocalTensorBuffer<ArchTag, AscendC::TPosition::VECCALC> : LocalTensorBufferBase {
public:
  static constexpr AscendC::TPosition Position = AscendC::TPosition::VECCALC;
  __forceinline__ [aicore]
  LocalTensorBuffer() {
    AscendC::TBuf<AscendC::TPosition::VECCALC> tbufVECCALC;
    GetTPipePtr()->InitBuffer(tbufVECCALC, ArchTag::UB_SIZE);
    tensor = tbufVECCALC.Get<uint8_t>();
  }
};

template <class ArchTag> struct Resource {
public:
  AscendC::TPipe pipe;
  LocalTensorBuffer<ArchTag, AscendC::TPosition::A1> l1Buf;
  LocalTensorBuffer<ArchTag, AscendC::TPosition::A2> l0ABuf;
  LocalTensorBuffer<ArchTag, AscendC::TPosition::B2> l0BBuf;
  LocalTensorBuffer<ArchTag, AscendC::TPosition::C2> btBuf;
  LocalTensorBuffer<ArchTag, AscendC::TPosition::CO1> l0CBuf;
  LocalTensorBuffer<ArchTag, AscendC::TPosition::VECCALC> ubBuf;

  __forceinline__ [aicore]
  Resource() { pipe.Destroy(); }
};

} // namespace Catlass::Arch

namespace Catlass::Gemm {
template <class Element_, class Layout_,
          AscendC::TPosition POSITION_ = AscendC::TPosition::GM>
struct GemmType {
  using Element = Element_;
  using Layout = Layout_;
  static constexpr AscendC::TPosition POSITION = POSITION_;
};

template <bool ENABLE_UNIT_FLAG_ = false, bool ENABLE_SHUFFLE_K_ = false,
          bool ENABLE_ABBA_ = false>
struct GemmAtlasA2 {
  using ArchTag = Arch::AtlasA2;
  static constexpr uint32_t STAGES = 2;
  static constexpr bool ENABLE_UNIT_FLAG = ENABLE_UNIT_FLAG_;
  static constexpr bool ENABLE_SHUFFLE_K = ENABLE_SHUFFLE_K_;
  static constexpr bool ENABLE_ABBA = ENABLE_ABBA_;
};
} // namespace Catlass::Gemm

namespace Catlass::Epilogue {
struct EpilogueAtlasA2Gemm {
  using ArchTag = Arch::AtlasA2;
};
} // namespace Catlass::Epilogue

namespace Catlass::Epilogue::Tile {

 template<
     class ArchTag_,
     class ComputeType_,
     uint32_t COMPUTE_LENGTH_
 >
 struct TileElemWiseMuls{
     using ArchTag = ArchTag_;
     using ElementCompute = typename ComputeType_::Element;

     static constexpr uint32_t COMPUTE_LENGTH = COMPUTE_LENGTH_;

     __forceinline__ [aicore]
     TileElemWiseMuls(){}

     __forceinline__ [aicore]
     void operator()(
         AscendC::LocalTensor<ElementCompute> dstLocal,
         AscendC::LocalTensor<ElementCompute> srcTensor,
         ElementCompute scalar
     ){
         AscendC::Muls(dstLocal, srcTensor, scalar, COMPUTE_LENGTH);
     }
 };

template <class DstType_, class SrcType_, class TileShape_>
struct TileCast<Catlass::Arch::AtlasA2, DstType_, SrcType_, TileShape_> {
  using ArchTag = Catlass::Arch::AtlasA2;
  using ElementDst = typename DstType_::Element;
  using ElementSrc = typename SrcType_::Element;
  using TileShape = TileShape_;

  __forceinline__ [aicore]
  TileCast() {}

  __forceinline__ [aicore]
  void operator()(AscendC::LocalTensor<ElementDst> const &ubOut,
                  AscendC::LocalTensor<ElementSrc> const &ubIn) {
    AscendC::Cast(ubOut, ubIn, AscendC::RoundMode::CAST_RINT, TileShape::COUNT);
  }
};

template <
    class ArchTag,
    class GmType
>
struct CopyGm2Ub {
    static_assert(DEPENDENT_FALSE<ArchTag>, "Unsupported copy gm to ub, can not find the specialization.");
};

template <typename Element>
struct CopyGm2Ub<Arch::AtlasA2, Gemm::GemmType<Element, layout::RowMajor>> {
    using LayoutSrc = layout::RowMajor;
    using LayoutDst = layout::RowMajor;

    static constexpr uint32_t ELE_NUM_PER_BLK = BYTE_PER_BLK / sizeof(Element);

    __forceinline__ [aicore]
    CopyGm2Ub() = default;

    __forceinline__ [aicore]
    void operator()(
        AscendC::LocalTensor<Element> const &dstTensor,
        AscendC::GlobalTensor<Element> const &srcTensor,
        layout::RowMajor const &layoutDst,
        layout::RowMajor const &layoutSrc)
    {
        AscendC::DataCopyExtParams dataCopyParams(
            layoutSrc.shape(0),
            layoutSrc.shape(1) * sizeof(Element),
            (layoutSrc.stride(0) - layoutSrc.shape(1)) * sizeof(Element),
            (layoutDst.stride(0) - layoutDst.shape(1)) / ELE_NUM_PER_BLK,
            0
        );
        AscendC::DataCopyPadExtParams<Element> padParams(false, 0, 0, 0);
        AscendC::DataCopyPad(dstTensor, srcTensor, dataCopyParams, padParams);
    };
};

template <
    class ArchTag,
    class GmType
>
struct CopyUb2Gm {
    static_assert(DEPENDENT_FALSE<ArchTag>, "Unsupported copy ub to gm, can not find the specialization.");
};

template <typename Element>
struct CopyUb2Gm<Arch::AtlasA2, Gemm::GemmType<Element, layout::RowMajor>> {
    using LayoutDst = layout::RowMajor;
    using LayoutSrc = layout::RowMajor;

    static constexpr uint32_t ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(Element);

    __forceinline__ [aicore]
    CopyUb2Gm() = default;

    __forceinline__ [aicore]
    void operator()(
        AscendC::GlobalTensor<Element> const &dstTensor,
        AscendC::LocalTensor<Element> const &srcTensor,
        layout::RowMajor const &layoutDst,
        layout::RowMajor const &layoutSrc)
    {
        AscendC::DataCopyExtParams dataCopyParams(
            layoutDst.shape(0),
            layoutDst.shape(1) * sizeof(Element),
            (layoutSrc.stride(0) - layoutSrc.shape(1)) / ELE_NUM_PER_C0,
            (layoutDst.stride(0) - layoutDst.shape(1)) * sizeof(Element),
            0
        );
        AscendC::DataCopyPad(dstTensor, srcTensor, dataCopyParams);
    }
};

template <
    /// Tag indicating architecture
    class ArchTag,
    class... Args
>
struct TileCopy {
    static_assert(DEPENDENT_FALSE<ArchTag>, "Unsupported tile copy, can not find the specialization.");
};

template <
    class ArchTag,
    /// GemmType for C matrix operand
    class CType,
    /// GemmType for X matrix operand
    class XType,
    /// GemmType for D matrix operand
    class DType
>
struct TileCopy<ArchTag, CType, XType, DType> {
    using ElementC = typename CType::Element;
    using ElementX = typename XType::Element;
    using ElementD = typename DType::Element;

    using CopyGmToUbC = CopyGm2Ub<ArchTag, CType>;
    using CopyGmToUbX = CopyGm2Ub<ArchTag, XType>;
    using CopyUbToGmD = CopyUb2Gm<ArchTag, DType>;
};
} // namespace Catlass::Epilogue::Tile

namespace Catlass::Gemm::helper {

template <class ElementA, class ElementB> struct ElementAccumulatorSelector {
  static_assert(DEPENDENT_FALSE<ElementA>,
                "Unsupported element accumulator selector, can not find the "
                "specialization.");
};

template<>
struct ElementAccumulatorSelector<half, half> {
    using ElementAccumulator = float;
};

template <class GmAType, class GmBType> struct L1AndL0TypeSelectorGemm {
  static_assert(DEPENDENT_FALSE<GmAType>,
                "Unsupported layout selector, can not find the specialization.");
  static_assert(DEPENDENT_FALSE<GmBType>,
                "Unsupported layout selector, can not find the specialization.");
};

template<class Element>
struct L1AndL0TypeSelectorGemm<Gemm::GemmType<Element, layout::RowMajor>, Gemm::GemmType<Element, layout::RowMajor>>{
    using L1AType = Gemm::GemmType<Element, layout::zN, AscendC::TPosition::A1>;
    using L1BType = Gemm::GemmType<Element, layout::zZ, AscendC::TPosition::B1>;
    using L0AType = Gemm::GemmType<Element, layout::zZ, AscendC::TPosition::A2>;
    using L0BType = Gemm::GemmType<Element, layout::nZ, AscendC::TPosition::B2>;
};

template<class Element>
struct L1AndL0TypeSelectorGemm<Gemm::GemmType<Element, layout::RowMajor>, Gemm::GemmType<Element, layout::ColumnMajor>>{
    using L1AType = Gemm::GemmType<Element, layout::zN, AscendC::TPosition::A1>;
    using L1BType = Gemm::GemmType<Element, layout::nZ, AscendC::TPosition::B1>;
    using L0AType = Gemm::GemmType<Element, layout::zZ, AscendC::TPosition::A2>;
    using L0BType = Gemm::GemmType<Element, layout::nZ, AscendC::TPosition::B2>;
};
} // namespace Catlass::Gemm::helper

namespace Catlass::Gemm::Tile {
template <
    class ArchTag,
    class L1Type,
    class L0Type = void
>
struct CopyL1ToL0A {
    static_assert(DEPENDENT_FALSE<ArchTag>, "Unsupported copy l1 to l0, can not find the specialization.");
};

////////////////////////////////
/// new add gemm
template<class ArchTag, class Element>
struct CopyL1ToL0A<ArchTag, Catlass::Gemm::GemmType<Element, layout::zN, AscendC::TPosition::A1>, Catlass::Gemm::GemmType<Element, layout::zZ, AscendC::TPosition::A2>>{
    using LayoutDst = layout::zZ;
    using LayoutSrc = layout::zN;

    static constexpr uint32_t ELE_NUM_PER_C0 =  BYTE_PER_C0 / sizeof(Element);
    static constexpr uint32_t ELE_NUM_PER_FRACTAL = BYTE_PER_FRACTAL / sizeof(Element);

    __forceinline__ [aicore]
    CopyL1ToL0A(){}

    __forceinline__ [aicore]
    void operator()(
        AscendC::LocalTensor<Element> dstTensor,
        AscendC::LocalTensor<Element> srcTensor,
        LayoutDst layoutDst, LayoutSrc layoutSrc
    ){
        AscendC::LoadData2DParams loadDataParams;
        loadDataParams.startIndex = 0;
        loadDataParams.repeatTimes = static_cast<uint16_t>(layoutDst.shape(3));
        loadDataParams.srcStride = layoutSrc.stride(3) / ELE_NUM_PER_FRACTAL;
        loadDataParams.sid = 0;
        loadDataParams.dstGap = layoutDst.stride(3) / ELE_NUM_PER_FRACTAL - 1;
        loadDataParams.ifTranspose = false;
        loadDataParams.addrMode = 0;

        for (uint32_t i = 0; i < layoutDst.shape(1); i++) {
            AscendC::LoadData(dstTensor[i * layoutDst.stride(1)], srcTensor[i * layoutSrc.stride(1)], loadDataParams);
        }
    }
};

/// Partial specialization for zN in and zZ out.
template <class ArchTag, class Element>
struct CopyL1ToL0A<ArchTag, Gemm::GemmType<Element, layout::zN, AscendC::TPosition::A1>> {
    using LayoutDst = layout::zZ;
    using LayoutSrc = layout::zN;

    static constexpr uint32_t ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(Element);
    static constexpr uint32_t ELE_NUM_PER_FRACTAL = BYTE_PER_FRACTAL / sizeof(Element);

    // Methods

    __forceinline__ [aicore]
    CopyL1ToL0A() {};

    __forceinline__ [aicore]
    void operator()(
        AscendC::LocalTensor<Element> const &dstTensor,
        AscendC::LocalTensor<Element> const &srcTensor,
        LayoutDst const &layoutDst, LayoutSrc const &layoutSrc)
    {
        AscendC::LoadData2DParams loadDataParams;

        loadDataParams.startIndex = 0;
        loadDataParams.repeatTimes = static_cast<uint16_t>(layoutDst.shape(3));
        loadDataParams.srcStride = layoutSrc.stride(3) / ELE_NUM_PER_FRACTAL;
        loadDataParams.sid = 0;
        loadDataParams.dstGap = layoutDst.stride(3) / ELE_NUM_PER_FRACTAL - 1;
        loadDataParams.ifTranspose = false;
        loadDataParams.addrMode = 0;

        for (uint32_t i = 0; i < layoutDst.shape(1); i++) {
            AscendC::LoadData(dstTensor[i * layoutDst.stride(1)], srcTensor[i * layoutSrc.stride(1)], loadDataParams);
        }
    }
};



template <
    class ArchTag,
    class L1Type,
    class L0Type = void
>
struct CopyL1ToL0B {
    static_assert(DEPENDENT_FALSE<ArchTag>, "Unsupported copy l1 to l0, can not find the specialization.");
};

////////////////////////////////////////
/// new add gemm
template<class ArchTag, class Element>
struct CopyL1ToL0B<ArchTag, Catlass::Gemm::GemmType<Element, layout::zZ, AscendC::TPosition::B1>, Catlass::Gemm::GemmType<Element, layout::nZ, AscendC::TPosition::B2>>{
    using LayoutDst = layout::nZ;
    using LayoutSrc = layout::zZ;

    static constexpr uint32_t ELE_NUM_PER_C0 =  BYTE_PER_C0 / sizeof(Element);

    __forceinline__ [aicore]
    CopyL1ToL0B(){}

    __forceinline__ [aicore]
    void operator()(
        AscendC::LocalTensor<Element> dstTensor,
        AscendC::LocalTensor<Element> srcTensor,
        LayoutDst layoutDst, LayoutSrc layoutSrc
    ){
        AscendC::LoadData2DParams loadDataParams;
        loadDataParams.startIndex = 0;
        loadDataParams.repeatTimes = static_cast<uint16_t>(CeilDiv<ELE_NUM_PER_C0>(layoutSrc.orgShape(1)));
        loadDataParams.srcStride = 1;
        loadDataParams.sid = 0;
        loadDataParams.dstGap = 0;
        loadDataParams.ifTranspose = true;
        loadDataParams.addrMode = 0;
        for(uint32_t i = 0; i < CeilDiv<C0_NUM_PER_FRACTAL>(layoutDst.orgShape(0)); i++){  // K N
            AscendC::LoadData(dstTensor[i * layoutDst.stride(1)], srcTensor[i * layoutSrc.stride(1)], loadDataParams);
        }
    }
};

template <class ArchTag, class Element>
struct CopyL1ToL0B<ArchTag, Catlass::Gemm::GemmType<Element, layout::nZ, AscendC::TPosition::B1>, Catlass::Gemm::GemmType<Element, layout::nZ, AscendC::TPosition::B2>> {
    using LayoutDst = layout::nZ;
    using LayoutSrc = layout::nZ;

    static constexpr uint32_t ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(Element);
    static constexpr uint32_t ELE_NUM_PER_FRACTAL = BYTE_PER_FRACTAL / sizeof(Element);

    // Methods

    __forceinline__ [aicore]
    CopyL1ToL0B() {};

    __forceinline__ [aicore]
    void operator()(
        AscendC::LocalTensor<Element> const &dstTensor,
        AscendC::LocalTensor<Element> const &srcTensor,
        LayoutDst const &layoutDst, LayoutSrc const &layoutSrc)
    {
        AscendC::LoadData2DParams loadDataParams;

        loadDataParams.startIndex = 0;
        loadDataParams.repeatTimes = static_cast<uint16_t>(layoutDst.shape(3));
        loadDataParams.srcStride = layoutSrc.stride(3) / ELE_NUM_PER_FRACTAL;
        loadDataParams.sid = 0;
        loadDataParams.dstGap = layoutDst.stride(3) / ELE_NUM_PER_FRACTAL - 1;
        loadDataParams.ifTranspose = false;
        loadDataParams.addrMode = 0;

        for (uint32_t i = 0; i < layoutDst.shape(1); i++) {
            AscendC::LoadData(dstTensor[i * layoutDst.stride(1)], srcTensor[i * layoutSrc.stride(1)], loadDataParams);
        }
    }
};

/// Partial specialization for nZ in and nZ out. (Transpose B)
template <class ArchTag, class Element>
struct CopyL1ToL0B<ArchTag, Gemm::GemmType<Element, layout::nZ, AscendC::TPosition::A1>> {
    using LayoutDst = layout::nZ;
    using LayoutSrc = layout::nZ;

    static constexpr uint32_t ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(Element);
    static constexpr uint32_t ELE_NUM_PER_FRACTAL = BYTE_PER_FRACTAL / sizeof(Element);

    // Methods

    __forceinline__ [aicore]
    CopyL1ToL0B() {};

    __forceinline__ [aicore]
    void operator()(
        AscendC::LocalTensor<Element> const &dstTensor,
        AscendC::LocalTensor<Element> const &srcTensor,
        LayoutDst const &layoutDst, LayoutSrc const &layoutSrc)
    {
        AscendC::LoadData2DParams loadDataParams;
        if (layoutSrc.shape(3) == layoutDst.shape(3)) {
            loadDataParams.startIndex = 0;
            loadDataParams.repeatTimes = static_cast<uint16_t>(layoutDst.shape(1) * layoutDst.shape(3));
            loadDataParams.srcStride = 1;
            loadDataParams.sid = 0;
            loadDataParams.dstGap = 0;
            loadDataParams.ifTranspose = false;
            loadDataParams.addrMode = 0;

            AscendC::LoadData(dstTensor, srcTensor, loadDataParams);
        } else {
            loadDataParams.startIndex = 0;
            loadDataParams.repeatTimes = static_cast<uint16_t>(layoutDst.shape(3));
            loadDataParams.srcStride = layoutSrc.stride(3) / ELE_NUM_PER_FRACTAL;
            loadDataParams.sid = 0;
            loadDataParams.dstGap = layoutDst.stride(3) / ELE_NUM_PER_FRACTAL - 1;
            loadDataParams.ifTranspose = false;
            loadDataParams.addrMode = 0;

            for (uint32_t i = 0; i < layoutDst.shape(1); i++) {
                AscendC::LoadData(dstTensor[i * layoutDst.stride(1)], srcTensor[i * layoutSrc.stride(1)], loadDataParams);
            }
        }

    }
};

template <
    class ArchTag,
    /// GemmType for matrix operand
    class GmType,
    class L1Type = void
>
struct CopyGmToL1 {
    static_assert(DEPENDENT_FALSE<ArchTag>, "Unsupported copy gm to l1, can not find the specialization.");
};

template <class ArchTag, class Element>
struct CopyGmToL1<ArchTag, Gemm::GemmType<Element, layout::RowMajor>, Gemm::GemmType<Element, layout::zN, AscendC::TPosition::A1>> {
    using LayoutDst = layout::zN;
    using LayoutSrc = layout::RowMajor;

    static constexpr uint32_t ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(Element);

    // Mehtods

    __forceinline__ [aicore]
    CopyGmToL1() {};

    __forceinline__ [aicore]
    void operator()(
        AscendC::LocalTensor<Element> const &dstTensor,
        AscendC::GlobalTensor<Element> const &srcTensor,
        LayoutDst const &layoutDst, LayoutSrc const &layoutSrc)
    {
        AscendC::Nd2NzParams intriParams;

        intriParams.ndNum = 1;
        intriParams.dValue = layoutSrc.shape(1);
        intriParams.srcNdMatrixStride = 0;
        intriParams.dstNzC0Stride = layoutDst.stride(3) / ELE_NUM_PER_C0;
        intriParams.dstNzMatrixStride = 0;

        if (layoutSrc.stride(0) < STRIDE_LIMIT) {
            intriParams.nValue = layoutSrc.shape(0);
            intriParams.srcDValue = layoutSrc.stride(0);
            intriParams.dstNzNStride = layoutDst.stride(0) / ELE_NUM_PER_C0;
            AscendC::DataCopy(dstTensor, srcTensor, intriParams);
        } else {
            intriParams.nValue = 1;
            intriParams.srcDValue = 0;
            intriParams.dstNzNStride = 0;
            for (uint32_t i = 0; i < layoutSrc.shape(0); i++) {
                AscendC::DataCopy(dstTensor[i * ELE_NUM_PER_C0], srcTensor[i * layoutSrc.stride(0)], intriParams);
            }
        }
    }
};

template <class ArchTag, class Element>
struct CopyGmToL1<ArchTag, Gemm::GemmType<Element, layout::RowMajor>, Gemm::GemmType<Element, layout::zZ, AscendC::TPosition::B1>> {
    using LayoutDst = layout::zZ;
    using LayoutSrc = layout::RowMajor;

    static constexpr uint32_t ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(Element);

    // Mehtods

    __forceinline__ [aicore]
    CopyGmToL1() {};

    __forceinline__ [aicore]
    void operator()(
        AscendC::LocalTensor<Element> const &dstTensor,
        AscendC::GlobalTensor<Element> const &srcTensor,
        LayoutDst const &layoutDst, LayoutSrc const &layoutSrc)
    {
        AscendC::Nd2NzParams intriParams;
        uint32_t srcNdStride = C0_NUM_PER_FRACTAL * layoutSrc.stride(0);
        uint32_t ndNum = layoutSrc.shape(0) / C0_NUM_PER_FRACTAL;
        uint32_t remains = layoutSrc.shape(0) % C0_NUM_PER_FRACTAL;
        if (srcNdStride < STRIDE_LIMIT) {
            if (ndNum) {
                intriParams.ndNum = ndNum;
                intriParams.nValue = C0_NUM_PER_FRACTAL;
                intriParams.dValue = layoutSrc.shape(1);
                intriParams.srcNdMatrixStride = srcNdStride;
                intriParams.srcDValue = layoutSrc.stride(0);

                intriParams.dstNzC0Stride = layoutDst.stride(3) / ELE_NUM_PER_C0;
                intriParams.dstNzNStride = layoutDst.stride(0) / ELE_NUM_PER_C0;

                intriParams.dstNzMatrixStride = layoutDst.stride(1);

                AscendC::DataCopy(dstTensor, srcTensor, intriParams);
            }

            if (remains) {
                AscendC::Nd2NzParams tailParams;
                tailParams.ndNum = 1;
                tailParams.nValue = remains;
                tailParams.dValue = layoutSrc.shape(1);
                tailParams.srcNdMatrixStride = srcNdStride;
                tailParams.srcDValue = layoutSrc.stride(0);

                tailParams.dstNzC0Stride = layoutDst.stride(3) / ELE_NUM_PER_C0;
                tailParams.dstNzNStride = layoutDst.stride(0) / ELE_NUM_PER_C0;
                tailParams.dstNzMatrixStride = 0;  //`

                AscendC::DataCopy(dstTensor[ndNum * layoutDst.stride(1)], srcTensor[ndNum * srcNdStride], tailParams);
            }
        } else if (layoutSrc.stride(0) < STRIDE_LIMIT) {
            for (uint32_t i = 0; i < ndNum; i++) {
                AscendC::Nd2NzParams intriParams;
                intriParams.ndNum = 1;
                intriParams.nValue = C0_NUM_PER_FRACTAL;
                intriParams.dValue = layoutSrc.shape(1);
                intriParams.srcNdMatrixStride = 0;
                intriParams.srcDValue = layoutSrc.stride(0);

                intriParams.dstNzC0Stride = layoutDst.stride(3) / ELE_NUM_PER_C0;
                intriParams.dstNzNStride = layoutDst.stride(0) / ELE_NUM_PER_C0;
                intriParams.dstNzMatrixStride = 0;

                AscendC::DataCopy(dstTensor[i * layoutDst.stride(1)], srcTensor[i * srcNdStride], intriParams);
            }
            if (remains) {
                AscendC::Nd2NzParams tailParams;
                tailParams.ndNum = 1;
                tailParams.nValue = remains;
                tailParams.dValue = layoutSrc.shape(1);
                tailParams.srcNdMatrixStride = 0;
                tailParams.srcDValue = layoutSrc.stride(0);

                tailParams.dstNzC0Stride = layoutDst.stride(3) / ELE_NUM_PER_C0;
                tailParams.dstNzNStride = layoutDst.stride(0) / ELE_NUM_PER_C0;
                tailParams.dstNzMatrixStride = 0;

                AscendC::DataCopy(dstTensor[ndNum * layoutDst.stride(1)], srcTensor[ndNum * srcNdStride], tailParams);
            }
        } else {
            for (uint32_t i = 0; i < layoutSrc.shape(0); i++) {
                uint32_t idxR0 = i / C0_NUM_PER_FRACTAL;
                uint32_t idxInR0 = i % C0_NUM_PER_FRACTAL;

                AscendC::Nd2NzParams intriParams;
                intriParams.ndNum = 1;
                intriParams.nValue = 1;
                intriParams.dValue = layoutSrc.shape(1);
                intriParams.srcNdMatrixStride = 0;
                intriParams.srcDValue = 0;

                intriParams.dstNzC0Stride = layoutDst.stride(3) / ELE_NUM_PER_C0;
                intriParams.dstNzNStride = 0;
                intriParams.dstNzMatrixStride = 0;

                uint32_t offsetDst = i * idxR0 * layoutDst.stride(1) + idxInR0 * ELE_NUM_PER_C0;
                uint32_t offsetSrc = i * layoutSrc.stride(0);
                AscendC::DataCopy(dstTensor[offsetDst], srcTensor[offsetSrc], intriParams);
            }
        }
    }
};

template <class ArchTag, class Element>
struct CopyGmToL1<ArchTag, Gemm::GemmType<Element, layout::ColumnMajor>, Gemm::GemmType<Element, layout::nZ, AscendC::TPosition::B1>> {
    using LayoutDst = layout::nZ;
    using LayoutSrc = layout::ColumnMajor;

    static constexpr uint32_t ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(Element);

    // Mehtods

    __forceinline__ [aicore]
    CopyGmToL1() {};

    __forceinline__ [aicore]
    void operator()(
        AscendC::LocalTensor<Element> const &dstTensor,
        AscendC::GlobalTensor<Element> const &srcTensor,
        LayoutDst const &layoutDst, LayoutSrc const &layoutSrc)
    {
        AscendC::Nd2NzParams intriParams;

        intriParams.ndNum = 1;
        intriParams.dValue = layoutSrc.shape(0);
        intriParams.srcNdMatrixStride = 0;
        intriParams.dstNzC0Stride = layoutDst.stride(1) / ELE_NUM_PER_C0;
        intriParams.dstNzMatrixStride = 0;

        if (layoutSrc.stride(1) < STRIDE_LIMIT) {
            intriParams.nValue = layoutSrc.shape(1);
            intriParams.srcDValue = layoutSrc.stride(1);
            intriParams.dstNzNStride = layoutDst.stride(2) / ELE_NUM_PER_C0;
            AscendC::DataCopy(dstTensor, srcTensor, intriParams);
        } else {
            intriParams.nValue = 1;
            intriParams.srcDValue = 0;
            intriParams.dstNzNStride = 0;
            for (uint32_t i = 0; i < layoutSrc.shape(1); i++) {
                AscendC::DataCopy(dstTensor[i * ELE_NUM_PER_C0], srcTensor[i * layoutSrc.stride(1)], intriParams);
            }
        }
    }
};

template <class ArchTag, class Element>
struct CopyGmToL1<ArchTag, Gemm::GemmType<Element, layout::ColumnMajor>, Gemm::GemmType<Element, layout::nZ, AscendC::TPosition::A1>> {
    using LayoutDst = layout::nZ;
    using LayoutSrc = layout::ColumnMajor;

    static constexpr uint32_t ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(Element);

    // Mehtods

    __forceinline__ [aicore]
    CopyGmToL1() {};

    __forceinline__ [aicore]
    void operator()(
        AscendC::LocalTensor<Element> const &dstTensor,
        AscendC::GlobalTensor<Element> const &srcTensor,
        LayoutDst const &layoutDst, LayoutSrc const &layoutSrc)
    {
        AscendC::Nd2NzParams intriParams;

        intriParams.ndNum = 1;
        intriParams.dValue = layoutSrc.shape(0);
        intriParams.srcNdMatrixStride = 0;
        intriParams.dstNzC0Stride = layoutDst.stride(1) / ELE_NUM_PER_C0;
        intriParams.dstNzMatrixStride = 0;

        if (layoutSrc.stride(1) < STRIDE_LIMIT) {
            intriParams.nValue = layoutSrc.shape(1);
            intriParams.srcDValue = layoutSrc.stride(1);
            intriParams.dstNzNStride = layoutDst.stride(2) / ELE_NUM_PER_C0;
            AscendC::DataCopy(dstTensor, srcTensor, intriParams);
        } else {
            intriParams.nValue = 1;
            intriParams.srcDValue = 0;
            intriParams.dstNzNStride = 0;
            for (uint32_t i = 0; i < layoutSrc.shape(1); i++) {
                AscendC::DataCopy(dstTensor[i * ELE_NUM_PER_C0], srcTensor[i * layoutSrc.stride(1)], intriParams);
            }
        }
    }
};
////////////////////////////////////////


template <class ArchTag, class Element>
struct CopyGmToL1<ArchTag, Gemm::GemmType<Element, layout::RowMajor>, Gemm::GemmType<Element, layout::zN, AscendC::TPosition::B1>> {
    using LayoutDst = layout::zN;
    using LayoutSrc = layout::RowMajor;

    static constexpr uint32_t ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(Element);

    // Methods

    __forceinline__ [aicore]
    CopyGmToL1() {};

    __forceinline__ [aicore]
    void operator()(
        AscendC::LocalTensor<Element> const &dstTensor,
        AscendC::GlobalTensor<Element> const &srcTensor,
        LayoutDst const &layoutDst, LayoutSrc const &layoutSrc)
    {
        AscendC::Nd2NzParams intriParams;

        intriParams.ndNum = 1;
        intriParams.dValue = layoutSrc.shape(1);
        intriParams.srcNdMatrixStride = 0;
        intriParams.dstNzC0Stride = layoutDst.stride(3) / ELE_NUM_PER_C0;
        intriParams.dstNzMatrixStride = 0;

        if (layoutSrc.stride(0) < STRIDE_LIMIT) {
            intriParams.nValue = layoutSrc.shape(0);
            intriParams.srcDValue = layoutSrc.stride(0);
            intriParams.dstNzNStride = layoutDst.stride(0) / ELE_NUM_PER_C0;
            AscendC::DataCopy(dstTensor, srcTensor, intriParams);
        } else {
            intriParams.nValue = 1;
            intriParams.srcDValue = 0;
            intriParams.dstNzNStride = 0;
            for (uint32_t i = 0; i < layoutSrc.shape(0); i++) {
                AscendC::DataCopy(dstTensor[i * ELE_NUM_PER_C0], srcTensor[i * layoutSrc.stride(0)], intriParams);
            }
        }
    }
};
/////////////////////////////////

/// Partial specialization for AtlasA2, RowMajor in and zN out.
template <class Element>
struct CopyGmToL1<Arch::AtlasA2, Gemm::GemmType<Element, layout::RowMajor>> {
    using LayoutDst = layout::zN;
    using LayoutSrc = layout::RowMajor;

    static constexpr uint32_t ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(Element);

    // Mehtods

    __forceinline__ [aicore]
    CopyGmToL1() {};

    __forceinline__ [aicore]
    void operator()(
        AscendC::LocalTensor<Element> const &dstTensor,
        AscendC::GlobalTensor<Element> const &srcTensor,
        LayoutDst const &layoutDst, LayoutSrc const &layoutSrc)
    {
        AscendC::Nd2NzParams intriParams;

        intriParams.ndNum = 1;
        intriParams.dValue = layoutSrc.shape(1);
        intriParams.srcNdMatrixStride = 0;
        intriParams.dstNzC0Stride = layoutDst.stride(3) / ELE_NUM_PER_C0;
        intriParams.dstNzMatrixStride = 0;

        if (layoutSrc.stride(0) < STRIDE_LIMIT) {
            intriParams.nValue = layoutSrc.shape(0);
            intriParams.srcDValue = layoutSrc.stride(0);
            intriParams.dstNzNStride = layoutDst.stride(0) / ELE_NUM_PER_C0;
            AscendC::DataCopy(dstTensor, srcTensor, intriParams);
        } else {
            intriParams.nValue = 1;
            intriParams.srcDValue = 0;
            intriParams.dstNzNStride = 0;
            for (uint32_t i = 0; i < layoutSrc.shape(0); i++) {
                AscendC::DataCopy(dstTensor[i * ELE_NUM_PER_C0], srcTensor[i * layoutSrc.stride(0)], intriParams);
            }
        }
    }

    // layoutSrc must be the layout of one of the src matrices
    __forceinline__ [aicore]
    void operator()(
        AscendC::LocalTensor<Element> const &dstTensor,
        AscendC::GlobalTensor<Element> const &srcTensor,
        LayoutDst const &layoutDst, LayoutSrc const &layoutSrc,
        uint32_t ndNum, uint32_t srcNdMatrixStride,
        uint32_t dstNzNStride, uint32_t dstNzMatrixStride,
        uint32_t dstNzC0Stride)
    {
        AscendC::Nd2NzParams intriParams;

        intriParams.nValue = layoutSrc.shape(0);
        intriParams.dValue = layoutSrc.shape(1);
        intriParams.srcDValue = layoutSrc.stride(0);
        intriParams.dstNzNStride = dstNzNStride;
        intriParams.dstNzC0Stride = dstNzC0Stride;
        if (srcNdMatrixStride < STRIDE_LIMIT) {
            intriParams.ndNum = ndNum;
            intriParams.srcNdMatrixStride = srcNdMatrixStride;
            intriParams.dstNzMatrixStride = dstNzMatrixStride;
            AscendC::DataCopy(dstTensor, srcTensor, intriParams);
        } else {
            intriParams.ndNum = 1;
            intriParams.srcNdMatrixStride = 0;
            intriParams.dstNzMatrixStride = 0;
            for (uint32_t i = 0; i < ndNum; i++) {
                AscendC::DataCopy(dstTensor[i * ELE_NUM_PER_C0], srcTensor[i * srcNdMatrixStride], intriParams);
            }
        }
    }
};

template <
    class Element
>
struct CopyGmToL1<Arch::AtlasA2, Gemm::GemmType<Element, layout::ColumnMajor>> {
    using LayoutDst = layout::nZ;
    using LayoutSrc = layout::ColumnMajor;

    static constexpr uint32_t ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(Element);

    // Mehtods

    __forceinline__ [aicore]
    CopyGmToL1() {};

    __forceinline__ [aicore]
    void operator()(
        AscendC::LocalTensor<Element> const &dstTensor,
        AscendC::GlobalTensor<Element> const &srcTensor,
        LayoutDst const &layoutDst, LayoutSrc const &layoutSrc)
    {
        AscendC::Nd2NzParams intriParams;

        intriParams.ndNum = 1;
        intriParams.dValue = layoutSrc.shape(0);
        intriParams.srcNdMatrixStride = 0;
        intriParams.dstNzC0Stride = layoutDst.stride(1) / ELE_NUM_PER_C0;
        intriParams.dstNzMatrixStride = 0;

        if (layoutSrc.stride(1) < STRIDE_LIMIT) {
            intriParams.nValue = layoutSrc.shape(1);
            intriParams.srcDValue = layoutSrc.stride(1);
            intriParams.dstNzNStride = layoutDst.stride(2) / ELE_NUM_PER_C0;
            AscendC::DataCopy(dstTensor, srcTensor, intriParams);
        } else {
            intriParams.nValue = 1;
            intriParams.srcDValue = 0;
            intriParams.dstNzNStride = 0;
            for (uint32_t i = 0; i < layoutSrc.shape(1); i++) {
                AscendC::DataCopy(dstTensor[i * ELE_NUM_PER_C0], srcTensor[i * layoutSrc.stride(1)], intriParams);
            }
        }
    }
};


enum class ScaleGranularity {
    UNDEFINED = -1,
    NO_QUANT = 0,
    PER_TENSOR,
    PER_CHANNEL,
    PER_GROUP
};
template <
    class ArchTag,
    class ElementSrc,
    class ElementDst,
    ScaleGranularity DEQUANT_GRANULARITY = ScaleGranularity::NO_QUANT
>
struct CopyL0CToGmQuantMode {
    static_assert(DEPENDENT_FALSE<ArchTag>, "Unsupported copy l0c to gm, can not find the specialization.");
};

// CopyL0CToGm cast fp32 to fp16
template <>
struct CopyL0CToGmQuantMode<
    Catlass::Arch::AtlasA2,
    float, half,
    ScaleGranularity::NO_QUANT
> {
    static constexpr auto VALUE = QuantMode_t::F322F16;
};

// CopyL0CToGm cast fp32 to bf16
template <>
struct CopyL0CToGmQuantMode<
    Catlass::Arch::AtlasA2,
    float, bfloat16_t,
    ScaleGranularity::NO_QUANT
> {
    static constexpr auto VALUE = QuantMode_t::F322BF16;
};

// CopyL0CToGm output fp32
template <>
struct CopyL0CToGmQuantMode<
    Catlass::Arch::AtlasA2,
    float, float,
    ScaleGranularity::NO_QUANT
> {
    static constexpr auto VALUE = QuantMode_t::NoQuant;
};

// CopyL0CToGm output int32
template <>
struct CopyL0CToGmQuantMode<
    Catlass::Arch::AtlasA2,
    int32_t, int32_t,
    ScaleGranularity::NO_QUANT
> {
    static constexpr auto VALUE = QuantMode_t::NoQuant;
};

template <
    class ArchTag,
    class ElementAccumulator,
    class GmType,
    ScaleGranularity DEQUANT_GRANULARITY = ScaleGranularity::NO_QUANT,
    bool ReluEnable = false
>
struct CopyL0CToGm {
    static_assert(DEPENDENT_FALSE<ArchTag>, "Unsupported copy l0c to gm, can not find the specialization.");
};

template <
    class ElementAccumulator_,
    class ElementDst_,
    bool ReluEnable_
>
struct CopyL0CToGm<Catlass::Arch::AtlasA2,
                   ElementAccumulator_,
                   Gemm::GemmType<ElementDst_, layout::RowMajor>,
                   ScaleGranularity::NO_QUANT,
                   ReluEnable_>
{
    using ArchTag = Catlass::Arch::AtlasA2;
    using ElementDst = ElementDst_;
    using ElementSrc = ElementAccumulator_;
    using LayoutSrc = Catlass::layout::zN;
    using LayoutDst = Catlass::layout::RowMajor;
    static constexpr auto quantPre = CopyL0CToGmQuantMode<ArchTag, ElementSrc, ElementDst,
        ScaleGranularity::NO_QUANT>::VALUE;
    static constexpr auto reluEn = ReluEnable_;

    __forceinline__ [aicore]
    void operator()(AscendC::GlobalTensor<ElementDst> const &dst, AscendC::LocalTensor<ElementSrc> const &src,
        LayoutDst const &dstLayout, LayoutSrc const &srcLayout, uint8_t unitFlag = 0)
    {
        AscendC::FixpipeParamsV220 intriParams;

        // Fixpipe layout information
        intriParams.nSize = dstLayout.shape(1);
        intriParams.mSize = dstLayout.shape(0);
        intriParams.srcStride = srcLayout.stride(3) / srcLayout.stride(0);
        intriParams.dstStride = dstLayout.stride(0);

        // Fixpipe auxiliary arguments
        intriParams.quantPre = quantPre;
        intriParams.reluEn = reluEn;
        intriParams.unitFlag = unitFlag;

        // Call AscendC Fixpipe
        AscendC::Fixpipe<ElementDst, ElementSrc, AscendC::CFG_ROW_MAJOR>(dst, src, intriParams);
    }
};

template <
    /// Tag indicating architecture
    class ArchTag,
    /// GemmType for A matrix operand
    class AType,
    /// GemmType type for B matrix operand
    class BType,
    /// GemmType type for C matrix operand
    class CType,
    /// GemmTpe type for Bias operand
    class BiasType = void
>
struct TileCopyGemm {
    using ElementA = typename AType::Element;
    using ElementB = typename BType::Element;
    using ElementAccumulator =
        typename Gemm::helper::ElementAccumulatorSelector<ElementA, ElementB>::ElementAccumulator;
    // change structual
    using L1AType = typename Gemm::helper::L1AndL0TypeSelectorGemm<AType, BType>::L1AType;
    using L1BType = typename Gemm::helper::L1AndL0TypeSelectorGemm<AType, BType>::L1BType;
    using L0AType = typename Gemm::helper::L1AndL0TypeSelectorGemm<AType, BType>::L0AType;
    using L0BType = typename Gemm::helper::L1AndL0TypeSelectorGemm<AType, BType>::L0BType;

    using CopyGmToL1A = Gemm::Tile::CopyGmToL1<ArchTag, AType, L1AType>;
    using CopyGmToL1B = Gemm::Tile::CopyGmToL1<ArchTag, BType, L1BType>;
    using CopyL1ToL0A = Gemm::Tile::CopyL1ToL0A<ArchTag, L1AType, L0AType>;
    using CopyL1ToL0B = Gemm::Tile::CopyL1ToL0B<ArchTag, L1BType, L0BType>;
    using CopyL0CToGm = Gemm::Tile::CopyL0CToGm<ArchTag, ElementAccumulator, CType>;
};

template <
    /// Tag indicating architecture
    class ArchTag_,
    /// GemmType for A matrix operand
    class AType_,
    /// GemmType type for B matrix operand
    class BType_,
    /// GemmType type for Bias operand
    class BiasType_
>
struct TileMmad {
    using ElementA = typename AType_::Element;
    using ElementB = typename BType_::Element;
    using ElementAccumulator =
        typename Gemm::helper::ElementAccumulatorSelector<ElementA, ElementB>::ElementAccumulator;

    // Methods

    __forceinline__ [aicore]
    TileMmad() {}

    __forceinline__ [aicore]
    void operator()(AscendC::LocalTensor<ElementAccumulator> const &l0CTensor,
         AscendC::LocalTensor<ElementA> const &l0ATensor,
         AscendC::LocalTensor<ElementB> const &l0BTensor,
         uint32_t m, uint32_t n, uint32_t k,
         bool initC = true, uint8_t unitFlag = 0)
    {
        AscendC::MmadParams mmadParams;
        mmadParams.m = m;
        mmadParams.n = n;
        mmadParams.k = k;
        mmadParams.unitFlag = unitFlag;
        mmadParams.cmatrixInitVal = initC;
        if constexpr (std::is_same_v<ElementA, float> && std::is_same_v<typename AType_::Layout, layout::ColumnMajor>) {
            mmadParams.kDirectionAlign = true;
        }

        AscendC::Mmad(l0CTensor,
                      l0ATensor,
                      l0BTensor,
                      mmadParams);

        const uint32_t PIPE_M_BARRIER_THRESHOLD = 10;
        if ((m / C0_NUM_PER_FRACTAL) * (n / C0_NUM_PER_FRACTAL) < PIPE_M_BARRIER_THRESHOLD) {
            AscendC::PipeBarrier<PIPE_M>();
        }
    }

    __forceinline__ [aicore]
    void operator()(AscendC::LocalTensor<ElementAccumulator> const &l0CTensor,
         AscendC::LocalTensor<ElementA> const &l0ATensor,
         AscendC::LocalTensor<ElementB> const &l0BTensor,
         AscendC::LocalTensor<ElementAccumulator> const &l0BiasTensor,
         uint32_t m, uint32_t n, uint32_t k,
         bool initC = true, uint8_t unitFlag = 0)
    {
        AscendC::MmadParams mmadParams;
        mmadParams.m = m;
        mmadParams.n = n;
        mmadParams.k = k;
        mmadParams.unitFlag = unitFlag;
        mmadParams.cmatrixInitVal = false;
        if constexpr (std::is_same_v<ElementA, float> && std::is_same_v<typename AType_::Layout, layout::ColumnMajor>) {
            mmadParams.kDirectionAlign = true;
        }

        AscendC::Mmad(l0CTensor,
                      l0ATensor,
                      l0BTensor,
                      l0BiasTensor,
                      mmadParams);

        const uint32_t PIPE_M_BARRIER_THRESHOLD = 10;
        if ((m / C0_NUM_PER_FRACTAL) * (n / C0_NUM_PER_FRACTAL) < PIPE_M_BARRIER_THRESHOLD) {
            AscendC::PipeBarrier<PIPE_M>();
        }
    }
};
} // namespace Catlass::Gemm::Tile

namespace Catlass::Epilogue::Block {
template <class DispatchPolicy, class... Args> class BlockEpilogue {
  static_assert(DEPENDENT_FALSE<DispatchPolicy>,
                "Could not find an epilogue specialization");
};

template<
    class CType_,
    class XType_,
    class DType_,
    class TileElemWiseEpilogueAdd_,
    class TileElemWiseEpilogueMuls_,
    class TileElemWiseCastD_,
    class TileCopy_
>
class BlockEpilogue<
    EpilogueAtlasA2Gemm,
    CType_,
    XType_,
    DType_,
    TileElemWiseEpilogueAdd_,
    TileElemWiseEpilogueMuls_,
    TileElemWiseCastD_,
    TileCopy_
>{
public:
    // Type aliases
    using DispatchPolicy = EpilogueAtlasA2Gemm;
    using ArchTag = typename DispatchPolicy::ArchTag;
    using ElementC = typename CType_::Element;
    using LayoutC = typename CType_::Layout;
    using ElementX = typename XType_::Element;
    using LayoutX = typename XType_::Layout;
    using ElementD = typename DType_::Element;
    using LayoutD = typename DType_::Layout;
    using TileElemWiseEpilogueAdd = TileElemWiseEpilogueAdd_;
    using TileElemWiseEpilogueMuls = TileElemWiseEpilogueMuls_;
    using TileElemWiseCastD = TileElemWiseCastD_;
    using CopyGmToUbC = typename TileCopy_::CopyGmToUbC;
    using CopyGmToUbX = typename TileCopy_::CopyGmToUbX;
    using CopyUbToGmD = typename TileCopy_::CopyUbToGmD;

    const uint32_t SubNum = AscendC::GetSubBlockNum();
    const uint32_t UBSize = ArchTag::UB_SIZE;
    static constexpr bool isNeedCast = !std::is_same<ElementC, ElementX>::value;
    static constexpr uint32_t COMPUTE_LENGTH = TileElemWiseEpilogueAdd::COMPUTE_LENGTH;

    using ElementCompute = typename Catlass::Gemm::helper::ElementAccumulatorSelector<ElementX, ElementD>::ElementAccumulator;
    using ElementScalar = ElementCompute;

    // Check if ArchTag is matched
    static_assert(std::is_same_v<typename TileElemWiseEpilogueAdd::ArchTag, ArchTag>, "Tile epilogue's ArchTag mismatch");
    static_assert(std::is_same_v<typename TileElemWiseEpilogueMuls::ArchTag, ArchTag>, "Tile epilogue's ArchTag mismatch");
    static_assert(std::is_same_v<LayoutC, layout::RowMajor>, "LayoutC only support RowMajor yet!");

    struct Params{
        ElementScalar alpha;
        ElementScalar beta;
        GM_ADDR ptrX;
        LayoutX layoutX;
        GM_ADDR ptrD;
        LayoutD layoutD;

        __forceinline__ [aicore]
        Params(){}

        __forceinline__ [aicore]
        Params(ElementScalar alpha_, ElementScalar beta_, GM_ADDR ptrX_, LayoutX layoutX_, GM_ADDR ptrD_, LayoutD layoutD_)
            : alpha(alpha_), beta(beta_), ptrX(ptrX_), layoutX(layoutX_), ptrD(ptrD_), layoutD(layoutD_){}
    };

    __forceinline__ [aicore]
    BlockEpilogue(Arch::Resource<ArchTag> &resource, GemmCoord blockShape_, Params const& params_ ,uint32_t ubByteStart = 0) : blockShapeMNK(blockShape_), params(params_)
    {
        uint32_t maxMPerBlock = blockShapeMNK.m();
        uint32_t maxNPerBlock = blockShapeMNK.n();
        uint32_t tileSize = maxMPerBlock * maxNPerBlock / SubNum;
        uint32_t ubCSize = tileSize * sizeof(ElementC);
        uint32_t ubXSize = tileSize * sizeof(ElementX);
        uint32_t ubDSize = tileSize * sizeof(ElementD);
        uint32_t ubXCastSize = tileSize * sizeof(ElementCompute);
        uint32_t ubDCastSize = tileSize * sizeof(ElementCompute);
        ubCTensor = resource.ubBuf.template GetBufferByByte<ElementC>(ubByteStart);
        ubByteStart += ubCSize;
        ubXTensor = resource.ubBuf.template GetBufferByByte<ElementX>(ubByteStart);
        ubByteStart += ubXSize;
        ubDTensor = resource.ubBuf.template GetBufferByByte<ElementD>(ubByteStart);
        ubByteStart += ubDSize;
        if constexpr (isNeedCast)
        {
            ubXTensorCast = resource.ubBuf.template GetBufferByByte<ElementCompute>(ubByteStart);
            ubByteStart += ubXCastSize;
            ubDTensorCast = resource.ubBuf.template GetBufferByByte<ElementCompute>(ubByteStart);;
            ubByteStart += ubDCastSize;
        }
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
    }

    __forceinline__ [aicore]
    ~BlockEpilogue(){
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
    }

    __forceinline__ [aicore]
    void operator()(
        GemmCoord const &actualShapeMNK, GemmCoord const &blockCoordMNK,
        AscendC::GlobalTensor<ElementC> const &gmBlockC, LayoutC const &layoutC,
        uint64_t const &offset)
    {
        AscendC::GlobalTensor<ElementX> gmBlockX;
        gmBlockX.SetGlobalBuffer(reinterpret_cast<__gm__ ElementX*>(params.ptrX));
        AscendC::GlobalTensor<ElementD> gmBlockD;
        gmBlockD.SetGlobalBuffer(reinterpret_cast<__gm__ ElementD*>(params.ptrD));
        MatrixCoord blockShapeMN = blockShapeMNK.GetCoordMN();
        MatrixCoord blockCoordMN = blockCoordMNK.GetCoordMN();
        MatrixCoord actualShapeMN = actualShapeMNK.GetCoordMN();
        MatrixCoord blockOffset = blockCoordMN * blockShapeMN;
        MatrixCoord subblockShape{
            CeilDiv(actualShapeMN.row(), SubNum),
            actualShapeMN.column()
        };
        MatrixCoord subblockCoord{AscendC::GetSubBlockIdx(), 0};
        MatrixCoord actualSubblockShape = MatrixCoord::Min(
            subblockShape, actualShapeMN - subblockCoord * subblockShape);
        MatrixCoord subblockOffset = subblockCoord * subblockShape;
        LayoutC layoutInUb{blockShapeMN.row() / SubNum, blockShapeMN.column()};
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
        auto layoutTileX = params.layoutX.GetTileLayout(actualSubblockShape);
        auto layoutXInUb = layoutInUb.GetTileLayout(actualSubblockShape);
        auto gmTileX = gmBlockX[offset + params.layoutX.GetOffset(blockOffset + subblockOffset)];
        copyGmToUbX(ubXTensor, gmTileX, layoutXInUb, layoutTileX);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID1);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID1);
        if constexpr (isNeedCast)
        {
            AscendC::Cast<ElementCompute, ElementX>(ubXTensorCast, ubXTensor, AscendC::RoundMode::CAST_NONE, COMPUTE_LENGTH);
            AscendC::PipeBarrier<PIPE_V>();
            tileElemWiseEpilogueMuls(ubXTensorCast, ubXTensorCast, (ElementCompute)params.beta);
        } else {
            tileElemWiseEpilogueMuls(ubXTensor, ubXTensor, (ElementX)params.beta);
        }
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
        auto layoutTileC = layoutC.GetTileLayout(actualSubblockShape);
        auto layoutCInUb = layoutInUb.GetTileLayout(actualSubblockShape);
        auto gmTileC = gmBlockC[offset + layoutC.GetOffset(blockOffset + subblockOffset)];
        copyGmToUbC(ubCTensor, gmTileC, layoutCInUb, layoutTileC);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
        tileElemWiseEpilogueMuls(ubCTensor, ubCTensor, (ElementC)params.alpha);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
        AscendC::PipeBarrier<PIPE_V>();
        if constexpr (isNeedCast)
        {
            tileElemWiseEpilogueAdd(ubDTensorCast, ubCTensor, ubXTensorCast);
        } else {
            tileElemWiseEpilogueAdd(ubDTensor, ubCTensor, ubXTensor);
        }
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
        AscendC::PipeBarrier<PIPE_V>();
        if constexpr (isNeedCast)
        {
            tileElemWiseCastD(ubDTensor, ubDTensorCast);
        }
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
        auto layoutDInGm = params.layoutD.GetTileLayout(actualSubblockShape);
        auto layoutTileD = layoutInUb.GetTileLayout(actualSubblockShape);
        auto gmTileD = gmBlockD[offset + params.layoutD.GetOffset(blockOffset + subblockOffset)];
        copyUbToGmD(gmTileD, ubDTensor, layoutDInGm, layoutTileD);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
    }
private:
    GemmCoord blockShapeMNK;
    Params params;

    AscendC::LocalTensor<ElementC> ubCTensor;
    AscendC::LocalTensor<ElementX> ubXTensor;
    AscendC::LocalTensor<ElementD> ubDTensor;
    AscendC::LocalTensor<ElementCompute> ubXTensorCast;
    AscendC::LocalTensor<ElementCompute> ubDTensorCast;

    CopyGmToUbC copyGmToUbC;
    CopyGmToUbX copyGmToUbX;
    CopyUbToGmD copyUbToGmD;

    TileElemWiseEpilogueAdd tileElemWiseEpilogueAdd;
    TileElemWiseEpilogueMuls tileElemWiseEpilogueMuls;
    TileElemWiseCastD tileElemWiseCastD;
};
} // namespace Catlass::Epilogue::Block

namespace Catlass::Gemm::Block {
template <class DispatchPolicy, class L1TileShape, class L0TileShape, class AType,
          class BType, class CType, class BiasType = void,
          class TileCopy = Gemm::Tile::TileCopyGemm<
              typename DispatchPolicy::ArchTag, AType, BType, CType, BiasType>,
          class TileMmad = Gemm::Tile::TileMmad<typename DispatchPolicy::ArchTag, AType,
                                                BType, BiasType>>
struct BlockGemm {
  static_assert(DEPENDENT_FALSE<DispatchPolicy>,
                "BlockMmad is not implemented for this DispatchPolicy");
};

template<
    bool ENABLE_UNIT_FLAG_,
    bool ENABLE_SHUFFLE_K_,
    bool ENABLE_ABBA_,
    class L1TileShape_,
    class L0TileShape_,
    class AType_,
    class BType_,
    class CType_,
    class BiasType_,
    class TileCopy_,
    class TileMmad_
>
struct BlockGemm<
    Gemm::GemmAtlasA2<ENABLE_UNIT_FLAG_, ENABLE_SHUFFLE_K_, ENABLE_ABBA_>,
    L1TileShape_,
    L0TileShape_,
    AType_,
    BType_,
    CType_,
    BiasType_,
    TileCopy_,
    TileMmad_
>{
public:
    using DispatchPolicy = Gemm::GemmAtlasA2<ENABLE_UNIT_FLAG_, ENABLE_SHUFFLE_K_, ENABLE_ABBA_>;
    using ArchTag = typename DispatchPolicy::ArchTag;
    using L1TileShape = L1TileShape_;
    using L0TileShape = L0TileShape_;
    using ElementA = typename AType_::Element;
    using LayoutA = typename AType_::Layout;
    using ElementB = typename BType_::Element;
    using LayoutB = typename BType_::Layout;
    using ElementC = typename CType_::Element;
    using LayoutC = typename CType_::Layout;

    using TileMmad = TileMmad_;
    using CopyGmToL1A = typename TileCopy_::CopyGmToL1A;
    using CopyGmToL1B = typename TileCopy_::CopyGmToL1B;
    using CopyL1ToL0A = typename TileCopy_::CopyL1ToL0A;
    using CopyL1ToL0B = typename TileCopy_::CopyL1ToL0B;
    using CopyL0CToGm = typename TileCopy_::CopyL0CToGm;
    using ElementAccumulator =
        typename Gemm::helper::ElementAccumulatorSelector<ElementA, ElementB>::ElementAccumulator;
    using LayoutAInL1 = typename CopyGmToL1A::LayoutDst;
    using LayoutBInL1 = typename CopyGmToL1B::LayoutDst;
    using LayoutAInL0 = typename CopyL1ToL0A::LayoutDst;
    using LayoutBInL0 = typename CopyL1ToL0B::LayoutDst;
    using LayoutCInL0 = layout::zN;

    static constexpr uint32_t STAGES = DispatchPolicy::STAGES;
    static constexpr bool ENABLE_UNIT_FLAG = DispatchPolicy::ENABLE_UNIT_FLAG;
    static constexpr bool ENABLE_SHUFFLE_K = DispatchPolicy::ENABLE_SHUFFLE_K;
    static constexpr bool ENABLE_ABBA = DispatchPolicy::ENABLE_ABBA;
    const uint32_t L1Size = ArchTag::L1_SIZE;
    const uint32_t L1ASize = L1TileShape::M * L1TileShape::K * sizeof(ElementA);
    const uint32_t L1BSize = L1TileShape::K * L1TileShape::N * sizeof(ElementB);
    const uint32_t cSize = L1TileShape::M * L1TileShape::N * sizeof(ElementAccumulator);
    const uint32_t BlockCnt = L1TileShape::M * L1TileShape::N;
    const uint32_t L0ASize = ArchTag::L0A_SIZE;
    const uint32_t L0BSize = ArchTag::L0B_SIZE;
    const uint32_t L0CSize = ArchTag::L0C_SIZE;
    const uint32_t L0A_PINGPONG_BUF_LEN = (L0ASize / STAGES);
    const uint32_t L0B_PINGPONG_BUF_LEN = (L0BSize / STAGES);
    const uint32_t l0CBlockNum = ArchTag::L0C_SIZE / cSize;

    static_assert(std::is_same_v<LayoutC, layout::RowMajor>, "LayoutC only support RowMajor yet!");

    __forceinline__ [aicore]
    BlockGemm(Arch::Resource<ArchTag> &resource, uint32_t l1BufAddrStart = 0)
    {
        uint32_t l1AOffset = l1BufAddrStart;
        uint32_t l1BOffset = l1BufAddrStart + L1ASize * STAGES;
        for(uint32_t i = 0; i < STAGES; i++){
            l1ATensor[i] = resource.l1Buf.template GetBufferByByte<ElementA>(l1AOffset + L1ASize * i);
            l1BTensor[i] = resource.l1Buf.template GetBufferByByte<ElementB>(l1BOffset + L1BSize * i);
            l0ATensor[i] = resource.l0ABuf.template GetBufferByByte<ElementA>(L0A_PINGPONG_BUF_LEN * i);
            l0BTensor[i] = resource.l0BBuf.template GetBufferByByte<ElementB>(L0B_PINGPONG_BUF_LEN * i);
            // Assign event ID for each stages
            l1AEventList[i] = i;
            l1BEventList[i] = i + STAGES;
            l0AEventList[i] = i;
            l0BEventList[i] = i + STAGES;
            // The event id that needs to be set before the loop
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1AEventList[i]);
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEventList[i]);
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0AEventList[i]);
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0BEventList[i]);
        }
        l0CTensor = resource.l0CBuf.template GetBufferByByte<ElementAccumulator>(0);
    }
    // destroy function
    __forceinline__ [aicore]
    ~BlockGemm()
    {
        for (uint32_t i = 0; i < STAGES; i++)
        {
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1AEventList[i]);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEventList[i]);
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0AEventList[i]);
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0BEventList[i]);
        }
    }

    __forceinline__ [aicore]
    void operator()(
        AscendC::GlobalTensor<ElementA> const &gmA, LayoutA const &layoutA,
        AscendC::GlobalTensor<ElementB> const &gmB, LayoutB const &layoutB,
        AscendC::GlobalTensor<ElementC> const &gmC, LayoutC const &layoutC,
        AscendC::GlobalTensor<ElementA> const &gmNextBlockA,
        AscendC::GlobalTensor<ElementB> const &gmNextBlockB,
        GemmCoord const &actualShape, GemmCoord const &actualShapeNext,
        bool isFirstBlock, bool hasNextBlock, uint32_t singleIdx)
    {
        uint32_t K = actualShape.k();
        uint32_t maxKPerBlock = L1TileShape::K;
        uint32_t kLoops = CeilDiv(K, maxKPerBlock);
        uint32_t kLoopsNext = CeilDiv(actualShapeNext.k(), maxKPerBlock);
        uint32_t startTileIdx{0};
        if (ENABLE_SHUFFLE_K)
        {
            startTileIdx = AscendC::GetBlockIdx();
        }
        uint32_t firstTileIdx = startTileIdx % kLoops;
        uint32_t firstTileIdxNext = startTileIdx % kLoopsNext;
        uint32_t lastTileIdx = (startTileIdx + kLoops - 1) % kLoops;
        uint32_t kGmActual = (firstTileIdx == kLoops - 1) ? (K - firstTileIdx * maxKPerBlock) : maxKPerBlock;
        auto layoutAInL1 = LayoutAInL1::template MakeLayout<ElementA>(L1TileShape::M, L1TileShape::K);
        auto layoutBInL1 = LayoutBInL1::template MakeLayout<ElementB>(L1TileShape::K, L1TileShape::N);
        for (uint32_t kIdx = 0; kIdx < kLoops; kIdx++)
        {
            uint32_t shuffleKIdx = (startTileIdx + kIdx) % kLoops;
            if (shuffleKIdx == firstTileIdx && isFirstBlock)
            {
                auto layoutTileA = layoutA.GetTileLayout(MakeCoord(actualShape.m(), kGmActual));
                auto layoutTileB = layoutB.GetTileLayout(MakeCoord(kGmActual, actualShape.n()));
                MatrixCoord gmTileAOffset{0, shuffleKIdx * maxKPerBlock};
                auto gmTileA = gmA[layoutA.GetOffset(gmTileAOffset)];
                MatrixCoord gmTileBOffset{shuffleKIdx * maxKPerBlock, 0};
                auto gmTileB = gmB[layoutB.GetOffset(gmTileBOffset)];
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1AEventList[l1ListId]);
                copyGmToL1A(l1ATensor[l1ListId], gmTileA, layoutAInL1, layoutTileA);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1AEventList[l1ListId]);
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEventList[l1ListId]);
                copyGmToL1B(l1BTensor[l1ListId], gmTileB, layoutBInL1, layoutTileB);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1BEventList[l1ListId]);
            }
            l1ListIdNext = 1 - l1ListId;
            uint32_t kGmActualNext = 0;
            if (shuffleKIdx != lastTileIdx)
            {
                uint32_t shuffleKIdxNext = (startTileIdx + kIdx + 1) % kLoops;
                kGmActualNext = (shuffleKIdxNext == kLoops - 1) ? (K - shuffleKIdxNext * maxKPerBlock) : maxKPerBlock;
                auto layoutTileA = layoutA.GetTileLayout(MakeCoord(actualShape.m(), kGmActualNext));
                auto layoutTileB = layoutB.GetTileLayout(MakeCoord(kGmActualNext, actualShape.n()));
                MatrixCoord gmTileAOffset{0, shuffleKIdxNext * maxKPerBlock};
                auto gmTileA = gmA[layoutA.GetOffset(gmTileAOffset)];
                MatrixCoord gmTileBOffset{shuffleKIdxNext * maxKPerBlock, 0};
                auto gmTileB = gmB[layoutB.GetOffset(gmTileBOffset)];
                if (ENABLE_ABBA)
                {
                    if (shuffleKIdxNext % 2 == 1)
                    {
                        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEventList[l1ListIdNext]);
                        copyGmToL1B(l1BTensor[l1ListIdNext], gmTileB, layoutBInL1, layoutTileB);
                        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1BEventList[l1ListIdNext]);
                        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1AEventList[l1ListIdNext]);
                        copyGmToL1A(l1ATensor[l1ListIdNext], gmTileA, layoutAInL1, layoutTileA);
                        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1AEventList[l1ListIdNext]);
                    } else {
                        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1AEventList[l1ListIdNext]);
                        copyGmToL1A(l1ATensor[l1ListIdNext], gmTileA, layoutAInL1, layoutTileA);
                        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1AEventList[l1ListIdNext]);
                        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEventList[l1ListIdNext]);
                        copyGmToL1B(l1BTensor[l1ListIdNext], gmTileB, layoutBInL1, layoutTileB);
                        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1BEventList[l1ListIdNext]);
                    }
                } else {
                    AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1AEventList[l1ListIdNext]);
                    copyGmToL1A(l1ATensor[l1ListIdNext], gmTileA, layoutAInL1, layoutTileA);
                    AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1AEventList[l1ListIdNext]);
                    AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEventList[l1ListIdNext]);
                    copyGmToL1B(l1BTensor[l1ListIdNext], gmTileB, layoutBInL1, layoutTileB);
                    AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1BEventList[l1ListIdNext]);
                }
            }
            if (shuffleKIdx == lastTileIdx && hasNextBlock)
            {
                kGmActualNext = (firstTileIdxNext == kLoopsNext - 1)
                    ? (actualShapeNext.k() - firstTileIdxNext * maxKPerBlock) : maxKPerBlock;
                auto layoutTileA = layoutA.GetTileLayout(MakeCoord(actualShapeNext.m(), kGmActualNext));
                auto layoutTileB = layoutB.GetTileLayout(MakeCoord(kGmActualNext, actualShapeNext.n()));
                MatrixCoord gmTileAOffset{0, firstTileIdxNext * maxKPerBlock};
                auto gmNextTileA = gmNextBlockA[layoutA.GetOffset(gmTileAOffset)];
                MatrixCoord gmTileBOffset{firstTileIdxNext * maxKPerBlock, 0};
                auto gmNextTileB = gmNextBlockB[layoutB.GetOffset(gmTileBOffset)];
                if (ENABLE_ABBA)
                {
                    if (shuffleKIdx % 2 == 0)
                    {
                        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEventList[l1ListIdNext]);
                        copyGmToL1B(l1BTensor[l1ListIdNext], gmNextTileB, layoutBInL1, layoutTileB);
                        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1BEventList[l1ListIdNext]);
                        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1AEventList[l1ListIdNext]);
                        copyGmToL1A(l1ATensor[l1ListIdNext], gmNextTileA, layoutAInL1, layoutTileA);
                        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1AEventList[l1ListIdNext]);
                    } else {
                        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1AEventList[l1ListIdNext]);
                        copyGmToL1A(l1ATensor[l1ListIdNext], gmNextTileA, layoutAInL1, layoutTileA);
                        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1AEventList[l1ListIdNext]);
                        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEventList[l1ListIdNext]);
                        copyGmToL1B(l1BTensor[l1ListIdNext], gmNextTileB, layoutBInL1, layoutTileB);
                        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1BEventList[l1ListIdNext]);
                    }
                } else {
                    AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1AEventList[l1ListIdNext]);
                    copyGmToL1A(l1ATensor[l1ListIdNext], gmNextTileA, layoutAInL1, layoutTileA);
                    AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1AEventList[l1ListIdNext]);
                    AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEventList[l1ListIdNext]);
                    copyGmToL1B(l1BTensor[l1ListIdNext], gmNextTileB, layoutBInL1, layoutTileB);
                    AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1BEventList[l1ListIdNext]);
                }
            }

            uint32_t kL0TileSize = L0TileShape::K;
            uint32_t kL0Loops = CeilDiv(kGmActual, kL0TileSize);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(l1AEventList[l1ListId]);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(l1BEventList[l1ListId]);
            auto l1ATile = l1ATensor[l1ListId];
            auto l1BTile = l1BTensor[l1ListId];
            uint32_t mActual{0};
            uint32_t nActual{0};
            for (uint32_t kL0Idx = 0; kL0Idx < kL0Loops; kL0Idx++)
            {
                uint32_t kL0Actual = (kL0Idx == kL0Loops - 1) ? (kGmActual - kL0Idx * kL0TileSize) : kL0TileSize;
                LayoutAInL0 layoutAInL0 = LayoutAInL0::template MakeLayout<ElementA>(L1TileShape::M, kL0Actual);
                LayoutBInL0 layoutBInL0 = LayoutBInL0::template MakeLayout<ElementB>(kL0Actual, L1TileShape::N);
                uint32_t l1TileAOffset = layoutAInL1.GetOffset(MatrixCoord(0, kL0Idx * kL0TileSize));
                uint32_t l1TileBOffset = layoutBInL1.GetOffset(MatrixCoord(kL0Idx * kL0TileSize, 0));
                auto l1TileA = l1ATile[l1TileAOffset];
                auto l1TileB = l1BTile[l1TileBOffset];
                auto l0TileA = l0ATensor[l0ListId];
                auto l0TileB = l0BTensor[l0ListId];
                mActual = L1TileShape::M;
                nActual = L1TileShape::N;
                if (ENABLE_ABBA)
                {
                    if (shuffleKIdx % 2 == 0)
                    {
                        if (kL0Idx % 2 == 0) {
                            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0BEventList[l0ListId]);
                            copyL1ToL0B(l0TileB, l1TileB, layoutBInL0, layoutBInL1);
                            AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(l0BEventList[l0ListId]);
                            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0AEventList[l0ListId]);
                            copyL1ToL0A(l0TileA, l1TileA, layoutAInL0, layoutAInL1);
                            AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(l0AEventList[l0ListId]);
                        } else {
                            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0AEventList[l0ListId]);
                            copyL1ToL0A(l0TileA, l1TileA, layoutAInL0, layoutAInL1);
                            AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(l0AEventList[l0ListId]);
                            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0BEventList[l0ListId]);
                            copyL1ToL0B(l0TileB, l1TileB, layoutBInL0, layoutBInL1);
                            AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(l0BEventList[l0ListId]);
                        }
                    } else {
                        if (kL0Idx % 2 == 0) {
                            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0AEventList[l0ListId]);
                            copyL1ToL0A(l0TileA, l1TileA, layoutAInL0, layoutAInL1);
                            AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(l0AEventList[l0ListId]);
                            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0BEventList[l0ListId]);
                            copyL1ToL0B(l0TileB, l1TileB, layoutBInL0, layoutBInL1);
                            AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(l0BEventList[l0ListId]);
                        } else {
                            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0BEventList[l0ListId]);
                            copyL1ToL0B(l0TileB, l1TileB, layoutBInL0, layoutBInL1);
                            AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(l0BEventList[l0ListId]);
                            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0AEventList[l0ListId]);
                            copyL1ToL0A(l0TileA, l1TileA, layoutAInL0, layoutAInL1);
                            AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(l0AEventList[l0ListId]);
                        }
                    }
                } else {
                    AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0AEventList[l0ListId]);
                    copyL1ToL0A(l0TileA, l1TileA, layoutAInL0, layoutAInL1);
                    AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(l0AEventList[l0ListId]);
                    AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0BEventList[l0ListId]);
                    copyL1ToL0B(l0TileB, l1TileB, layoutBInL0, layoutBInL1);
                    AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(l0BEventList[l0ListId]);
                }
                if (kL0Idx == kL0Loops - 1)
                {
                    AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1AEventList[l1ListId]);
                    AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEventList[l1ListId]);
                    l1ListId = l1ListIdNext;
                    kGmActual = kGmActualNext;
                }
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(l0BEventList[l0ListId]);
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(l0AEventList[l0ListId]);
                tileMmad(l0CTensor[(singleIdx % l0CBlockNum) * BlockCnt], l0TileA, l0TileB, mActual, nActual, kL0Actual, (kIdx == 0) && (kL0Idx == 0));
                AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0AEventList[l0ListId]);
                AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0BEventList[l0ListId]);
                l0ListId = 1 - l0ListId;
            }
        }
        AscendC::SetFlag<AscendC::HardEvent::M_FIX>((int32_t)(singleIdx % l0CBlockNum));
        AscendC::WaitFlag<AscendC::HardEvent::M_FIX>((int32_t)(singleIdx % l0CBlockNum));
        auto layoutInL0X = LayoutCInL0::MakeLayoutInL0C(MakeCoord(L1TileShape::M, L1TileShape::N));
        LayoutC layoutBlock = layoutC.GetTileLayout(MakeCoord(actualShape.m(), actualShape.n()));
        copyL0CToGm(gmC, l0CTensor[(singleIdx % l0CBlockNum) * BlockCnt], layoutBlock, layoutInL0X);
    }
private:
    AscendC::LocalTensor<ElementA> l1ATensor[STAGES];
    AscendC::LocalTensor<ElementB> l1BTensor[STAGES];
    AscendC::LocalTensor<ElementA> l0ATensor[STAGES];
    AscendC::LocalTensor<ElementB> l0BTensor[STAGES];
    AscendC::LocalTensor<ElementAccumulator> l0CTensor;
    // Multi-stage event id list
    int32_t l1AEventList[STAGES];
    int32_t l1BEventList[STAGES];
    int32_t l0AEventList[STAGES];
    int32_t l0BEventList[STAGES];

    // The id of current stage
    uint32_t l1ListId{0};
    uint32_t l0ListId{0};
    uint32_t l1ListIdNext{0};

    TileMmad tileMmad;
    CopyGmToL1A copyGmToL1A;
    CopyGmToL1B copyGmToL1B;
    CopyL1ToL0A copyL1ToL0A;
    CopyL1ToL0B copyL1ToL0B;
    CopyL0CToGm copyL0CToGm;
};
} // namespace Catlass::Gemm::Block

namespace Catlass::Gemm::Kernel {

template<
    class ArchTag_,
    class Element_,
    class Layout_,
    uint32_t COMPUTE_LENGTH
>
struct PaddingMatrix {
public:
    using ArchTag = ArchTag_;
    using Element = Element_;
    using Layout = Layout_;
    using CopyGm2Ub = Catlass::Epilogue::Tile::CopyGm2Ub<
        ArchTag, Gemm::GemmType<Element, Catlass::layout::RowMajor>>;
    using CopyUb2Gm = Catlass::Epilogue::Tile::CopyUb2Gm<
        ArchTag, Gemm::GemmType<Element, Catlass::layout::RowMajor>>;
    using ComputeLayout = Catlass::layout::RowMajor;

    CopyGm2Ub copyGm2Ub;
    CopyUb2Gm copyUb2Gm;

    __forceinline__ [aicore]
    PaddingMatrix(Arch::Resource<ArchTag> &resource)
    {
        int64_t bufferOffset = 0;
        for (uint32_t i = 0; i < BUFFER_NUM; i++) {  //
            inputBuffer[i] = resource.ubBuf.template GetBufferByByte<Element>(bufferOffset * sizeof(Element));
            bufferOffset += COMPUTE_LENGTH;
        }
    }

    __forceinline__ [aicore]
    ComputeLayout GetPaddingComputeLayout(layout::RowMajor const &layout)
    {
        return ComputeLayout(layout.shape(0), layout.shape(1), layout.stride(0));
    }

    __forceinline__ [aicore]
    ComputeLayout GetPaddingComputeLayout(layout::ColumnMajor const &layout)
    {
        return ComputeLayout(layout.shape(1), layout.shape(0), layout.stride(1));
    }

    __forceinline__ [aicore]
    void operator()(AscendC::GlobalTensor<Element> const &dst, AscendC::GlobalTensor<Element> const &src,
        Layout layoutDst, Layout layoutSrc)
    {
        ComputeLayout computeLayoutSrc = GetPaddingComputeLayout(layoutSrc);
        ComputeLayout computeLayoutDst = GetPaddingComputeLayout(layoutDst);

        uint32_t aivNum = AscendC::GetBlockNum() * AscendC::GetSubBlockNum();
        uint32_t aivId = AscendC::GetBlockIdx();

        // Each line is a tile.
        uint32_t tilesNum = computeLayoutSrc.shape(0);
        uint32_t tileLen = computeLayoutSrc.shape(1);
        uint32_t paddingStride = computeLayoutDst.stride(0);

        uint32_t tilesPerAiv = tilesNum / aivNum;
        uint32_t tileRemain = tilesNum % aivNum;
        if (aivId < tileRemain) {
            tilesPerAiv++;
        }
        uint32_t mIdx = aivId * tilesPerAiv;
        if (aivId >= tileRemain) {
            mIdx += tileRemain;
        }
        MatrixCoord blockOffset(mIdx, 0);

        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(eventIds[0]);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(eventIds[1]);
        uint32_t coreLoops{0};
        if (paddingStride > COMPUTE_LENGTH) {
            // Handle the same tile on multiple loops.
            uint32_t loopsPerTile = CeilDiv(tileLen, COMPUTE_LENGTH);
            coreLoops = tilesPerAiv * loopsPerTile;
            for (uint32_t loopIdx = 0; loopIdx < coreLoops; ++loopIdx) {
                uint32_t tileIdx = loopIdx / loopsPerTile;
                uint32_t inTileLoopIdx = loopIdx % loopsPerTile;
                MatrixCoord loopOffset(tileIdx, inTileLoopIdx * COMPUTE_LENGTH);
                uint64_t gmSrcOffset = computeLayoutSrc.GetOffset(blockOffset + loopOffset);
                uint32_t actualDataNum = COMPUTE_LENGTH;
                if (tileLen - inTileLoopIdx * COMPUTE_LENGTH < COMPUTE_LENGTH) {
                    actualDataNum = tileLen - inTileLoopIdx * COMPUTE_LENGTH;
                }
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(eventIds[bufferIndex]);
                ComputeLayout dstLayout = computeLayoutDst.GetTileLayout(MatrixCoord(1, actualDataNum));
                ComputeLayout srcLayout = computeLayoutSrc.GetTileLayout(MatrixCoord(1, actualDataNum));
                ComputeLayout &ubLayout = dstLayout;
                copyGm2Ub(inputBuffer[bufferIndex], src[gmSrcOffset], ubLayout, srcLayout);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(eventIds[bufferIndex]);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(eventIds[bufferIndex]);
                uint64_t gmDstOffset = computeLayoutDst.GetOffset(blockOffset + loopOffset);
                copyUb2Gm(dst[gmDstOffset], inputBuffer[bufferIndex], dstLayout, ubLayout);
                AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(eventIds[bufferIndex]);
                bufferIndex = 1 - bufferIndex;
            }
        } else {
            // Handle multiple tile each loop.
            uint32_t tilesPerLoop = COMPUTE_LENGTH / paddingStride;
            coreLoops = CeilDiv(tilesPerAiv, tilesPerLoop);
            for (uint32_t loopIdx = 0; loopIdx < coreLoops; ++loopIdx) {
                uint32_t tileIdx = loopIdx * tilesPerLoop;
                MatrixCoord tileOffset(tileIdx, 0);
                uint64_t gmSrcOffset = computeLayoutSrc.GetOffset(blockOffset + tileOffset);
                uint32_t actualTilesNum = tilesPerLoop;
                if (tilesPerAiv - tileIdx < tilesPerLoop) {
                    actualTilesNum = tilesPerAiv - tileIdx;
                }
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(eventIds[bufferIndex]);
                ComputeLayout dstLayout = computeLayoutDst.GetTileLayout(MatrixCoord(actualTilesNum, tileLen));
                ComputeLayout srcLayout = computeLayoutSrc.GetTileLayout(MatrixCoord(actualTilesNum, tileLen));
                ComputeLayout &ubLayout = dstLayout;
                copyGm2Ub(inputBuffer[bufferIndex], src[gmSrcOffset], ubLayout, srcLayout);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(eventIds[bufferIndex]);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(eventIds[bufferIndex]);
                uint64_t gmDstOffset = computeLayoutDst.GetOffset(blockOffset + tileOffset);
                copyUb2Gm(dst[gmDstOffset], inputBuffer[bufferIndex], dstLayout, ubLayout);
                AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(eventIds[bufferIndex]);
                bufferIndex = 1 - bufferIndex;
            }
        }
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(eventIds[0]);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(eventIds[1]);
    }

    __forceinline__ [aicore]
    ~PaddingMatrix() {}

private:
    static const uint32_t BUFFER_NUM = 2;
    AscendC::LocalTensor<Element> inputBuffer[BUFFER_NUM];
    AscendC::TEventID eventIds[BUFFER_NUM] = {EVENT_ID0, EVENT_ID1};
    uint32_t bufferIndex{ 0 };
    static_assert(BUFFER_NUM * COMPUTE_LENGTH * sizeof(Element) <= ArchTag::UB_SIZE, "Excedding the UB space!");
};

template<
    class BlockGemm_,
    class BlockEpilogue_ ,
    class BlockScheduler_ = void
>
class KernelGroupGemm{
public:
    using BlockGemm = BlockGemm_;
    using ArchTag = typename BlockGemm::ArchTag;
    using L1TileShape = typename BlockGemm::L1TileShape;
    using ElementA = typename BlockGemm::ElementA;
    using LayoutA = typename BlockGemm::LayoutA;
    using LayoutWA = typename BlockGemm::LayoutA;
    using ElementB = typename BlockGemm::ElementB;
    using LayoutB = typename BlockGemm::LayoutB;
    using LayoutWB = typename BlockGemm::LayoutB;
    using ElementC = typename BlockGemm::ElementC;
    using LayoutC = typename BlockGemm::LayoutC;
    using ElementAccumulator = typename BlockGemm::ElementAccumulator;

    using BlockEpilogue = BlockEpilogue_;
    using EpilogueParams = typename BlockEpilogue::Params;
    using ElementCompute =
        typename Catlass::Gemm::helper::ElementAccumulatorSelector<ElementA, ElementB>::ElementAccumulator;
    using ElementScalar = ElementCompute;
    static constexpr uint32_t MAX_TENSOR_COUNT = 32;

    const uint32_t maxMPerBlock = L1TileShape::M;
    const uint32_t maxNPerBlock = L1TileShape::N;
    const uint32_t cSize = maxMPerBlock * maxNPerBlock * sizeof(ElementAccumulator);
    const uint32_t l0CBlockNum = ArchTag::L0C_SIZE / cSize;

    static constexpr uint32_t STAGES = BlockGemm::STAGES;
    using BlockScheduler = BlockScheduler_;

    static const uint32_t COMPUTE_LENGTH_A = 96 * 1024 / sizeof(ElementA);
    using PaddingA = PaddingMatrix<ArchTag, ElementA, LayoutA, COMPUTE_LENGTH_A>;
    static const uint32_t COMPUTE_LENGTH_B = 96 * 1024 / sizeof(ElementB);
    using PaddingB = PaddingMatrix<ArchTag, ElementB, LayoutB, COMPUTE_LENGTH_B>;

    struct Params {
        // Data members
        uint32_t problemCount;
        GM_ADDR ptrA;
        GM_ADDR ptrB;
        GM_ADDR ptrWorkspace;
        GM_ADDR ptrWA;
        GM_ADDR ptrWB;
        GM_ADDR ptrX;
        GM_ADDR ptrD;

        ElementScalar alphaList[MAX_TENSOR_COUNT];
        ElementScalar betaList[MAX_TENSOR_COUNT];
        GemmCoord problemShapeList[MAX_TENSOR_COUNT];
        LayoutA layoutAList[MAX_TENSOR_COUNT];
        LayoutB layoutBList[MAX_TENSOR_COUNT];
        LayoutC layoutWorkspaceList[MAX_TENSOR_COUNT];
        LayoutA layoutWAList[MAX_TENSOR_COUNT];
        LayoutB layoutWBList[MAX_TENSOR_COUNT];

        // Methods
        __forceinline__ [aicore]
        Params() {}

        __forceinline__ [aicore]
        Params(uint32_t problemCount_, GM_ADDR ptrA_, GM_ADDR ptrB_, GM_ADDR ptrWorkspace_, GM_ADDR ptrWA_,
            GM_ADDR ptrWB_, GM_ADDR ptrX_, GM_ADDR ptrD_)
            : problemCount(problemCount_), ptrA(ptrA_), ptrB(ptrB_), ptrWorkspace(ptrWorkspace_),
              ptrWA(ptrWA_), ptrWB(ptrWB_), ptrX(ptrX_), ptrD(ptrD_)
        {}
    };

    __forceinline__ [aicore]
    KernelGroupGemm() {}

    __forceinline__ [aicore]
    ~KernelGroupGemm() {}

    __forceinline__ [aicore]
    size_t GetWorkspaceLen(layout::RowMajor layout)
    {
        return layout.shape(0) * layout.stride(0);
    }

    __forceinline__ [aicore]
    size_t GetWorkspaceLen(layout::ColumnMajor layout)
    {
        return layout.shape(1) * layout.stride(1);
    }

    template <int32_t CORE_TYPE = g_coreType>
    __forceinline__ [aicore]
    void operator()(Params &params) {}

    template <>
    __forceinline__ [aicore]
    void operator()<AscendC::AIC>(Params &params)
    {
        // Get matmul information from parameters
        uint32_t coreIdx = AscendC::GetBlockIdx();
        uint32_t coreNum = AscendC::GetBlockNum();
        uint64_t inGroupOffsetA = 0;
        uint64_t inGroupOffsetB = 0;
        uint64_t inGroupOffsetWorkspace = 0;
        uint32_t startCoreIdx = 0;
        uint32_t startLoopIdx;
        Arch::Resource<ArchTag> resource;
        BlockGemm blockGemm(resource);
        for (uint32_t groupIdx = 0; groupIdx < params.problemCount; ++groupIdx) {
            GemmCoord problemShape = params.problemShapeList[groupIdx];
            LayoutA layoutA = params.layoutAList[groupIdx];
            LayoutB layoutB = params.layoutBList[groupIdx];
            LayoutC layoutWorkspace = params.layoutWorkspaceList[groupIdx];
            LayoutA layoutWA = params.layoutWAList[groupIdx];
            LayoutB layoutWB = params.layoutWBList[groupIdx];
            Arch::CrossCoreWaitFlag(flagAivFinishPadding);
            AscendC::GlobalTensor<ElementA> gmA;
            gmA.SetGlobalBuffer((__gm__ ElementA *)params.ptrWA);
            AscendC::GlobalTensor<ElementB> gmB;
            gmB.SetGlobalBuffer((__gm__ ElementB *)params.ptrWB);
            AscendC::GlobalTensor<ElementC> gmC;
            gmC.SetGlobalBuffer((__gm__ ElementC *)params.ptrWorkspace);
            uint32_t M = problemShape.m();
            uint32_t N = problemShape.n();
            uint32_t K = problemShape.k();
            #pragma unroll
            for (uint32_t i = 0; i < l0CBlockNum; i++) {
                AscendC::SetFlag<AscendC::HardEvent::FIX_M>((int32_t)i);
            }
            uint32_t mLoops = CeilDiv(M, maxMPerBlock);
            uint32_t nLoops = CeilDiv(N, maxNPerBlock);
            uint32_t coreLoops = mLoops * nLoops;
            // Determine the starting loopIdx of the current core under the current groupIdx
            if (coreIdx < startCoreIdx) {
                startLoopIdx = coreIdx + coreNum - startCoreIdx;
            } else {
                startLoopIdx = coreIdx - startCoreIdx;
            }
            uint32_t singleIdx = 0;
            for (uint32_t loopIdx = startLoopIdx; loopIdx < coreLoops; loopIdx += AscendC::GetBlockNum()) {
                uint32_t mGmBlockIdx = loopIdx / nLoops;
                uint32_t nGmBlockIdx = loopIdx % nLoops;
                uint32_t mGmActual = (mGmBlockIdx == mLoops - 1) ? (M - mGmBlockIdx * maxMPerBlock) : maxMPerBlock;
                uint32_t nGmActual = (nGmBlockIdx == nLoops - 1) ? (N - nGmBlockIdx * maxNPerBlock) : maxNPerBlock;
                bool isFirstBlock = (loopIdx == startLoopIdx);
                bool hasNextBlock = false;
                GemmCoord nextActualShape;
                uint32_t mNextGmBlockIdx = 0;
                uint32_t nNextGmBlockIdx = 0;
                if (loopIdx + AscendC::GetBlockNum() < coreLoops) {
                    hasNextBlock = true;
                    uint32_t nextLoopIdx = loopIdx + AscendC::GetBlockNum();
                    mNextGmBlockIdx = nextLoopIdx / nLoops;
                    nNextGmBlockIdx = nextLoopIdx % nLoops;
                    uint32_t mNextGmActual =
                        (mNextGmBlockIdx == mLoops - 1) ? (M - mNextGmBlockIdx * maxMPerBlock) : maxMPerBlock;
                    uint32_t nNextGmActual =
                        (nNextGmBlockIdx == nLoops - 1) ? (N - nNextGmBlockIdx * maxNPerBlock) : maxNPerBlock;
                    nextActualShape = MakeCoord(mNextGmActual, nNextGmActual, K);
                }
                GemmCoord actualShape{mGmActual, nGmActual, K};
                AscendC::WaitFlag<AscendC::HardEvent::FIX_M>((int32_t)singleIdx);
                MatrixCoord gmTileAOffset{mGmBlockIdx * maxMPerBlock, 0};
                auto gmTileA = gmA[inGroupOffsetA + layoutWA.GetOffset(gmTileAOffset)];
                MatrixCoord gmTileBOffset{0, nGmBlockIdx * maxNPerBlock};
                auto gmTileB = gmB[inGroupOffsetB + layoutWB.GetOffset(gmTileBOffset)];
                MatrixCoord gmTileCOffset{mGmBlockIdx * maxMPerBlock, nGmBlockIdx * maxNPerBlock};
                auto gmTileC = gmC[inGroupOffsetWorkspace + layoutWorkspace.GetOffset(gmTileCOffset)];
                MatrixCoord gmTileNextAOffset{mNextGmBlockIdx * maxMPerBlock, 0};
                auto gmTileNextA = gmA[inGroupOffsetA + layoutWA.GetOffset(gmTileNextAOffset)];
                MatrixCoord gmTileNextBOffset{0, nNextGmBlockIdx * maxNPerBlock};
                auto gmTileNextB = gmB[inGroupOffsetB + layoutWB.GetOffset(gmTileNextBOffset)];
                blockGemm(
                    gmTileA, layoutWA,
                    gmTileB, layoutWB,
                    gmTileC, layoutWorkspace,
                    gmTileNextA, gmTileNextB,
                    actualShape, nextActualShape, isFirstBlock, hasNextBlock, singleIdx
                );
                Arch::CrossCoreSetFlagWithReverse<0x2, PIPE_FIX>(flagAicFinishStore);
                AscendC::SetFlag<AscendC::HardEvent::FIX_M>((int32_t)singleIdx);
                singleIdx = (singleIdx + 1) % l0CBlockNum;
            }
            inGroupOffsetA += GetWorkspaceLen(layoutWA);
            inGroupOffsetB += GetWorkspaceLen(layoutWB);
            inGroupOffsetWorkspace += problemShape.m() * problemShape.n();
            startCoreIdx = (startCoreIdx + coreLoops) % coreNum;
            #pragma unroll
            for (uint32_t i = 0; i < l0CBlockNum; i++) {
                AscendC::WaitFlag<AscendC::HardEvent::FIX_M>((int32_t)i);
            }
        }
    }

    template <>
    __forceinline__ [aicore]
    void operator()<AscendC::AIV>(Params &params)
    {
        uint32_t coreIdx = AscendC::GetBlockIdx() / AscendC::GetSubBlockNum();
        uint32_t coreNum = AscendC::GetBlockNum();
        uint64_t inGroupOffsetA = 0;
        uint64_t inGroupOffsetWA = 0;
        uint64_t inGroupOffsetB = 0;
        uint64_t inGroupOffsetWB = 0;
        uint64_t inGroupOffsetWorkspace = 0;
        uint32_t startCoreIdx = 0;
        uint32_t startLoopIdx;
        GemmCoord blockShape = L1TileShape::ToCoord();
        Arch::Resource<ArchTag> resource;
        AscendC::GlobalTensor<ElementA> gmA;
        AscendC::GlobalTensor<ElementA> gmWA;
        gmA.SetGlobalBuffer(reinterpret_cast<__gm__ ElementA *>(params.ptrA));
        gmWA.SetGlobalBuffer(reinterpret_cast<__gm__ ElementA *>(params.ptrWA));
        AscendC::GlobalTensor<ElementB> gmB;
        AscendC::GlobalTensor<ElementB> gmWB;
        gmB.SetGlobalBuffer(reinterpret_cast<__gm__ ElementB *>(params.ptrB));
        gmWB.SetGlobalBuffer(reinterpret_cast<__gm__ ElementB *>(params.ptrWB));
        PaddingA paddingA(resource);
        PaddingB paddingB(resource);
        AscendC::GlobalTensor<ElementC> gmC;
        gmC.SetGlobalBuffer((__gm__ ElementC*)params.ptrWorkspace);
        for (uint32_t groupIdx = 0; groupIdx < params.problemCount; ++groupIdx) {
            GemmCoord problemShape = params.problemShapeList[groupIdx];
            LayoutA layoutA = params.layoutAList[groupIdx];
            LayoutB layoutB = params.layoutBList[groupIdx];
            LayoutC layoutWorkspace = params.layoutWorkspaceList[groupIdx];
            ElementScalar alpha_ = params.alphaList[groupIdx];
            ElementScalar beta_ = params.betaList[groupIdx];
            LayoutA layoutWA = params.layoutWAList[groupIdx];
            LayoutB layoutWB = params.layoutWBList[groupIdx];
            paddingA(gmWA[inGroupOffsetWA], gmA[inGroupOffsetA], layoutWA, layoutA);
            paddingB(gmWB[inGroupOffsetWB], gmB[inGroupOffsetB], layoutWB, layoutB);
            Catlass::Arch::CrossCoreBarrier<0x0, PIPE_MTE3>();
            Catlass::Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(flagAivFinishPadding);
            EpilogueParams epilogueParams{alpha_, beta_, params.ptrX, layoutWorkspace, params.ptrD, layoutWorkspace};
            BlockEpilogue blockEpilogue(resource, blockShape, epilogueParams);
            uint32_t M = problemShape.m();
            uint32_t N = problemShape.n();
            uint32_t K = problemShape.k();
            uint32_t mLoops = CeilDiv(M, maxMPerBlock);
            uint32_t nLoops = CeilDiv(N, maxNPerBlock);
            uint32_t coreLoops = mLoops * nLoops;
            if (coreIdx < startCoreIdx) {
                startLoopIdx = coreIdx + coreNum - startCoreIdx;
            } else {
                startLoopIdx = coreIdx - startCoreIdx;
            }
            for (uint32_t loopIdx = startLoopIdx; loopIdx < coreLoops; loopIdx += AscendC::GetBlockNum()) {
                uint32_t mGmBlockIdx = loopIdx / nLoops;
                uint32_t nGmBlockIdx = loopIdx % nLoops;
                uint32_t mGmActual = (mGmBlockIdx == mLoops - 1) ? (M - mGmBlockIdx * maxMPerBlock) : maxMPerBlock;
                uint32_t nGmActual = (nGmBlockIdx == nLoops - 1) ? (N - nGmBlockIdx * maxNPerBlock) : maxNPerBlock;
                GemmCoord actualShape{mGmActual, nGmActual, K};
                GemmCoord blockCoord{mGmBlockIdx, nGmBlockIdx, 0};
                Arch::CrossCoreWaitFlagWithReverse<0x2, PIPE_MTE3>(flagAicFinishStore);
                blockEpilogue(actualShape, blockCoord, gmC, layoutWorkspace, inGroupOffsetWorkspace);
            }
            inGroupOffsetA += problemShape.m() * problemShape.k();
            inGroupOffsetWA += GetWorkspaceLen(layoutWA);
            inGroupOffsetB += problemShape.k() * problemShape.n();
            inGroupOffsetWB += GetWorkspaceLen(layoutWB);
            inGroupOffsetWorkspace += problemShape.m() * problemShape.n();

            startCoreIdx = (startCoreIdx + coreLoops) % coreNum;
        }
    }

private:
    static constexpr Arch::FlagID FLAG_AIC_FINISH_STORE = 0;
    static constexpr Arch::FlagID RV_FLAG_AIC_FINISH_STORE = 1;
    Arch::CrossCoreFlagWithReverse<> flagAicFinishStore{FLAG_AIC_FINISH_STORE, RV_FLAG_AIC_FINISH_STORE};
    static constexpr Arch::FlagID FLAG_AIV_FINISH_STORE = 0;
    Arch::CrossCoreFlag flagAivFinishPadding{FLAG_AIV_FINISH_STORE};
};
} // namespace Catlass::Gemm::Kernel

using namespace Catlass;
using namespace matmul;

using ScalarType = float;
__forceinline__ [aicore]
layout::RowMajor GetWorkspaceLayout(layout::RowMajor layout, uint32_t align)
{
    if (align == 0)
    {
        return layout;
    }
    return layout::RowMajor(layout.shape(0), layout.shape(1),
        RoundUp(layout.shape(1), align));
}
__forceinline__ [aicore]
layout::ColumnMajor GetWorkspaceLayout(layout::ColumnMajor layout, uint32_t align)
{
    if (align == 0)
    {
        return layout;
    }
    return layout::ColumnMajor(layout.shape(0), layout.shape(1),
        RoundUp(layout.shape(0), align));
}

extern "C" __global__ __aicore__ void group_gemm(GM_ADDR a, GM_ADDR b, GM_ADDR c, GM_ADDR alpha, GM_ADDR beta,
                                                GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tiling_data, tiling);
    uint32_t problemCount = tiling_data.groupCount;
    uint32_t align = tiling_data.align;
    uint32_t paddingASize = tiling_data.paddingASize;
    uint32_t paddingBSize = tiling_data.paddingBSize;

    using LayoutA = layout::RowMajor;
    using LayoutB = layout::RowMajor;
    using LayoutC = layout::RowMajor;

    GM_ADDR gmWA = paddingASize > 0 ? workspace : a;
    GM_ADDR gmWB = paddingBSize > 0 ? workspace + paddingASize : b;
    GM_ADDR gmWorkspace = workspace + paddingASize + paddingBSize;

    using ArchTag = Arch::AtlasA2;
    constexpr bool enableUnitFlag = true;
    constexpr bool enableShuffleK = true;
    constexpr bool enableABBA = true;
    using GemmBlockDispatchPolicy = Gemm::GemmAtlasA2<enableUnitFlag, enableShuffleK, enableABBA>;
    using EpilogueBlockDispatchPolicy = Epilogue::EpilogueAtlasA2Gemm;
    using AType = Gemm::GemmType<half, LayoutA>;
    using BType = Gemm::GemmType<half, LayoutB>;
    using CType = Gemm::GemmType<half, LayoutC>;
    using XType = Gemm::GemmType<half, LayoutC>;
    using DType = XType;
    using ComputeType = CType;
    using L1TileShape = GemmShape<128, 256, 256>;
    using L0TileShape = GemmShape<128, 256, 64>;
    using TileShapeCast = MatrixShape<L1TileShape::M / 2, L1TileShape::N>;
    using GemmBlock = Gemm::Block::BlockGemm<GemmBlockDispatchPolicy, L1TileShape, L0TileShape, AType, BType, CType>;
    constexpr uint32_t computeLength = L1TileShape::MN / 2;
    using TileElemWiseAddGemm = Epilogue::Tile::TileElemWiseAdd<ArchTag, ComputeType, computeLength>;
    using TileElemWiseMulsGemm = Epilogue::Tile::TileElemWiseMuls<ArchTag, ComputeType, computeLength>;
    using TileElemWistCastD = Epilogue::Tile::TileCast<ArchTag, DType, ComputeType, TileShapeCast>;
    using EpilogueTileCopy = Epilogue::Tile::TileCopy<ArchTag, CType, XType, DType>;
    using EpilogueBlock = Epilogue::Block::BlockEpilogue<EpilogueBlockDispatchPolicy, CType, XType, DType, TileElemWiseAddGemm, TileElemWiseMulsGemm, TileElemWistCastD, EpilogueTileCopy>;
    using GroupGemmKernel = Gemm::Kernel::KernelGroupGemm<GemmBlock, EpilogueBlock>;
    typename GroupGemmKernel::Params params{problemCount, a, b, gmWorkspace, gmWA, gmWB, c, c}; // 这里得修改 gmX保存A * B

    for (uint32_t i = 0; i < problemCount; ++i) {
        uint32_t m = tiling_data.mList[i];
        uint32_t k = tiling_data.kList[i];
        uint32_t n = tiling_data.nList[i];
        params.problemShapeList[i] = GemmCoord{m, n, k};
        params.layoutAList[i] = LayoutA{m, k};
        params.layoutBList[i] = LayoutB{k, n};
        params.alphaList[i] = reinterpret_cast<__gm__ ScalarType *>(alpha)[i];
        params.betaList[i] = reinterpret_cast<__gm__ ScalarType *>(beta)[i];
        params.layoutWorkspaceList[i] = LayoutC{m, n};
        params.layoutWAList[i] = GetWorkspaceLayout(params.layoutAList[i], align);
        params.layoutWBList[i] = GetWorkspaceLayout(params.layoutBList[i], align);
    }
    GroupGemmKernel groupgemm;
    groupgemm(params);
}