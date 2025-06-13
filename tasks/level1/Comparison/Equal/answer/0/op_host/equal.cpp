#include "equal_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

constexpr uint32_t DATA_SIZE_4 = 4;
constexpr uint32_t DATA_SIZE_2 = 2;
constexpr uint32_t DATA_SIZE_1 = 1;
constexpr uint32_t BLOCK_SIZE = 32;

namespace optiling
{

  static ge::graphStatus TilingFunc(gert::TilingContext *context)
  {
    EqualTilingData tiling;
    uint64_t ubSize;
    uint32_t bufferNum = 16;
    auto ascendcPlatform =
        platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    uint32_t dataType = context->GetInputDesc(0)->GetDataType();
    uint32_t totalLength = context->GetInputShape(0)->GetStorageShape().GetShapeSize();
    auto coreNum = ascendcPlatform.GetCoreNumAiv();
    uint32_t dataSize = 0;
    switch (dataType)
    {
    case ge::DT_FLOAT:
      dataSize = DATA_SIZE_4;
      break;
    case ge::DT_FLOAT16:
      dataSize = DATA_SIZE_2;
      break;
    case ge::DT_INT8:
      dataSize = DATA_SIZE_1;
      break;
    case ge::DT_UINT8:
      dataSize = DATA_SIZE_1;
      break;
    case ge::DT_INT32:
      dataSize = DATA_SIZE_4;
      break;
    case ge::DT_UINT32:
      dataSize = DATA_SIZE_4;
      break;
    default:
      dataSize = DATA_SIZE_4;
      break;
    }

    uint32_t pad32 = BLOCK_SIZE;
    uint32_t padMax = (ubSize / bufferNum / dataSize) / (2 * BLOCK_SIZE) * (2 * BLOCK_SIZE);

    if (totalLength < pad32 * coreNum)
    {
      coreNum =
          totalLength % pad32 ? totalLength / pad32 + 1 : totalLength / pad32;
    }
    context->SetBlockDim(coreNum);
    tiling.set_totalLength(totalLength);

    uint32_t tileNumMean = 0;
    uint32_t tileNumEnd = 0;
    uint32_t tileLengthMean = 0;
    uint32_t tileLengthEnd = 0;
    uint32_t blockLengthMean = 0;
    uint32_t blockLengthEnd = 0;
    // 如果总数据比32B还小，直接当尾数处理
    if (totalLength < pad32)
    {
      blockLengthMean = pad32;
      blockLengthEnd = totalLength;
      tileNumMean = 1;
      tileNumEnd = 1;
      tileLengthMean = totalLength;
      tileLengthEnd = totalLength;
    }
    else
    { // 总数据至少比32B大时
      // 总数据至少比32B大时
      uint32_t realTotalLength =
          totalLength % (pad32 * coreNum)
              ? // 补足totalLength到32B倍核心数的整数倍
              ((totalLength / (pad32 * coreNum)) + 1) * (pad32 * coreNum)
              : totalLength;
      if (coreNum == 0)
      {
        return ge::GRAPH_FAILED;
      }
      uint32_t maxBlockLength = realTotalLength / coreNum;
      if (realTotalLength - totalLength > maxBlockLength)
      {
        maxBlockLength = totalLength / coreNum;
      }

      if (maxBlockLength >
          padMax)
      { // maxBlockLength大于padMax时对maxBlockLength进行判定
        uint32_t padTemp = 0;
        for (uint32_t i = padMax / 2; i <= padMax; i += pad32)
        {
          padTemp = maxBlockLength % i == 0 ? i : padTemp;
        }
        if (padTemp)
        { // 如果maxBlockLength可以被PadTemp整除，那么padTemp就是tilelength
          blockLengthMean = maxBlockLength;
          blockLengthEnd = totalLength - blockLengthMean * (coreNum - 1);
          tileNumMean = blockLengthMean / padTemp;
          tileNumEnd = tileNumMean;
          tileLengthMean = padTemp;
          tileLengthEnd = blockLengthEnd - padTemp * (tileNumEnd - 1);
        }
        else
        { // 如果maxBlockLength不能被PadTemp整除，那么padMax就是tilelength
          blockLengthMean = maxBlockLength - maxBlockLength % padMax;
          blockLengthEnd = totalLength - blockLengthMean * (coreNum - 1);
          tileNumMean = blockLengthMean / padMax;
          tileNumEnd = blockLengthEnd % padMax
                           ? blockLengthEnd / padMax + 1
                           : (blockLengthEnd /
                              padMax); // 计算最后一个核心会不会多一个尾数块
          if (padMax >= blockLengthEnd)
          {
            tileNumEnd = 1;
          }
          tileLengthMean = padMax;
          tileLengthEnd =
              blockLengthEnd -
              padMax * (tileNumEnd - 1); // 计算最后一个核心的尾数块长度
        }
      }
      else
      { // maxBlockLength小于padMax时直接取maxBlockLength中的最大Pad32倍数
        if (maxBlockLength >= pad32)
        { // maxBlockLength大于pad32时
          blockLengthMean = maxBlockLength - maxBlockLength % pad32;
          blockLengthEnd = totalLength - blockLengthMean * (coreNum - 1);
          tileNumMean = 1; // 只有一个tileNum
          tileNumEnd =
              blockLengthEnd % pad32
                  ? blockLengthEnd / blockLengthMean + 1
                  : blockLengthEnd /
                        blockLengthMean; // 如果尾块不能32B对齐则多分配一个尾块
          if (blockLengthMean >= blockLengthEnd)
          {
            tileNumEnd = 1;
          }
          tileLengthMean = blockLengthMean;
          tileLengthEnd =
              blockLengthEnd -
              tileLengthMean *
                  (tileNumEnd - 1); // 将尾数彻底分给最后一个核心的最后一个tile
        }
        else
        { // maxBlockLength小于pad32时，前面的block优先分配32B数据
          blockLengthMean = pad32;
          blockLengthEnd = totalLength - pad32 * (coreNum - 1);
          tileNumMean = 1;
          tileNumEnd = 1;
          tileLengthMean = pad32;
          tileLengthEnd = blockLengthEnd;
        }
      }
    }
    tiling.set_totalLength(totalLength);
    tiling.set_tileNumMean(tileNumMean);
    tiling.set_tileNumEnd(tileNumEnd);
    tiling.set_tileLengthMean(tileLengthMean);
    tiling.set_tileLengthEnd(tileLengthEnd);
    tiling.set_blockLengthMean(blockLengthMean);
    tiling.set_blockLengthEnd(blockLengthEnd);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(),
                        context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = 0;
    std::cout << "*******************START*******************" << std::endl;
  std::cout << "coreNum = " << coreNum << std::endl;
  std::cout << "totalLength = " << tiling.get_totalLength() << std::endl;
  std::cout << "tileNumMean = " << tiling.get_tileNumMean() << std::endl;
  std::cout << "tileNumEnd = " << tiling.get_tileNumEnd() << std::endl;
  std::cout << "tileLengthMean = " << tiling.get_tileLengthMean() << std::endl;
  std::cout << "tileLengthEnd = " << tiling.get_tileLengthEnd() << std::endl;
  std::cout << "blockLengthMean = " << tiling.get_blockLengthMean() << std::endl;
  std::cout << "blockLengthEnd = " << tiling.get_blockLengthEnd() << std::endl;
  std::cout << "*******************END*******************" << std::endl;
    return ge::GRAPH_SUCCESS;
  }
}


namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    const gert::Shape* x1_shape = context->GetInputShape(0);
    gert::Shape* y_shape = context->GetOutputShape(0);
    *y_shape = *x1_shape;
    return GRAPH_SUCCESS;
}
}


namespace ops {
class Equal : public OpDef {
public:
    explicit Equal(const char* name) : OpDef(name)
    {
        this->Input("x1")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT, ge::DT_INT8, ge::DT_UINT8, ge::DT_INT32, ge::DT_UINT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("x2")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT, ge::DT_INT8, ge::DT_UINT8, ge::DT_INT32, ge::DT_UINT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BOOL, ge::DT_BOOL, ge::DT_BOOL, ge::DT_BOOL, ge::DT_BOOL, ge::DT_BOOL, ge::DT_BOOL})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});

       
        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
        this->AICore().AddConfig("ascend910_93");
    }
};
OP_ADD(Equal);
}
