#include "kernel_operator.h"

using namespace AscendC;

typedef struct {
    uint32_t num_rows;
    uint32_t num_columns;
    uint32_t min_diag;
    uint32_t totalLength;
    uint32_t tileNum;
} TilingDataDef;

constexpr int32_t BUFFER_NUM = 2;

class KernelEye {
public:
    __aicore__ inline KernelEye() {}
    
    __aicore__ inline void Init(GM_ADDR y, GM_ADDR y_ref, uint32_t num_rows, uint32_t num_columns, 
                              uint32_t min_diag, uint32_t tileNum)
    {
        this->num_rows = num_rows;
        this->num_columns = num_columns;
        this->min_diag = min_diag;
        this->tileNum = tileNum;
        this->tileLength = num_rows * num_columns;

        // 每个核处理32个批次（48*32=1536总条目）
        uint32_t batch_stride = num_rows * num_columns;
        
        yGm.SetGlobalBuffer((__gm__ float*)y + GetBlockIdx() * 32 * batch_stride, 32 * batch_stride);
        pipe.InitBuffer(outQueue, BUFFER_NUM, tileLength * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        for (int32_t i = 0; i < 32; i++) {
            GenerateMatrix(i);
        }
    }

private:
    __aicore__ inline void GenerateMatrix(int32_t batch_idx)
    {
        LocalTensor<float> zLocal = outQueue.AllocTensor<float>();
        
        // 初始化为0
        AscendC::Duplicate(zLocal, 0.0f, tileLength);
        
        // 设置对角线为1
        for (uint32_t i = 0; i < min_diag; i++) {
            zLocal.SetValue(i * num_columns + i, 1.0f);
        }
        
        outQueue.EnQue(zLocal);
        
        // 数据搬运到GM
        LocalTensor<float> out_tensor = outQueue.DeQue<float>();
        AscendC::DataCopy(yGm[batch_idx * tileLength], out_tensor, tileLength);
        outQueue.FreeTensor(out_tensor);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> outQueue;
    AscendC::GlobalTensor<float> yGm;
    
    uint32_t num_rows;
    uint32_t num_columns;
    uint32_t min_diag;
    uint32_t tileNum;
    uint32_t tileLength;
};

extern "C" __global__ __aicore__ void eye(GM_ADDR y, GM_ADDR y_ref, GM_ADDR workspace) 
{
    TilingDataDef tiling_data = {
        .num_rows = 18,
        .num_columns = 10,
        .min_diag = 10,
        .totalLength = 180,
        .tileNum = 1
    };
    
    KernelEye op;
    op.Init(y, y_ref, tiling_data.num_rows, tiling_data.num_columns, 
            tiling_data.min_diag, tiling_data.tileNum);
    op.Process();
}