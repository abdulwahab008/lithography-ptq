from quant_layers.matmul import MinMaxQuantMatMul, PTQSLBatchingQuantMatMul, PTQSLQuantMatMul, SoSPTQSLBatchingQuantMatMul, SoSPTQSLQuantMatMul
from quant_layers.linear import MinMaxQuantLinear, PTQSLBatchingQuantLinear, PostGeluPTQSLBatchingQuantLinear, PostGeluPTQSLQuantLinear
import torch
import torch.nn as nn
import torch.nn.functional as F

def quantize_int_weight(module):
    """
    Integer weight grid for a quantized module (values stored as int8; bit-width 2..8).
    Bias are not quantized and you can use raw bias.
    """
    assert hasattr(module, 'weight'), f"module {module} does not have weight"
    assert 1 < module.w_bit <= 8, (
        f"module {module}'s weight uses {module.w_bit} bits; "
        "only 2..8 bit signed grids are supported (stored as int8 tensors)"
    )

    w_int = (module.weight/module.w_interval).round_().clamp_(-module.w_qmax, module.w_qmax-1)
    w_int = w_int.cpu().detach().to(torch.int8)
    return w_int

def dequantize_int_weight(module, w_int):
    """
    Make sure it's the same module that generates w_int
    """
    w_sim = module.w_interval.cpu() * w_int.float()
    return w_sim

def quantize_matmul_input(input, interval, qmax, n_G, n_V, n_H, crb_groups, crb_rows, crb_cols):
    """
    quantize input matrix of matmul operation, with respect to sublayerwise padding settings
    """
    pad_groups = crb_groups*n_G - input.shape[1]
    pad_rows = crb_rows*n_V - input.shape[2]
    pad_cols = crb_cols*n_H - input.shape[3]

    x = F.pad(input, [0,pad_cols,0,pad_rows,0,pad_groups])
    x = x.view(-1,n_G,crb_groups,n_V,crb_rows,n_H,crb_cols)
    x = (x/interval).round_().clamp(-qmax,qmax-1)
    x = x.view(-1,n_G*crb_groups,n_V*crb_rows,n_H*crb_cols)
    x = x[:,:x.shape[1]-pad_groups,:x.shape[2]-pad_rows,:x.shape[3]-pad_cols]

    return x


def quantize_int_activation(module, input):
    """
    Quantize current inputs into uint8 and store them as an attribute of the module.

    The function is a pre-forward hook that need to be manually added to the calibrated model.
    You need to manipulate the cached data before feeding another batch of pictures.
    Weight/activation grids of 2..8 bits are packed into int8/uint8 tensors (not true int4 dtypes).

    For twin quantization:
    - For softmax, the MSB being 1 means using large interval, while MSB being 0 means using small interval.
    - For post-GELU, the MSB serves as sign bit. We use 1 for positive values and 0 for negative values.
    """
    if isinstance(module, PostGeluPTQSLQuantLinear) or isinstance(module, PostGeluPTQSLBatchingQuantLinear):
        assert 1 < module.a_bit <= 8, (
            f"module {module}'s activation uses {module.a_bit} bits; "
            "twin uint8 packing supports 2..8 bits"
        )

        x = input[0]
        
        int_input_pos = (x/module.a_interval).round_().clamp_(0, module.a_qmax-1)
        int_input_pos = int_input_pos.detach().to(torch.uint8) + 128

        int_input_neg = (x/module.a_neg_interval).round_().clamp_(-module.a_qmax+1, 0).abs()
        int_input_neg = int_input_neg.detach().to(torch.uint8)

        int_input = (int_input_pos + int_input_neg).cpu()
        module.int_input = [int_input]
    
    elif isinstance(module, MinMaxQuantLinear):
        assert 1 < module.a_bit <= 8, (
            f"module {module}'s activation uses {module.a_bit} bits; "
            "int8 cache supports 2..8 bits"
        )

        x = input[0]
        int_input = (x/module.a_interval).round_().clamp_(-module.a_qmax, module.a_qmax-1)
        int_input = int_input.cpu().detach().to(torch.int8)

        module.int_input = [int_input]
    
    elif isinstance(module, SoSPTQSLQuantMatMul) or isinstance(module, SoSPTQSLBatchingQuantMatMul):
        assert 1 < module.A_bit <= 8, f"module {module}'s matrix A uses {module.A_bit} bits (supported: 2..8)"
        assert 1 < module.B_bit <= 8, f"module {module}'s matrix B uses {module.B_bit} bits (supported: 2..8)"

        A, B = input[0], input[1]

        A_high = (A.clamp(module.split, 1)*(module.A_qmax-1)).round_().clamp_(0,module.A_qmax-1)
        A_high = A_high.detach().to(torch.uint8) + 128

        A_low = (A.clamp(0, module.split)/module.A_interval).round_().clamp_(0,module.A_qmax-1)
        A_low = A_low.detach().to(torch.uint8)
        
        A_int = (A_high + A_low).cpu()

        B_int = quantize_matmul_input(B,module.B_interval,module.B_qmax,module.n_G_B,module.n_V_B,module.n_H_B,module.crb_groups_B,module.crb_rows_B,module.crb_cols_B)
        B_int = B_int.cpu().detach().to(torch.int8)

        module.int_input = [A_int, B_int]

    elif isinstance(module, PTQSLQuantMatMul) or isinstance(module, PTQSLBatchingQuantMatMul):
        assert 1 < module.A_bit <= 8, f"module {module}'s matrix A uses {module.A_bit} bits (supported: 2..8)"
        assert 1 < module.B_bit <= 8, f"module {module}'s matrix B uses {module.B_bit} bits (supported: 2..8)"

        A, B = input[0], input[1]

        A_int = quantize_matmul_input(A,module.A_interval,module.A_qmax,module.n_G_A,module.n_V_A,module.n_H_A,module.crb_groups_A,module.crb_rows_A,module.crb_cols_A)
        A_int = A_int.cpu().detach().to(torch.int8)

        B_int = quantize_matmul_input(B,module.B_interval,module.B_qmax,module.n_G_B,module.n_V_B,module.n_H_B,module.crb_groups_B,module.crb_rows_B,module.crb_cols_B)
        B_int = B_int.cpu().detach().to(torch.int8)

        module.int_input = [A_int, B_int]


def get_model_int_weight(wrapped_modules):
    """
    Get quantized weights (in int8) of a model.

    Return:
        A dict, with modules' names as keys, and int weights as values.
    """

    int_weights = {}

    for name, m in wrapped_modules.items():
        try:
            int_weights[name] = quantize_int_weight(m)
        except:
            pass
    
    return int_weights
