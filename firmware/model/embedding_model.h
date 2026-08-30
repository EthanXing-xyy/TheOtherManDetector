/*
* ImagiNet Compiler 5.11.5765.0+bf21cb94530df006d12234416a3023c30ee7e1f7
* Copyright © 2023- Imagimob AB, All Rights Reserved.
* 
* Generated at 08/27/2026 15:56:16 UTC. Any changes will be lost.
* 
* Model ID  65d628b2-f96c-4d40-b01f-07df4b12ebc6
* 
* Memory    Size                      Efficiency
* Buffers   8 bytes (RAM)             100 %
* State     451584 bytes (RAM)        100 %
* Readonly  603148 bytes (Flash)      100 %
* 
* Exported functions:
* 
*  @param imu imu. Input float[720,6].
*  @param audio audio. Input float[180,40].
*  @param data_out mean. Output float[32].
*  @return IPWIN_RET_SUCCESS (0) or IPWIN_RET_NODATA (-1), IPWIN_RET_ERROR (-2), IPWIN_RET_STREAMEND (-3)
*  int TMAI_compute(const float *imu, const float *audio, float *data_out);
* 
*  @description: Closes and flushes streams, free any heap allocated memory.
*  void TMAI_finalize(void);
* 
*  @description: Resets windows and neural networks(i.e. RNNs) to initial state.
*  @return IPWIN_RET_SUCCESS (0) or IPWIN_RET_NODATA (-1), IPWIN_RET_ERROR (-2), IPWIN_RET_STREAMEND (-3)
*  int TMAI_soft_reset(void);
* 
*  @description: Initializes buffers to initial state.
*  @return IPWIN_RET_SUCCESS (0) or IPWIN_RET_NODATA (-1), IPWIN_RET_ERROR (-2), IPWIN_RET_STREAMEND (-3)
*  int TMAI_init(void);
* 
* 
* Disclaimer:
*   The generated code relies on the optimizations done by the C compiler.
*   For example many for-loops of length 1 must be removed by the optimizer.
*   This can only be done if the functions are inlined and simplified.
*   Check disassembly if unsure.
*   tl;dr Compile using gcc with -O3 or -Ofast
* 
* Notes:
* 	-> This code was generated with DEEPCRAFT™ Model Converter using:
* 		ml-coretools 3.3.0.9784.
* 		tensorflow 2.19.0.
* 	-> This code requires the following Modus Toolbox libraries (add them to your
* 	project using the Library Manager):
* 		ml-middleware 3.2.0.
* 		ml-tflite-micro 3.2.0.
*/

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include "mtb_ml_model.h"
#define TMAI_API_USER_DEFINED

typedef int8_t q7_t;         // 8-bit fractional data type in Q1.7 format.
typedef int16_t q15_t;       // 16-bit fractional data type in Q1.15 format.
typedef int32_t q31_t;       // 32-bit fractional data type in Q1.31 format.
typedef int64_t q63_t;       // 64-bit fractional data type in Q1.63 format.
typedef float timestamp_t;

// Model GUID (16 bytes)
#define TMAI_MODEL_ID {0xb2, 0x28, 0xd6, 0x65, 0x6c, 0xf9, 0x40, 0x4d, 0xb0, 0x1f, 0x07, 0xdf, 0x4b, 0x12, 0xeb, 0xc6}


// First nibble is bit encoding, second nibble is number of bytes
#define IMAGINET_TYPES_NONE	(0x0)
#define IMAGINET_TYPES_FLOAT32	(0x14)
#define IMAGINET_TYPES_FLOAT64	(0x18)
#define IMAGINET_TYPES_INT8	(0x21)
#define IMAGINET_TYPES_INT16	(0x22)
#define IMAGINET_TYPES_INT32	(0x24)
#define IMAGINET_TYPES_INT64	(0x28)
#define IMAGINET_TYPES_Q7	(0x31)
#define IMAGINET_TYPES_Q15	(0x32)
#define IMAGINET_TYPES_Q31	(0x34)
#define IMAGINET_TYPES_BOOL	(0x41)
#define IMAGINET_TYPES_STRING	(0x54)
#define IMAGINET_TYPES_D8	(0x61)
#define IMAGINET_TYPES_D16	(0x62)
#define IMAGINET_TYPES_D32	(0x64)
#define IMAGINET_TYPES_UINT8	(0x71)
#define IMAGINET_TYPES_UINT16	(0x72)
#define IMAGINET_TYPES_UINT32	(0x74)
#define IMAGINET_TYPES_UINT64	(0x78)


#define TMAI_COMPUTE_INPUTS (2)
#define TMAI_COMPUTE_OUTPUTS (1)
#define TMAI_COMPUTE_IN_TYPE float
#define TMAI_COMPUTE_IN_TYPE_ID IMAGINET_TYPES_FLOAT32
#define TMAI_COMPUTE_OUT_TYPE float
#define TMAI_COMPUTE_OUT_TYPE_ID IMAGINET_TYPES_FLOAT32
#define TMAI_COMPUTE_OUT_NO_COPY false

// imu [720,6] (17280 bytes)
#define TMAI_IMU_RANK (2)
#define TMAI_IMU_SHAPE ((int[]){6, 720})
#define TMAI_IMU_COUNT (4320)
#define TMAI_IMU_BYTES (17280)
#define TMAI_IMU_TYPE float
#define TMAI_IMU_TYPE_ID IMAGINET_TYPES_FLOAT32
#define TMAI_IMU_SHIFT 0
#define TMAI_IMU_OFFSET 0
#define TMAI_IMU_SCALE 1
#define TMAI_IMU_SYMBOLS { }

// audio [180,40] (28800 bytes)
#define TMAI_AUDIO_RANK (2)
#define TMAI_AUDIO_SHAPE ((int[]){40, 180})
#define TMAI_AUDIO_COUNT (7200)
#define TMAI_AUDIO_BYTES (28800)
#define TMAI_AUDIO_TYPE float
#define TMAI_AUDIO_TYPE_ID IMAGINET_TYPES_FLOAT32
#define TMAI_AUDIO_SHIFT 0
#define TMAI_AUDIO_OFFSET 0
#define TMAI_AUDIO_SCALE 1
#define TMAI_AUDIO_SYMBOLS { }

// data_out [32] (128 bytes)
#define TMAI_DATA_OUT_RANK (1)
#define TMAI_DATA_OUT_SHAPE ((int[]){32})
#define TMAI_DATA_OUT_COUNT (32)
#define TMAI_DATA_OUT_BYTES (128)
#define TMAI_DATA_OUT_TYPE float
#define TMAI_DATA_OUT_TYPE_ID IMAGINET_TYPES_FLOAT32
#define TMAI_DATA_OUT_SHIFT 0
#define TMAI_DATA_OUT_OFFSET 0
#define TMAI_DATA_OUT_SCALE 1
#define TMAI_DATA_OUT_SYMBOLS { }

#define TMAI_KEY_MAX (7)

#define TMAI_MODEL_RNN true

// Return codes
#define TMAI_RET_SUCCESS 0
#define TMAI_RET_NODATA -1
#define TMAI_RET_ERROR -2
#define TMAI_RET_STREAMEND -3

#define IPWIN_RET_SUCCESS 0
#define IPWIN_RET_NODATA -1
#define IPWIN_RET_ERROR -2
#define IPWIN_RET_STREAMEND -3

// Exported methods
int TMAI_compute(const float *restrict imu, const float *restrict audio, float *restrict data_out);
void TMAI_finalize(void);
int TMAI_soft_reset(void);
int TMAI_init(void);

// Symbol TMAI_PROFILING must be defined to enable profiling of models
// Symbol TMAI_PROFILING_LOG will enable printing the raw outputs of neural networks
/// @brief This method will print the region profiling results
void TMAI_print_region_profiling(void);
/// @brief Point this function pointer to your custom tick count function. This should populate the value pointed to by val with the current tick count.
extern int (*TMAI_get_ticks_ptr)(uint64_t*);

/// @brief This method will print neural network inference profiling results
void TMAI_mtb_models_profile_log();
/// @brief This method will print neural network information
void TMAI_mtb_models_print_info();
extern uint8_t TMAI_mtb_models_count;
extern mtb_ml_model_t* TMAI_mtb_models[];

// Profiling regions
#ifdef TMAI_PROFILING
    #define TMAI_REGIONS_COUNT 4
    #define TMAI_REGIONS_NAMES {\
    	"SET INPUT 0",\
    	"SET INPUT 1",\
    	"INVOKE CROSS_ONLY_T90_SEED2026_32D_PSOC6_CHUNKED_GRU",\
    	"GET OUTPUT 0",\
    }
    #define TMAI_REGIONS_NOTES {\
    	NULL,\
    	NULL,\
    	"See the model CPU/NPU cycles breakdown.",\
    	NULL,\
    }
#else
    #define TMAI_REGIONS_COUNT 0
    #define TMAI_REGIONS_NAMES {}
    #define TMAI_REGIONS_NOTES {}
#endif

// Call macros — invoke any exported function via a void* array of arguments
#define TMAI_COMPUTE_PTR(a) TMAI_compute((const float *)(a)[0], (const float *)(a)[1], (float *)(a)[2])
#define TMAI_FINALIZE_PTR(a) TMAI_finalize()
#define TMAI_SOFT_RESET_PTR(a) TMAI_soft_reset()
#define TMAI_INIT_PTR(a) TMAI_init()

typedef enum {
    TMAI_PARAM_UNDEFINED = 0,
    TMAI_PARAM_INPUT = 1,
    TMAI_PARAM_OUTPUT = 2,
    TMAI_PARAM_REFERENCE = 3,
    TMAI_PARAM_HANDLE = 7,
    TMAI_PARAM_CALLBACK = 8,
    TMAI_PARAM_OUTPUT_REF = 18,
} TMAI_param_attrib;

typedef char *label_text_t;

typedef struct {
    char* name;
    int size;
    label_text_t *labels;
} TMAI_shape_dim;

typedef struct {
    char* name;
    TMAI_param_attrib attrib;
    int32_t rank;
    TMAI_shape_dim *shape;
    int32_t count;
    int32_t bytes;
    int32_t type_id;
    float frequency;
    int shift;
    float scale;
    long offset;
} TMAI_param_def;

typedef enum {
    TMAI_FUNC_ATTRIB_NONE = 0,
    TMAI_FUNC_ATTRIB_CAN_FAIL = 1,
    TMAI_FUNC_ATTRIB_PUBLIC = 2,
    TMAI_FUNC_ATTRIB_INIT = 4,
    TMAI_FUNC_ATTRIB_DESTRUCTOR = 8,
} TMAI_func_attrib;

typedef struct {
    char* name;
    char* description;
    void* fn_ptr;
    TMAI_func_attrib attrib;
    int32_t param_count;
    TMAI_param_def *param_list;
} TMAI_func_def;

typedef struct {
    uint32_t size;
    uint32_t peak_usage;
} TMAI_mem_usage;

typedef enum {
    TMAI_API_TYPE_UNDEFINED = 0,
    TMAI_API_TYPE_FUNCTION = 1,
    TMAI_API_TYPE_QUEUE = 2,
    TMAI_API_TYPE_QUEUE_TIME = 3,
    TMAI_API_TYPE_USER_DEFINED = 4,
} TMAI_api_type;

typedef struct {
    uint32_t api_ver;
    uint8_t id[16];
    TMAI_api_type api_type;
    char* prefix;
    TMAI_mem_usage buffer_mem;
    TMAI_mem_usage static_mem;
    TMAI_mem_usage readonly_mem;
    int32_t func_count;
    TMAI_func_def *func_list;
} TMAI_api_def;

TMAI_api_def *TMAI_api(void);

#define TMAI_INPUT_META_COUNT 2
#define TMAI_OUTPUT_META_COUNT 1
#define TMAI_INPUT_META(i) ((TMAI_param_def*)(&TMAI_api()->func_list[0].param_list[(i)]))
#define TMAI_OUTPUT_META(i) ((TMAI_param_def*)(&TMAI_api()->func_list[0].param_list[(i) + 2]))

