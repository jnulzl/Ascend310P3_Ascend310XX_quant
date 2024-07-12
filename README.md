# 华为昇腾系列(Ascend310XX)模型量化(PTQ)

## 工具及文档

- [CANN-7.0.0.beta1](https://www.hiascend.com/developer/download/community/result?module=cann&cann=7.0.0.beta1)

	- [amct](https://www.hiascend.com/document/detail/zh/canncommercial/700/devtools/auxiliarydevtool/atlasamctonnx_16_0001.html)

	- [act](https://www.hiascend.com/document/detail/zh/canncommercial/700/inferapplicationdev/atctool/atlasatc_16_0002.html)

- [CANN-7.0.0-doc](https://www.hiascend.com/document/detail/zh/canncommercial/700/envdeployment/instg/instg_0001.html)

**参考官方文档安装以上工具(下面示例以CANN7.0.0为例说明)**

## 模型Int8量化(PTQ)

### 主要工具

- amct_onnx(模型量化，得到量化后的onnx模型)

- act(将onnx模型转化为华为NPU支持的模型格式，扩展名为**.om**)

### 模型量化(amct_onnx使用说明)

- 1. 量化数据准备

```shell
python3 scripts/preprocess_data.py models/yolov6_pre_params.json
```

其中`yolov6_pre_params.json`的内容如下：

```shell
{	
	"det_pre_params":
	{
		"img_root":"./imgs",
		"fixed_scale":1, 
		"batch_size":1, 
		"des_channels":3, 
		"des_height":640, 
		"des_width":640, 
		"is_BGR2RGB":1, 
		"means":[0, 0, 0], 
		"stds":[255, 255, 255]
	},

	# 以下内容对于生成量化数据这一步没用，故此省略
	......
}
```

对于上述`json`文件，运行完后生成的量化数据位于`data/calibration/imgs_batch1/`，如下所示：

```shell
batch1.bin
batch2.bin
batch3.bin
......
batchN.bin
```

- 2. 开始使用`amct_onnx`量化模型


```shell
# amct_onnx使用帮助如下所示
usage: amct_onnx [-h] command ...

amct_onnx command line tools.

positional arguments:
  command      The command option for amct_onnx specifies the function to be executed, which can be either of calibration or convert.
               
    calibration
               Run calibration-based Post-training Quantization. For more detailed usage, please run amct_onnx calibration --help.
               
    convert    Convert ONNX QAT model to Ascend quantized model. For more detailed usage, please run amct_onnx convert --help.
               

optional arguments:
  -h, --help   show this help message and exit
  
```
`amct_onnx`支持`PTQ`量化(`amct_onnx calibration`)以及装换`QAT`后的`onnx`模型(`amct_onnx convert`)。这里用到的是: `amct_onnx calibration`, 其使用说明如下:

```shell
amct_onnx calibration --help
usage: amct_onnx calibration [-h] --model MODEL --save_path SAVE_PATH [--input_shape INPUT_SHAPE] [--data_dir DATA_DIR] [--data_types DATA_TYPES] [--evaluator EVALUATOR]
                             [--calibration_config CALIBRATION_CONFIG] [--batch_num BATCH_NUM]

optional arguments:
  -h, --help            show this help message and exit
                        

required arguments:
  --model MODEL         The path to the input model. original model for calibration, qat model for convert
                        
  --save_path SAVE_PATH
                        The path to save the results, which should contain the prefix of the result model. For example: "./results/model_prefix"
                        

optional arguments:
  --input_shape INPUT_SHAPE
                        Shape of input data. Separate multiple nodes with semicolons (;). Use double quotation marks (") to enclose each argument. E.g.: "input_name1:n1,c1,h1,w1;input_name2:n2,c2,h2,w2"
                        
  --data_dir DATA_DIR   The path to the processed binary datasets. For a multi-input model, different input data must be stored in different directories. Names of all files in each directory must be sorted
                        in ascending lexicographic order. Use double quotation marks (") to enclose each argument. E.g.: "data/input1/;data/input2/"
                        
  --data_types DATA_TYPES
                        The dtype of the input data. Separate multiple nodes with semicolons (;). Use double quotation marks (") to enclose each argument. E.g.: "float32;float64"
                        
  --evaluator EVALUATOR
                        Python script contains evaluator based on base class "Evaluator".
                        
  --calibration_config CALIBRATION_CONFIG
                        The path to the user customized .cfg config_defination file. Default is set to None.
                        
  --batch_num BATCH_NUM
                        The number of data batches used to run PTQ calibration. Default is set to 1.
```
下面是一个使用例子:

```shell
amct_onnx calibration --model models/yolov6n.onnx --input_shape "images:12,3,640,640" --data_type float32 --data_dir data/calibration/imgs_batch1 --save_path ./output_Ascend310P3/yolov6n_int8 --batch_num 2
```
运行完后生成内容如下:

```shell
./output_Ascend310P3/yolov6n_int8_deploy_model.onnx：量化后的可在昇腾AI处理器部署的模型文件。
./output_Ascend310P3/yolov6n_int8_fake_quant_model.onnx：量化后的可在ONNX执行框架ONNX Runtime进行精度仿真的模型文件。
./output_Ascend310P3/yolov6n_int8_quant.json：量化信息文件（该文件名称和量化后模型名称保持统一），记录了量化模型同原始模型节点的映射关系，用于量化后模型同原始模型精度比对使用。
```

其它详细说明见:[AMCT_ONNX快速入门](https://www.hiascend.com/document/detail/zh/canncommercial/700/devtools/auxiliarydevtool/atlasamctonnx_16_0001.html)

### ONNX转OM

利用`atc`将`ONNX`模型转换为`OM`模型

- 转换量化后的模型

```shell
atc --model=./output_Ascend310P3/yolov6n_int8_deploy_model.onnx --framework=5 --output=./output_Ascend310P3/yolov6n_int8_deploy_model_inc_pre_int8 --input_shape="images:1,3,640,640" --soc_version=Ascend310P3 --output_type=FP32 --insert_op_conf=./models/model_aipp.cfg
```

- 直接转换原始FP32 ONNX模型(不量化)

```shell
atc --model=./models/yolov6n.onnx --framework=5 --output=./output_Ascend310P3/yolov6n_int8_deploy_model_inc_pre_fp16 --input_shape="images:1,3,640,640" --soc_version=Ascend310P3 --output_type=FP32 --insert_op_conf=./models/model_aipp.cfg
```
