import os
import sys
import json

def main(json_path):
    with open(json_path, "r") as fpR:
        quamt_all_params = json.load(fpR)
        
        onnx_path = quamt_all_params["onnxPath"]
        output_root_dir = quamt_all_params["outputRootDir"]
        input_shape = quamt_all_params["inputShape"]
        soc_version = quamt_all_params["socVersion"]
        insert_op_conf = quamt_all_params["insertOpConf"]
        
        # For int8 model
        if "detPreParams" in quamt_all_params and "quantInt8" in quamt_all_params:
            # 1. Generate quant data
            os.system("quant_data_preprocess %s"%(json_path))
            
            # 2. Quant
            quant_int8_params = quamt_all_params["quantInt8"]
            
            img_root = quamt_all_params["detPreParams"]["imgRoot"]
            calibration_data_root = quamt_all_params["detPreParams"]["calibrationDataRoot"]
            batch_size = quamt_all_params["detPreParams"]["batchSize"]

            calibration_data_path = "%s/batch%d"%(calibration_data_root, batch_size)
            save_model_name_prefix = os.path.basename(onnx_path).replace(".onnx","_int8")
            cali_batch_size = quant_int8_params["calBatchSize"]
            
            os.system("amct_onnx_act %s %s %s %s %d %s %s %s"%(
                onnx_path, input_shape, calibration_data_path, save_model_name_prefix, cali_batch_size, soc_version, insert_op_conf, output_root_dir))
                    
        if "quantFp16" in quamt_all_params or ("quantFp16" not in quamt_all_params and "quantInt8" not in quamt_all_params):
            save_model_name_prefix = os.path.basename(onnx_path).replace(".onnx","_fp16")
            
            os.system("act_onnx %s %s %s %s %s %s"%(
                onnx_path, input_shape, save_model_name_prefix, soc_version, insert_op_conf, output_root_dir))


if __name__ == '__main__':
    if 2 != len(sys.argv):
        print("Usage:\n\tpython %s json_file"%(sys.argv[0]))
        sys.exit(0)
    main(sys.argv[1])



  
