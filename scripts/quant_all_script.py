import os
import sys
import json

def main(json_path):
    with open(json_path, "r") as fpR:
        quamt_all_params = json.load(fpR)
        
        onnx_path = quamt_all_params["onnx_path"]
        input_shape = quamt_all_params["input_shape"]
        soc_version = quamt_all_params["soc_version"]
        insert_op_conf = quamt_all_params["insert_op_conf"]
        
        # For int8 model
        if "det_pre_params" in quamt_all_params and "quant_int8" in quamt_all_params:
            # 1. Generate quant data
            os.system("quant_data_preprocess %s"%(json_path))
            
            # 2. Quant
            quant_int8_params = quamt_all_params["quant_int8"]
            
            img_root = quamt_all_params["det_pre_params"]["img_root"]
            batch_size = quamt_all_params["det_pre_params"]["batch_size"]

            calibration_data_path = "data/calibration/%s_batch%d"%(os.path.basename(img_root), batch_size)
            save_model_name_prefix = os.path.basename(onnx_path).replace(".onnx","_int8")
            cali_batch_size = quant_int8_params["cali_batch_size"]
            
            os.system("amct_onnx_act %s %s %s %s %d %s %s"%(
                onnx_path, input_shape, calibration_data_path, save_model_name_prefix, cali_batch_size, soc_version, insert_op_conf))
                    
        if "quant_fp16" in quamt_all_params:
            save_model_name_prefix = os.path.basename(onnx_path).replace(".onnx","_fp16")
            
            os.system("act_onnx %s %s %s %s %s"%(
                onnx_path, input_shape, save_model_name_prefix, soc_version, insert_op_conf))


if __name__ == '__main__':
    if 2 != len(sys.argv):
        print("Usage:\n\tpython %s json_file"%(sys.argv[0]))
        sys.exit(0)
    main(sys.argv[1])



  
