onnx_path=$1
input_shape=$2
calibration_data_path=$3
save_model_name_prefix=$4
cali_batch_size=$5
soc_version=$6
insert_op_conf=$7
output_path_prefix=output_${soc_version}


export AMCT_LOG_FILE_LEVEL=INFO
export AMCT_LOG_LEVEL=INFO

mkdir -p ${output_path_prefix}

echo amct_onnx calibration --model ${onnx_path} --input_shape ${input_shape} --data_type "float32" --data_dir ${calibration_data_path}  --save_path ./${output_path_prefix}/${save_model_name_prefix} --batch_num ${cali_batch_size}
amct_onnx calibration --model ${onnx_path} --input_shape ${input_shape} --data_type "float32" --data_dir ${calibration_data_path}  --save_path ./${output_path_prefix}/${save_model_name_prefix} --batch_num ${cali_batch_size}

echo atc --model=./${output_path_prefix}/${save_model_name_prefix}_deploy_model.onnx --framework=5 --output=./${output_path_prefix}/${save_model_name_prefix}_deploy_model_inc_pre --input_shape=${input_shape} --soc_version=${soc_version} --output_type=FP32 --insert_op_conf=${insert_op_conf}
atc --model=./${output_path_prefix}/${save_model_name_prefix}_deploy_model.onnx --framework=5 --output=./${output_path_prefix}/${save_model_name_prefix}_deploy_model_inc_pre --input_shape=${input_shape} --soc_version=${soc_version} --output_type=FP32 --insert_op_conf=${insert_op_conf}

rm -rf amct_log
rm -rf fusion_result.json
rm -rf kernel_meta
