onnx_path=$1
input_shape=$2
save_model_name_prefix=$3
soc_version=$4
insert_op_conf=$5
output_path_prefix=output_${soc_version}


export AMCT_LOG_FILE_LEVEL=INFO
export AMCT_LOG_LEVEL=INFO

mkdir -p ${output_path_prefix}

echo atc --model=${onnx_path} --framework=5 --output=./${output_path_prefix}/${save_model_name_prefix}_deploy_model_inc_pre --input_shape=${input_shape} --soc_version=${soc_version} --output_type=FP32 --insert_op_conf=${insert_op_conf}
atc --model=${onnx_path} --framework=5 --output=./${output_path_prefix}/${save_model_name_prefix}_deploy_model_inc_pre --input_shape=${input_shape} --soc_version=${soc_version} --output_type=FP32 --insert_op_conf=${insert_op_conf}

rm -rf amct_log
rm -rf fusion_result.json
rm -rf kernel_meta
