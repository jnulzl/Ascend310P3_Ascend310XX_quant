PYTHON_PACKAGES_PATH=$PYTHON_ENV/lib/python3.8/site-packages
pyinstaller --add-data "${PYTHON_PACKAGES_PATH}/amct_onnx/custom_op/libamct_onnx_ops.so:amct_onnx/custom_op/" --add-data "${PYTHON_PACKAGES_PATH}/amct_onnx/lib/libamct_ncx.so:amct_onnx/lib/" --add-data "${PYTHON_PACKAGES_PATH}/amct_onnx/lib/libquant_onnx.so:amct_onnx/lib/" --add-data "${PYTHON_PACKAGES_PATH}/amct_onnx/.version:./amct_onnx" --add-data "${PYTHON_PACKAGES_PATH}/onnxruntime/capi/libonnxruntime_providers_shared.so:./onnxruntime/capi/" --add-data "${PYTHON_PACKAGES_PATH}/amct_onnx/capacity/capacity_config.csv:./amct_onnx/capacity/"  --onefile  /home/malong/jnulzl/Ascend310/amct_python3/bin/amct_onnx
