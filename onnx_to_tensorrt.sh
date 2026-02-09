/usr/src/tensorrt/bin/trtexec \
    --onnx=results/last_model/best_34030_iter_acc_0.8723_class_score_0.8508_unknown_score_0.1308.onnx \
    --saveEngine=results/last_model/best_34030_iter_acc_0.8723_class_score_0.8508_unknown_score_0.1308.plan \
    --fp16
    # --inputIOFormats=fp16:chw \
    # --outputIOFormats=fp16:chw \
    # --verbose
