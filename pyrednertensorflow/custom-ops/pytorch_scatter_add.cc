/* Copyright 2019 Seyoung Park. All Rights Reserved.

==============================================================================*/

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"
#include <stdint.h>

using namespace tensorflow;

/* Tensorflow custom ops does not allow parameter types of list of 
   various data types. Therefore, we can't pass a list but we have
   to pass each objects individually. 

   Consult Tensorflow source code: /tensorflow/core/framework/tensor.h
   for what is supported by Tensorflow

   References:
   - https://www.tensorflow.org/guide/extend/op

   - Multi-threaded CPU kernels: https://www.tensorflow.org/guide/extend/op#multi-threaded_cpu_kernels

*/

REGISTER_OP("PytorchScatterAdd")
    .Input("ref: float")  // Tensor
    .Input("indices: int32")  // Tensor
    .Input("updates: float")  // Tensor
    .Output("output: float")  // Tensor
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(0)); // The output shape is same as ref
      return Status::OK();
    });


class PytorchScatterAddOp : public OpKernel {
 public:
  explicit PytorchScatterAddOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // Grab the input tensor

    const Tensor& ref_tensor = context->input(0);
    const Tensor& indices_tensor = context->input(1);
    const Tensor& updates_tensor = context->input(2);

    // Check shapes of inputs: https://www.tensorflow.org/guide/extend/op#conditional_checks_and_validation
    OP_REQUIRES(
        context, 
        TensorShapeUtils::IsMatrix(ref_tensor.shape()),
        errors::InvalidArgument("PytorchScatterAdd:ref expects a R x 2 Matrix.")
    );
    OP_REQUIRES(
        context, 
        TensorShapeUtils::IsMatrix(indices_tensor.shape()),
        errors::InvalidArgument("PytorchScatterAdd:indices expects a R x 2 Matrix.")
    );
    OP_REQUIRES(
        context, 
        TensorShapeUtils::IsMatrix(updates_tensor.shape()),
        errors::InvalidArgument("PytorchScatterAdd:updates expects a R x 2 Matrix.")
    );
    
    

    auto ref = ref_tensor.matrix<float>();
    auto indices = indices_tensor.matrix<int32>();
    auto updates = updates_tensor.matrix<float>();
    
    // Create an output tensor
    // NOTE: The output datatype must match the Ops definition!!!.
    Tensor* output_tensor = NULL;
    OP_REQUIRES_OK(
        context, 
        context->allocate_output(0, ref_tensor.shape(), &output_tensor)  // output_tensor.shape == ref.shape
        
      );
    
    auto output = output_tensor->matrix<float>();
    
    for (int i = 0; i < ref_tensor.shape().dim_size(0); ++i) {
        for (int j = 0; j < ref_tensor.shape().dim_size(1); ++j) {
            output(i, j) = ref(i, j);
        }  
    }

    for (int i = 0; i < indices_tensor.shape().dim_size(0); ++i) {
        for (int j = 0; j < indices_tensor.shape().dim_size(1); ++j) {
            int index = indices(i, j);
            output(index, j) += updates(i, j);
        }  
    }

  }
};


REGISTER_KERNEL_BUILDER(
  Name("PytorchScatterAdd").Device(DEVICE_CPU),
  PytorchScatterAddOp);
  

