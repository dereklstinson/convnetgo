package dnn

import "testing"

func TestConvolution_Forward(t *testing.T) {

	var err error
	tensor, err := CreateTensor([]int{32, 20, 32, 32}, false)
	if err != nil {
		t.Error(err)
	}
	dxtensor, err := CreateTensor([]int{32, 20, 32, 32}, false)
	if err != nil {
		t.Error(err)
	}
	weights, err := CreateTensor([]int{32, 20, 5, 5}, false)
	if err != nil {
		t.Error(err)
	}
	wb, err := CreateTensor([]int{128, 1, 1, 1}, false)
	for i := range tensor.f32data {
		tensor.f32data[i] = float32(i) / 100
	}
	for i := range weights.f32data {
		weights.f32data[i] = float32(i)
	}
	convolution := CreateConvolution()
	err = convolution.Set([]int{2, 2}, []int{2, 2}, []int{2, 2}, false)
	if err != nil {
		t.Error(err)
	}
	outputdims := convolution.FindOutputDims(tensor, weights)

	output, err := CreateTensor(outputdims, false)
	if err != nil {
		t.Error(err)
	}
	doutput, err := CreateTensor(outputdims, false)
	if err != nil {
		t.Error(err)
	}
	for i := 0; i < 5; i++ {
		err = convolution.Forward(tensor, weights, wb, output)
		if err != nil {
			t.Error(err)
		}
		err = convolution.BackwardData(dxtensor, weights, doutput)
		if err != nil {
			t.Error(err)
		}
		err = convolution.BackwardData(tensor, weights, doutput)
		if err != nil {
			t.Error(err)
		}
	}

}
