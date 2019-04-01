package dnn

import "errors"

//Tensor is the basic data structure of convolutional neural networks.  It can be used for regular neural networks too.
type Tensor struct {
	dims    []int32   // I use int32 because there is a ton of libraries in c, and this will make the conversion easier.
	stride  []int32   //stride is used for placement of values and grabing values.
	f32data []float32 //place to hold data in float32
	f64data []float64 //place to hold data in float64
	nhwc    bool
}

//CreateTensor creates a tensor according to the values passed. If len(dims) not >=4 and <=8 an error will return!
//f64 is a place holder only thing available is float32
func CreateTensor(dims []int32, f64, NHWC bool) (*Tensor, error) {
	if len(dims) < 4 || len(dims) > 8 {
		return nil, errors.New("Not a valid length of dims")
	}
	f64 = false
	if f64 {
		return &Tensor{
			dims:    dims,
			stride:  findstride(dims),
			f64data: make([]float64, findvolume(dims)),
			nhwc:    NHWC,
		}, nil
	}
	return &Tensor{
		dims:    dims,
		stride:  findstride(dims),
		f32data: make([]float32, findvolume(dims)),
		nhwc:    NHWC,
	}, nil
}

func (t *Tensor) Set(value float64, dimlocation []int32) {
	if t.f32data != nil {

	}
}
func findrecursivelocation(dims, stride []int32, val int32) (location int32) {
	if len(dims) == 1 {
		val += dims[0] * stride[0]
		return val

	}
	val += dims[0] * stride[0]
	return findrecursivelocation(dims[1:], stride[1:], val)
}

//findvolume will save us a lot of trouble.
func findvolume(dims []int32) (vol int32) {
	vol = 1
	for i := range dims {
		vol *= dims[i]
	}
	return vol
}
func findstride(dims []int32) (stride []int32) {
	stride = make([]int32, len(dims))
	strider := int32(1)
	for i := len(dims) - 1; i >= 0; i-- {
		stride[i] = strider
		strider *= dims[i]
	}
	return stride
}
