package convnetgo

import (
	"errors"
	"math"
	"math/rand"
)

//Tensor is the basic data structure of convolutional neural networks.  It can be used for regular neural networks too.
type Tensor struct {
	dims    []int     // I use int because there is a ton of libraries in c, and this will make the conversion easier.
	stride  []int     //stride is used for placement of values and grabing values.
	f32data []float32 //place to hold data in float32
	nhwc    bool
}

const maxdimsize = 4
const mindimsize = 4

//CreateTensorEx creates a tensor and copies data into Tensor if len(data)!=(t*Tensor)Volume() then error will return
//If len(dims) not ==4 an error will return!
func CreateTensorEx(dims []int, data []float32, NHWC bool) (*Tensor, error) {
	if len(data) != findvolume(dims) {
		return nil, errors.New("CreateTensorEx - len(data) not equal to vol of dims")
	}
	t, err := CreateTensor(dims, NHWC)
	if err != nil {
		return nil, err
	}
	for i := range data {
		t.f32data[i] = data[i]
	}
	return t, nil
}

//Volume returns volume of tensor (num of elements)
func (t *Tensor) Volume() int {
	return findvolume(t.dims)
}

//AddAll adds val to all elements in t
func (t *Tensor) AddAll(val float32) {
	for i := range t.f32data {
		t.f32data[i] += val
	}
}

//MultAll multiplies val to all elements in t
func (t *Tensor) MultAll(val float32) {

	for i := range t.f32data {
		t.f32data[i] *= val
	}
}

//Average returns the average ot t
func (t *Tensor) Average() float32 {
	var adder float32
	for i := range t.f32data {
		adder += t.f32data[i]
	}
	return adder / (float32)(len(t.f32data))
}

//Add does a t[i]=t[i]*beta + A[i]*alpha1 +B[i]*alpha2
func (t *Tensor) Add(A, B *Tensor, alpha1, alpha2, beta float32) error {
	if len(t.f32data) != len(A.f32data) || len(t.f32data) != len(B.f32data) {
		return errors.New("(t *Tensor)Add length of t, A, B data not equal")
	}
	if beta == 0 {
		for i := range t.f32data {
			t.f32data[i] = (A.f32data[i] * alpha1) + (B.f32data[i] * alpha2)
		}
	}
	for i := range t.f32data {
		t.f32data[i] = t.f32data[i]*beta + (A.f32data[i] * alpha1) + (B.f32data[i] * alpha2)
	}
	return nil
}

//Mult does a t[i]=t[i]*beta + A[i]*alpha1 *B[i]*alpha2
func (t *Tensor) Mult(A, B *Tensor, alpha1, alpha2, beta float32) error {
	if len(t.f32data) != len(A.f32data) || len(t.f32data) != len(B.f32data) {
		return errors.New("(t *Tensor)Mult length of t, A, B data not equal")
	}
	for i := range t.f32data {
		t.f32data[i] = t.f32data[i]*beta + A.f32data[i]*alpha1*B.f32data[i]*alpha2
	}
	return nil
}

//Div does a t[i]=t[i]*beta + A[i]*alpha1 / B[i]*alpha2
func (t *Tensor) Div(A, B *Tensor, alpha1, alpha2, beta float32) error {
	if len(t.f32data) != len(A.f32data) || len(t.f32data) != len(B.f32data) {
		return errors.New("(t *Tensor)Div length of t, A, B data not equal")
	}
	if alpha2 == 0 {
		return errors.New("(t *Tensor) Div : alpha2 == 0  cant devide by zero")
	}
	for i := range t.f32data {
		t.f32data[i] = t.f32data[i]*beta + A.f32data[i]*alpha1/B.f32data[i]*alpha2
	}
	return nil
}

//Dims returns a copy of tensor dims
func (t *Tensor) Dims() (dims []int) {
	dims = make([]int, len(t.dims))
	copy(dims, t.dims)
	return dims
}

//Stride returns a copy of tensor stride
func (t *Tensor) Stride() (stride []int) {
	stride = make([]int, len(t.stride))
	copy(stride, t.stride)
	return stride
}

//CreateRandomizedWeightsTensor creates a tensor with randomized weights.
func CreateRandomizedWeightsTensor(wdims, xdims []int, NHWC bool) (*Tensor, error) {
	t, err := CreateTensor(wdims, NHWC)
	if err != nil {
		return nil, err
	}
	fanin := float64(findvolume(xdims[1:]))
	for i := range t.f32data {
		t.f32data[i] = (float32)(gauassian(0, 2, fanin))
	}
	return t, nil
}

//CreateTensor creates a tensor according to the values passed. If len(dims) not ==4 an error will return!
//f64 is a place holder only thing available is float32
func CreateTensor(dims []int, NHWC bool) (*Tensor, error) {
	if len(dims) < 4 || len(dims) > 4 {
		return nil, errors.New("Not a valid length of dims")
	}
	//fmt.Println(dims)
	return &Tensor{
		dims:    dims,
		stride:  findstride(dims),
		f32data: make([]float32, findvolume(dims)),
		nhwc:    NHWC,
	}, nil
}

func (t *Tensor) Set(value float32, dimlocation []int) {
	var location int
	for i := range dimlocation {
		location += dimlocation[i] * t.stride[i]
	}
	t.f32data[location] = value
}
func (t *Tensor) Get(dimlcoation []int) (value float32) {
	var location int
	for i := range dimlcoation {
		location += dimlcoation[i] * t.stride[i]
	}
	value = t.f32data[location]
	return value
}

func gauassian(mean, std, ninputelements float64) float64 {

	//Polar method
	var x, y, z float64
	for z >= 1 || z == 0 {
		x = (2 * rand.Float64()) - float64(1)
		y = (2 * rand.Float64()) - float64(1)
		z = x*x + y*y
	}
	return (mean + std*float64(x*math.Sqrt(-2*math.Log(z)/z))) * (math.Sqrt((2.0) / (ninputelements)))
}

//ZeroClone returns a zeroed out clone of t
func (t *Tensor) ZeroClone() (*Tensor, error) {
	return CreateTensor(t.dims, t.nhwc)
}

//SetAll sets all the elments in t to value.
func (t *Tensor) SetAll(value float32) {
	for i := range t.f32data {
		t.f32data[i] = value
	}
}

//LoadFromSlice will load from values into the tensor. It starts at zero til the length of values.
//If values is longer than the volume of the tensor an error will return.
func (t *Tensor) LoadFromSlice(values []float32) (err error) {
	if len(values) > len(t.f32data) {
		return errors.New("values slice larger than tensor volume")
	}
	for i := range values {
		t.f32data[i] = values[i]
	}
	return nil
}
func findrecursivelocation(dims, stride []int, val int) (location int) {
	if len(dims) == 1 {
		val += dims[0] * stride[0]
		return val

	}
	val += dims[0] * stride[0]
	return findrecursivelocation(dims[1:], stride[1:], val)
}
func comparedims(xdims, ydims []int) bool {
	if len(xdims) != len(ydims) {
		return false
	}
	for i := range xdims {
		if xdims[i] != ydims[i] {
			return false
		}

	}
	return true
}

//findvolume will save us a lot of trouble.
func findvolume(dims []int) (vol int) {
	vol = 1
	for i := range dims {
		vol *= dims[i]
	}
	return vol
}
func findstride(dims []int) (stride []int) {
	stride = make([]int, len(dims))
	strider := int(1)
	for i := len(dims) - 1; i >= 0; i-- {
		stride[i] = strider
		strider *= dims[i]
	}
	return stride
}
