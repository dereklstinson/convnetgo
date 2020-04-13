package dnn

import (
	"errors"
)

//Relu holds the methods to do Relu activation
type Relu struct {
	ceiling float32
}

//CreateRelu will create the relu function if ceiling <= 0 then
//there won't be a ceiling
func CreateRelu(ceiling float32) (l *Relu) {
	l = new(Relu)
	l.ceiling = ceiling
	return l
}

//Set sets the ceiling
func (r *Relu) Set(ceiling float32) {
	r.ceiling = ceiling
}

//Get gets the ceiling
func (r *Relu) Get() (ceiling float32) {
	ceiling = r.ceiling
	return ceiling
}

//Forward does the forward operation
func (r *Relu) Forward(x, y *Tensor) (err error) {
	if findvolume(x.dims) != findvolume(y.dims) {
		return errors.New("(r *Relu)Forward  findvolume(x.dims)!=findvolume(y.dims)")
	}
	if r.ceiling <= 0 {
		for i := range y.f32data {
			if x.f32data[i] < 0 {
				y.f32data[i] = 0
			} else {
				y.f32data[i] = x.f32data[i]
			}
		}
		return nil
	}
	for i := range y.f32data {
		if x.f32data[i] < 0 {
			y.f32data[i] = 0
		} else if x.f32data[i] > r.ceiling {
			y.f32data[i] = r.ceiling
		} else {
			y.f32data[i] = x.f32data[i]
		}
	}
	return nil
}

//Backward does the Backward operation
func (r *Relu) Backward(x, dx, dy *Tensor) (err error) {
	if findvolume(dx.dims) != findvolume(dy.dims) || findvolume(dx.dims) != findvolume(x.dims) {
		return errors.New("(r *Relu)Backward  findvolume(dx.dims) != findvolume(dy.dims) ||findvolume(dx.dims)!=findvolume(x.dims)")
	}

	for i := range dy.f32data {
		if x.f32data[i] < 0 {
			dx.f32data[i] = 0
		} else {
			dx.f32data[i] = dy.f32data[i]
		}
	}
	return nil

}

//LeakyRelu is a struct that holds the neg and pos coef
type LeakyRelu struct {
	negcoef, poscoef float32
}

//CreateLeakyRelu creates a leaky relu
func CreateLeakyRelu(negcoef, poscoef float32) (l *LeakyRelu, err error) {
	if negcoef == poscoef {
		return nil, errors.New("CreateLeakyRelu() negcoef==poscoef")
	}
	l = new(LeakyRelu)
	l.negcoef = negcoef
	l.poscoef = poscoef
	return l, nil
}

//Set sets the coefs
func (l *LeakyRelu) Set(negcoef, poscoef float32) (err error) {
	if negcoef == poscoef {
		return errors.New("CreateLeakyRelu() negcoef==poscoef")
	}
	l.poscoef = poscoef
	l.negcoef = negcoef
	return nil
}

//Get gets the coefs
func (l *LeakyRelu) Get() (negcoef, poscoef float32) {

	poscoef = l.poscoef
	negcoef = l.negcoef
	return negcoef, poscoef
}

//Forward does the leaky relu activation
func (l *LeakyRelu) Forward(x, y *Tensor) (err error) {

	if len(x.f32data) != len(y.f32data) {
		return errors.New("LeakyReluForward() Volume of x != Volume of y")
	}

	for i := range x.f32data {
		if x.f32data[i] < 0 {
			y.f32data[i] = x.f32data[i] * l.negcoef
		} else {
			y.f32data[i] = x.f32data[i] * l.poscoef
		}

	}
	return nil
}

//Backward does the backward relu activation
func (l *LeakyRelu) Backward(x, dx, dy *Tensor) (err error) {
	if len(x.f32data) != len(dy.f32data) || len(x.f32data) != len(dx.f32data) {
		return errors.New("LeakyReluForward()  Volume of x != Volume of dy ||  Volume of x != Volume of dx")
	}

	for i := range x.f32data {
		if x.f32data[i] < 0 {
			dx.f32data[i] = dy.f32data[i] * l.negcoef
		} else {
			dx.f32data[i] = dy.f32data[i] * l.poscoef
		}

	}
	return nil
}
