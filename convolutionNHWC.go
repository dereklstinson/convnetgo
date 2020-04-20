package convnetgo

import (
	"sync"
)

func (c *Convolution) forwardNHWC4d(x, w, wb, y *Tensor, alpha, beta float32) {
	//formated the code to look like nchw where n is batch for input tensors

	//n is number of feature maps for weights
	//n is batch for input and output tensors
	//h is height
	//w is width
	//c is channel
	var wg sync.WaitGroup

	for yn := 0; yn < x.dims[0]; yn++ { //x and y output batch
		wg.Add(1)
		go func(yn int) {
			sh := -c.padding[0]
			for yh := 0; yh < y.dims[1]; yh, sh = yh+1, sh+c.stride[0] { //output dims1
				sw := -c.padding[1]
				for yw := 0; yw < y.dims[2]; yw, sw = yw+1, sw+c.stride[1] { //output dim2
					for yc := 0; yc < y.dims[3]; yc++ { //wn == yc so the number of feature maps equals the size of the output channel.
						//	wg.Add(1)
						//	go func(yn, yh, yw, yc, sh, sw int) {
						dh := c.dilation[0]
						dw := c.dilation[1]
						xn := yn
						wn := yc
						var adder float32
						var xh int
						for wh := 0; wh < w.dims[1]; wh++ { //w input feature maps for x and w
							xh = sh + (wh * dh)

							if xh >= 0 && xh < x.dims[1] {
								var xw int
								for ww := 0; ww < w.dims[2]; ww++ { //w dim
									xw = sw + (ww * dw)

									if xw >= 0 && xw < x.dims[2] {

										for wc := 0; wc < w.dims[3]; wc++ {
											adder += x.f32data[(x.stride[0]*xn)+(x.stride[1]*xh)+(x.stride[2]*xw)+(x.stride[3]*wc)] * //wc and xc are the same
												w.f32data[(w.stride[0]*wn)+(w.stride[1]*wh)+(w.stride[2]*ww)+(w.stride[3]*wc)]
										}
									}

								}

							}
						}
						adder += wb.f32data[wn] //wn is the output channel chich the bias gets add into it
						//	previous := y.Get([]int{yn, yh, yw, yc})
						//	previous = beta*previous + alpha*adder
						y.Set(adder, []int{yn, yh, yw, yc}) //Set the output channel
						//		wg.Done()
						//	}(yn, yh, yw, yc, sh, sw)

					}
				}

			}
			wg.Done()
		}(yn)

	}
	wg.Wait()
}
func (c *Convolution) backwardfilterNHWC4d(x, dw, dwb, dy *Tensor) {
	var wg sync.WaitGroup
	var mux sync.RWMutex

	for yn := 0; yn < x.dims[0]; yn++ { //x and y output batch
		wg.Add(1)
		go func(yn int) {
			dwzclone, err := dw.ZeroClone()
			if err != nil {
				panic(err)
			}
			dbclone, err := dwb.ZeroClone()
			if err != nil {
				panic(err)
			}
			sh := -c.padding[0]
			for yh := 0; yh < dy.dims[1]; yh, sh = yh+1, sh+c.stride[0] { //output dims1
				sw := -c.padding[1]
				for yw := 0; yw < dy.dims[2]; yw, sw = yw+1, sw+c.stride[1] { //output dim2
					for yc := 0; yc < dy.dims[3]; yc++ { //wn == yc so the number of feature maps equals the size of the output channel.
						dilh := c.dilation[0]
						dilw := c.dilation[1]
						xn := yn
						wn := yc
						var xh int
						var grad = dy.Get([]int{yn, yh, yw, yc})
						for wh := 0; wh < dw.dims[1]; wh++ { //w input feature maps for x and w
							xh = sh + (wh * dilh)
							if xh >= 0 && xh < x.dims[1] {
								var xw int
								for ww := 0; ww < dw.dims[2]; ww++ { //w dim
									xw = sw + (ww * dilw)
									if xw >= 0 && xw < x.dims[2] {
										for wc := 0; wc < dw.dims[3]; wc++ {
											dwzclone.f32data[(dw.stride[0]*wn)+(dw.stride[1]*wh)+(dw.stride[2]*ww)+(dw.stride[3]*wc)] +=
												grad * x.f32data[(x.stride[0]*xn)+(x.stride[1]*xh)+(x.stride[2]*xw)+(x.stride[3]*wc)]
										}

									}

								}

							}
						}

						dbclone.f32data[wn] += grad //wn is the output channel chich the bias gets add into it

					}
				}

			}
			mux.Lock()
			dw.Add(dw, dwzclone, 1, 1, 0)
			dwb.Add(dwb, dbclone, 1, 1, 0)
			mux.Unlock()
			wg.Done()
		}(yn)

	}
	wg.Wait()
}
func (c *Convolution) backwarddatatNHWC4d(dx, w, dy *Tensor) {

	//formated the code to look like nchw where n is batch for input tensors
	//n is number of feature maps for weights
	//n is batch for input and output tensors
	//h is height
	//w is width
	//c is channel
	var wg sync.WaitGroup
	//var mux sync.RWMutex

	for yn := 0; yn < dy.dims[0]; yn++ { //x and y output batch
		wg.Add(1)
		go func(yn int) {
			dilh := c.dilation[0]
			dilw := c.dilation[1]
			sh := -c.padding[0]
			for yh := 0; yh < dy.dims[1]; yh, sh = yh+1, sh+c.stride[0] { //output dims1
				sw := -c.padding[1]
				for yw := 0; yw < dy.dims[2]; yw, sw = yw+1, sw+c.stride[1] { //output dim2
					for yc := 0; yc < dy.dims[3]; yc++ { //wn == yc so the number of feature maps equals the size of the output channel.
						//	wg.Add(1)
						//	go func(yn, yh, yw, yc, sh, sw int) {
						xn := yn
						wn := yc
						var grad = dy.Get([]int{yn, yh, yw, yc})
						var xh int
						for wh := 0; wh < w.dims[1]; wh++ { //w input feature maps for x and w
							xh = sh + (wh * dilh)

							if xh >= 0 && xh < dx.dims[1] {
								var xw int
								for ww := 0; ww < w.dims[2]; ww++ { //w dim
									xw = sw + (ww * dilw)

									if xw >= 0 && xw < dx.dims[2] {
										//	mux.Lock()
										for wc := 0; wc < w.dims[3]; wc++ {
											dx.f32data[(dx.stride[0]*xn)+(dx.stride[1]*xh)+(dx.stride[2]*xw)+(dx.stride[3]*wc)] +=
												grad * w.f32data[(w.stride[0]*wn)+(w.stride[1]*wh)+(w.stride[2]*ww)+(w.stride[3]*wc)]
										}
										//	mux.Unlock()
									}

								}

							}
						}
						//	wg.Done()
						//}(yn, yh, yw, yc, sh, sw)
					}
				}

			}
			wg.Done()
		}(yn)

	}
	wg.Wait()

}
