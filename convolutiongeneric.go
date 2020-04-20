package convnetgo

import "sync"

func (c *Convolution) forward4dgeneric(x, w, wb, y *Tensor, alpha, beta float32) {
	//formated the code to look like nchw where n is batch for input tensors

	xdims := make([]int, len(x.dims))
	xstride := make([]int, len(x.stride))
	ydims := make([]int, len(y.dims))
	ystride := make([]int, len(y.stride))
	wdims := make([]int, len(w.dims))
	wstride := make([]int, len(w.stride))
	copy(xdims, x.dims)
	copy(xstride, x.stride)
	copy(ydims, y.dims)
	copy(ystride, y.stride)
	copy(wdims, w.dims)
	copy(wstride, w.stride)
	if !x.nhwc {
		wdims[len(w.dims)-1] = w.dims[1]
		wstride[len(w.stride)-1] = w.stride[1]
		ydims[len(y.dims)-1] = y.dims[1]
		ystride[len(y.stride)-1] = y.stride[1]
		xdims[len(x.dims)-1] = x.dims[1]
		xstride[len(x.stride)-1] = x.stride[1]
		for i := 2; i < len(xdims); i++ {
			wdims[i-1] = w.dims[i]
			wstride[i-1] = w.stride[i]
			xdims[i-1] = x.dims[i]
			xstride[i-1] = x.stride[i]
			ydims[i-1] = y.dims[i]
			ystride[i-1] = y.stride[i]
		}
	}

	var wg sync.WaitGroup
	for yn := 0; yn < xdims[0]; yn++ { //x and y output batch
		wg.Add(1)
		go func(yn int) {
			sh := -c.padding[0]
			for yh := 0; yh < ydims[1]; yh, sh = yh+1, sh+c.stride[0] { //output dims1
				sw := -c.padding[1]
				for yw := 0; yw < ydims[2]; yw, sw = yw+1, sw+c.stride[1] { //output dim2
					for yc := 0; yc < ydims[3]; yc++ { //wn == yc so the number of feature maps equals the size of the output channel.
						dh := c.dilation[0]
						dw := c.dilation[1]
						xn := yn
						wn := yc
						var adder float32
						var xh int
						for wh := 0; wh < wdims[1]; wh++ { //w input feature maps for x and w
							xh = sh + (wh * dh)
							if xh >= 0 && xh < xdims[1] {
								var xw int
								for ww := 0; ww < wdims[2]; ww++ { //w dim
									xw = sw + (ww * dw)
									if xw >= 0 && xw < xdims[2] {
										for wc := 0; wc < wdims[3]; wc++ {
											adder += x.f32data[(xstride[0]*xn)+(xstride[1]*xh)+(xstride[2]*xw)+xstride[3]*wc] * //wc and xc are the same
												w.f32data[(wstride[0]*wn)+(wstride[1]*wh)+(wstride[2]*ww)+(wstride[3]*wc)]
										}
									}
								}
							}
						}
						adder += wb.f32data[wn]             //wn is the output channel chich the bias gets add into it
						y.Set(adder, []int{yn, yh, yw, yc}) //Set the output channel
					}
				}
			}
			wg.Done()
		}(yn)
	}
	wg.Wait()
}

func (c *Convolution) backwardfiltergeneric(x, dw, dwb, dy *Tensor) {
	var wg sync.WaitGroup
	var mux sync.RWMutex

	xdims := make([]int, len(x.dims))
	xstride := make([]int, len(x.stride))
	dydims := make([]int, len(dy.dims))
	dystride := make([]int, len(dy.stride))
	dwdims := make([]int, len(dw.dims))
	dwstride := make([]int, len(dw.stride))
	copy(xdims, x.dims)
	copy(xstride, x.stride)
	copy(dydims, dy.dims)
	copy(dystride, dy.stride)
	copy(dwdims, dw.dims)
	copy(dwstride, dw.stride)
	if !x.nhwc {
		dwdims[len(dw.dims)-1] = dw.dims[1]
		dwstride[len(dw.stride)-1] = dw.stride[1]
		dydims[len(dy.dims)-1] = dy.dims[1]
		dystride[len(dy.stride)-1] = dy.stride[1]
		xdims[len(x.dims)-1] = x.dims[1]
		xstride[len(x.stride)-1] = x.stride[1]
		for i := 2; i < len(xdims); i++ {
			dwdims[i-1] = dw.dims[i]
			dwstride[i-1] = dw.stride[i]
			xdims[i-1] = x.dims[i]
			xstride[i-1] = x.stride[i]
			dydims[i-1] = dy.dims[i]
			dystride[i-1] = dy.stride[i]
		}
	}

	for yn := 0; yn < xdims[0]; yn++ { //x and y output batch
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
			for yh := 0; yh < dydims[1]; yh, sh = yh+1, sh+c.stride[0] { //output dims1
				sw := -c.padding[1]
				for yw := 0; yw < dydims[2]; yw, sw = yw+1, sw+c.stride[1] { //output dim2
					for yc := 0; yc < dydims[3]; yc++ { //wn == yc so the number of feature maps equals the size of the output channel.
						dilh := c.dilation[0]
						dilw := c.dilation[1]
						xn := yn
						wn := yc
						var xh int
						var grad = dy.Get([]int{yn, yh, yw, yc})
						for wh := 0; wh < dwdims[1]; wh++ { //w input feature maps for x and w
							xh = sh + (wh * dilh)
							if xh >= 0 && xh < xdims[1] {
								var xw int
								for ww := 0; ww < dwdims[2]; ww++ { //w dim
									xw = sw + (ww * dilw)
									if xw >= 0 && xw < xdims[2] {
										for wc := 0; wc < dwdims[3]; wc++ {
											dwzclone.f32data[(dwstride[0]*wn)+(dwstride[1]*wh)+(dwstride[2]*ww)+(dwstride[3]*wc)] +=
												grad * x.f32data[(xstride[0]*xn)+(xstride[1]*xh)+(xstride[2]*xw)+(xstride[3]*wc)]
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
func (c *Convolution) backwarddatatgeneric(dx, w, dy *Tensor) {

	dxdims := make([]int, len(dx.dims))
	dxstride := make([]int, len(dx.stride))
	dydims := make([]int, len(dy.dims))
	dystride := make([]int, len(dy.stride))
	wdims := make([]int, len(w.dims))
	wstride := make([]int, len(w.stride))
	copy(dxdims, dx.dims)
	copy(dxstride, dx.stride)
	copy(dydims, dy.dims)
	copy(dystride, dy.stride)
	copy(wdims, w.dims)
	copy(wstride, w.stride)
	if !dx.nhwc {
		wdims[len(w.dims)-1] = w.dims[1]
		wstride[len(w.stride)-1] = w.stride[1]
		dydims[len(dy.dims)-1] = dy.dims[1]
		dystride[len(dy.stride)-1] = dy.stride[1]
		dxdims[len(dx.dims)-1] = dx.dims[1]
		dxstride[len(dx.stride)-1] = dx.stride[1]
		for i := 2; i < len(dxdims); i++ {
			wdims[i-1] = w.dims[i]
			wstride[i-1] = w.stride[i]
			dxdims[i-1] = dx.dims[i]
			dxstride[i-1] = dx.stride[i]
			dydims[i-1] = dy.dims[i]
			dystride[i-1] = dy.stride[i]
		}
	}

	var wg sync.WaitGroup
	for yn := 0; yn < dydims[0]; yn++ { //x and y output batch
		wg.Add(1)
		go func(yn int) {
			sh := -c.padding[0]
			for yh := 0; yh < dydims[1]; yh, sh = yh+1, sh+c.stride[0] { //output dims1
				sw := -c.padding[1]
				for yw := 0; yw < dydims[2]; yw, sw = yw+1, sw+c.stride[1] { //output dim2
					for yc := 0; yc < dydims[3]; yc++ { //wn == yc so the number of feature maps equals the size of the output channel.
						dilh := c.dilation[0]
						dilw := c.dilation[1]
						xn := yn
						wn := yc
						var grad = dy.Get([]int{yn, yh, yw, yc})
						var xh int
						for wh := 0; wh < wdims[1]; wh++ { //w input feature maps for x and w
							xh = sh + (wh * dilh)

							if xh >= 0 && xh < dxdims[1] {
								var xw int
								for ww := 0; ww < wdims[2]; ww++ { //w dim
									xw = sw + (ww * dilw)

									if xw >= 0 && xw < dxdims[2] {

										for wc := 0; wc < wdims[3]; wc++ {

											dx.f32data[(dxstride[0]*xn)+(dxstride[1]*xh)+(dxstride[2]*xw)+(dxstride[3]*wc)] +=
												grad * w.f32data[(wstride[0]*wn)+(wstride[1]*wh)+(wstride[2]*ww)+(wstride[3]*wc)]
										}

									}

								}

							}
						}
					}
				}
			}
			wg.Done()
		}(yn)
	}
	wg.Wait()
}
