package convnetgo

import "sync"

func (c *Convolution) forwardNCHW4d(x, w, wb, y *Tensor, alpha, beta float32) {
	var wg sync.WaitGroup
	//	var mux sync.RWMutex

	for yn := 0; yn < x.dims[0]; yn++ { //x and y output batch
		wg.Add(1)
		go func(yn int) {
			for yc := 0; yc < y.dims[1]; yc++ { // output feature maps for w and y

				sh := -c.padding[0]

				var wn, xn, xc int
				xn = yn
				wn = yc

				for yh := 0; yh < y.dims[2]; yh, sh = yh+1, sh+c.stride[0] { //output dimsw
					sw := -c.padding[1]
					for yw := 0; yw < y.dims[3]; yw, sw = yw+1, sw+c.stride[1] { //output dim2

						var adder float32
						for wc := 0; wc < w.dims[1]; wc++ { //w input feature maps for x and w
							xc = wc
							var xh int
							dilh := c.dilation[0]
							dilw := c.dilation[1]
							for wh := 0; wh < w.dims[2]; wh++ { //w dim
								xh = sh + (wh * dilh)
								if xh >= 0 && xh < x.dims[2] {
									var xw int
									for ww := 0; ww < w.dims[3]; ww++ { // w dim1
										xw = sw + (ww * dilw)

										if xw >= 0 && xw < x.dims[3] {
											adder += x.f32data[(x.stride[0]*xn)+(x.stride[1]*xc)+(x.stride[2]*xh)+x.stride[3]*xw] * w.f32data[(w.stride[0]*wn)+(w.stride[1]*wc)+(w.stride[2]*wh)+(w.stride[3]*ww)]
										}
									}

								}

							}
						}
						adder += wb.f32data[wn]
						y.Set(adder, []int{yn, yc, yh, yw})

					}
				}
				//	mux.RUnlock()

			}
			wg.Done()
		}(yn)
	}
	wg.Wait()
}
func (c *Convolution) backwardfilterNCHW4d(x, dw, dwb, dy *Tensor) {
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

			for yc := 0; yc < dy.dims[1]; yc++ { // output feature maps for w and y
				sh := -c.padding[0]
				for yh := 0; yh < dy.dims[2]; yh, sh = yh+1, sh+c.stride[0] { //output dimsw
					sw := -c.padding[1]
					for yw := 0; yw < dy.dims[3]; yw, sw = yw+1, sw+c.stride[1] { //output dim2

						xn := yn
						wn := yc
						dilh := c.dilation[0]
						dilw := c.dilation[1]
						var xc int
						var grad = dy.Get([]int{yn, yc, yh, yw})
						for wc := 0; wc < dwb.dims[1]; wc++ { //w input feature maps for x and w
							xc = wc
							var xh int
							for wh := 0; wh < dwb.dims[2]; wh++ { //w dim
								xh = sh + (wh * dilh)
								if xh >= 0 && xh < x.dims[2] {
									//mux.RLock()
									var xw int
									for ww := 0; ww < dwb.dims[3]; ww++ { // w dim1
										xw = sw + (ww * dilw)

										if xw >= 0 && xw < x.dims[3] {
											dwzclone.f32data[(dw.stride[0]*wn)+(dw.stride[1]*wc)+(dw.stride[2]*wh)+(dw.stride[3]*ww)] +=
												grad * x.f32data[(x.stride[0]*xn)+(x.stride[1]*xc)+(x.stride[2]*xh)+(x.stride[3]*xw)]
										}
									}
									//	mux.RUnlock()
								}

							}
						}
						//	mux.RLock()
						dbclone.f32data[wn] += grad
						//	mux.RUnlock()

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
func (c *Convolution) backwarddataNCHW4d(dx, w, dy *Tensor) {
	var wg sync.WaitGroup

	for yn := 0; yn < dx.dims[0]; yn++ { //x and y output batch
		wg.Add(1)
		go func(yn int) {
			for yc := 0; yc < dy.dims[1]; yc++ { // output feature maps for w and y
				sh := -c.padding[0]
				for yh := 0; yh < dy.dims[2]; yh, sh = yh+1, sh+c.stride[0] { //output dimsw
					sw := -c.padding[1]
					for yw := 0; yw < dy.dims[3]; yw, sw = yw+1, sw+c.stride[1] { //output dim2
						xn := yn
						wn := yc
						dilh := c.dilation[0]
						dilw := c.dilation[1]
						var xc int
						var grad = dy.Get([]int{yn, yc, yh, yw})
						for wc := 0; wc < w.dims[1]; wc++ { //w input feature maps for x and w
							xc = wc
							var xh int
							for wh := 0; wh < w.dims[2]; wh++ { //w dim
								xh = sh + (wh * dilh)
								if xh >= 0 && xh < dx.dims[2] {
									var xw int
									for ww := 0; ww < w.dims[3]; ww++ { // w dim1
										xw = sw + (ww * dilw)
										if xw >= 0 && xw < dx.dims[3] {
											dx.f32data[(dx.stride[0]*xn)+(dx.stride[1]*xc)+(dx.stride[2]*xh)+(dx.stride[3]*xw)] += grad * w.f32data[(w.stride[0]*wn)+(w.stride[1]*wc)+(w.stride[2]*wh)+(w.stride[3]*ww)]
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
