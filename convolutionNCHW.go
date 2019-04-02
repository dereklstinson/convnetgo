package dnn

import "sync"

func (c *Convolution) forwardNCHW4d(x, w, wb, y *Tensor) {
	var wg sync.WaitGroup
	var mux sync.RWMutex

	for y0 := 0; y0 < x.dims[0]; y0++ { //x and y utput batch
		for y1 := 0; y1 < y.dims[1]; y1++ { // output feature maps for w and y

			s0 := -c.padding[0]
			wg.Add(1)
			go func(y0, y1, s0 int) {
				var w0, x0, x1 int
				x0 = y0
				w0 = y1
				mux.RLock()
				for y2 := 0; y2 < y.dims[2]; y2, s0 = y2+1, s0+c.stride[0] { //output dims1
					s1 := -c.padding[1]
					for y3 := 0; y3 < y.dims[3]; y3, s1 = y3+1, s1+c.stride[1] { //output dim2
						var adder float32
						for w1 := 0; w1 < w.dims[1]; w1++ { //w input feature maps for x and w
							x1 = w1
							var x2 int
							d0 := c.dilation[0]
							d1 := c.dilation[1]
							for w2 := 0; w2 < w.dims[2]; w2++ { //w dim
								x2 = s0 + (w2 * d0)
								if x2 >= 0 && x2 < x.dims[2] {
									var x3 int
									for w3 := 0; w3 < w.dims[3]; w3++ { // w dim1
										x3 = s1 + (w3 * d1)

										if x3 >= 0 && x3 < x.dims[3] {
											adder += x.f32data[(x.stride[0]*x0)+(x.stride[1]*x1)+(x.stride[2]*x2)+x3] * w.f32data[(w.stride[0]*w0)+(w.stride[1]*w1)+(w.stride[2]*w2)+(w.stride[3]*w3)]
										}
									}

								}

							}
						}
						adder += wb.f32data[w0]
						y.Set(adder, []int{y0, y1, y2, y3})

					}
				}
				mux.RUnlock()
				wg.Done()
			}(y0, y1, s0)

		}

	}
	wg.Wait()
}

func (c *Convolution) forwardNCHW5dnew(x, w, wb, y *Tensor) {
	var wg sync.WaitGroup
	var mux sync.RWMutex
	for y0 := 0; y0 < x.dims[0]; y0++ { //x and y utput batch
		for y1 := 0; y1 < y.dims[1]; y1++ { // output feature maps for w and y
			s0 := -c.padding[0]
			wg.Add(1)
			go func(y0, y1, s0 int) {
				var w0, x0 int
				x0 = y0
				w0 = y1
				mux.RLock()
				for y2 := 0; y2 < y.dims[2]; y2, s0 = y2+1, s0+c.stride[0] { //output dims1
					s1 := -c.padding[1]
					for y3 := 0; y3 < y.dims[3]; y3, s1 = y3+1, s1+c.stride[1] { //output dim2
						s2 := -c.padding[2]
						for y4 := 0; y4 < y.dims[4]; y4, s2 = y4+1, s2+c.stride[2] {
							var adder float32
							d0 := c.dilation[0]
							d1 := c.dilation[1]
							d2 := c.dilation[2]
							for w1 := 0; w1 < w.dims[1]; w1++ { //w input feature maps for x and w
								x1 := w1
								for w2 := 0; w2 < w.dims[2]; w2++ { //w dim
									x2 := s0 + (w2 * d0)
									if x2 >= 0 && x2 < x.dims[2] {
										for w3 := 0; w3 < w.dims[3]; w3++ { // w dim1
											x3 := s1 + (w3 * d1)
											if x3 >= 0 && x3 < x.dims[3] {
												for w4 := 0; w4 < w.dims[4]; w4++ {
													x4 := s2 + (w4 * d2)
													if x4 >= 0 && x4 < x.dims[4] {
														adder += x.f32data[(x.stride[0]*x0)+(x.stride[1]*x1)+(x.stride[2]*x2)+(x.stride[3]*x3)+(x.stride[4]*x4)] * w.f32data[(w.stride[0]*w0)+(w.stride[1]*w1)+(w.stride[2]*w2)+(w.stride[3]*w3)+w.stride[4]*w4]
													}
												}

											}
										}

									}

								}
							}
							adder += wb.f32data[w0]
							y.Set(adder, []int{y0, y1, y2, y3, y4})
						}

					}
				}
				mux.RUnlock()
				wg.Done()
			}(y0, y1, s0)

		}
		wg.Wait()
	}
}

func (c *Convolution) forwardNCHW5d(x, w, wb, y *Tensor) {
	var wg sync.WaitGroup
	var mux sync.RWMutex

	for y0 := 0; y0 < y.dims[0]; y0++ { //x and y utput batch
		for y1 := 0; y1 < y.dims[1]; y1++ { // output feature maps for w and y
			s0 := -c.padding[0]
			wg.Add(1)
			go func(y0, y1, s0 int) {
				var w0, x0 int
				x0 = y0
				w0 = y1
				mux.RLock()
				for y2 := 0; y2 < y.dims[2]; y2, s0 = y2+1, s0+c.stride[0] {
					s1 := -c.padding[1]
					for y3 := 0; y3 < y.dims[3]; y3, s1 = y3+1, s1+c.stride[1] {
						s2 := -c.padding[2]
						for y4 := 0; y4 < y.dims[4]; y4, s2 = y4+1, s2+c.stride[2] {
							var adder float32
							for w1 := 0; w1 < w.dims[1]; w1++ { //w input feature maps for x and w
								x1 := w1
								for w2 := 0; w2 < w.dims[2]; w2++ {
									x2 := s0 + (w2 * c.dilation[0])
									if x2 >= 0 && x2 < x.dims[2] {
										for w3 := 0; w3 < w.dims[3]; w3++ {
											x3 := s1 + (w3 * c.dilation[1])
											if x3 >= 0 && x3 < x.dims[3] {
												for w4 := 0; w4 < w.dims[4]; w4++ {
													x4 := s2 + (w4 * c.dilation[2])
													if x4 >= 0 && x4 < x.dims[4] {

														adder += x.f32data[(x.stride[0]*x0)+
															(x.stride[1]*x1)+
															(x.stride[2]*x2)+
															(x.stride[3]*x3)+
															(x.stride[4]*x4)] *
															w.f32data[(w.stride[0]*w0)+
																(w.stride[1]*w1)+
																(w.stride[2]*w2)+
																(w.stride[3]*w3)+
																(w.stride[4]*w4)]

													}
												}
											}

										}
									}

								}

							}

							adder += wb.f32data[w0]
							y.Set(adder, []int{y0, y1, y2, y3, y4})

						}
					}
				}
				mux.RUnlock()
				wg.Done()
			}(y0, y1, s0)

		}

	}
	wg.Wait()
}

func (c *Convolution) forwardNCHW6d(x, w, wb, y *Tensor) {
	var wg sync.WaitGroup
	var mux sync.RWMutex

	for y0 := 0; y0 < x.dims[0]; y0++ { //x and y utput batch
		for y1 := 0; y1 < y.dims[1]; y1++ { // output feature maps for w and y

			s0 := -c.padding[0]
			wg.Add(1)
			go func(y0, y1, s0 int) {
				var w0, x0 int
				x0 = y0
				w0 = y1
				mux.RLock()
				for y2 := 0; y2 < y.dims[2]; y2, s0 = y2+1, s0+c.stride[0] {
					s1 := -c.padding[1]
					for y3 := 0; y3 < y.dims[3]; y3, s1 = y3+1, s1+c.stride[1] {
						s2 := -c.padding[2]
						for y4 := 0; y4 < y.dims[2]; y4, s2 = y4+1, s2+c.stride[0] {
							s3 := -c.padding[3]
							for y5 := 0; y4 < y.dims[3]; y5, s3 = y5+1, s3+c.stride[1] {
								var adder float32
								for w1 := 0; w1 < w.dims[1]; w1++ { //w input feature maps for x and w
									x1 := w1
									for w2 := 0; w2 < w.dims[2]; w2++ {
										x2 := s0 + (w2 * c.dilation[0])
										if x2 >= 0 && x2 < x.dims[2] {
											for w3 := 0; w3 < w.dims[3]; w3++ {
												x3 := s1 + (w3 * c.dilation[1])
												if x3 >= 0 && x3 < x.dims[3] {
													for w4 := 0; w4 < w.dims[4]; w4++ {
														x4 := s2 + (w4 * c.dilation[2])
														if x4 >= 0 && x4 < x.dims[4] {
															for w5 := 0; w5 < w.dims[5]; w5++ {
																x5 := s3 + (w5 * c.dilation[3])
																if x5 >= 0 && x5 < x.dims[5] {

																	adder += x.f32data[(x.stride[0]*x0)+
																		(x.stride[1]*x1)+
																		(x.stride[2]*x2)+
																		(x.stride[3]*x3)+
																		(x.stride[4]*x4)+
																		(x.stride[5]*x5)] *
																		w.f32data[(w.stride[0]*w0)+
																			(w.stride[1]*w1)+
																			(w.stride[2]*w2)+
																			(w.stride[3]*w3)+
																			(w.stride[4]*w4)+
																			(w.stride[5]*w5)]

																}
															}
														}

													}
												}

											}

										}
									}
								}
								adder += wb.f32data[w0]
								y.Set(adder, []int{y0, y1, y2, y3, y4, y5})

							}
						}

					}
				}
				mux.RUnlock()
				wg.Done()
			}(y0, y1, s0)

		}

	}
	wg.Wait()
}

func (c *Convolution) forwardNCHW7d(x, w, wb, y *Tensor) {

	var wg sync.WaitGroup
	var mux sync.RWMutex

	for y0 := 0; y0 < x.dims[0]; y0++ { //x and y utput batch
		for y1 := 0; y1 < y.dims[1]; y1++ { // output feature maps for w and y

			s0 := -c.padding[0]
			wg.Add(1)
			go func(y0, y1, s0 int) {
				var w0, x0, x1 int
				x0 = y0
				w0 = y1
				mux.RLock()
				for y2 := 0; y2 < y.dims[2]; y2, s0 = y2+1, s0+c.stride[0] {
					s1 := -c.padding[1]
					for y3 := 0; y3 < y.dims[3]; y3, s1 = y3+1, s1+c.stride[1] {
						s2 := -c.padding[2]
						for y4 := 0; y2 < y.dims[2]; y2, s0 = y2+1, s0+c.stride[0] {
							s3 := -c.padding[3]
							for y5 := 0; y3 < y.dims[3]; y3, s1 = y3+1, s1+c.stride[1] {
								s4 := -c.padding[4]
								for y6 := 0; y2 < y.dims[2]; y2, s0 = y2+1, s0+c.stride[0] {
									var adder float32
									for w1 := 0; w1 < w.dims[1]; w1++ { //w input feature maps for x and w
										x1 = w1
										for w2 := 0; w2 < w.dims[2]; w2++ {
											x2 := s0 + (w2 * c.dilation[0])
											if x2 >= 0 && x2 < x.dims[2] {
												for w3 := 0; w3 < w.dims[3]; w3++ {
													x3 := s1 + (w3 * c.dilation[1])
													if x3 >= 0 && x3 < x.dims[3] {
														for w4 := 0; w4 < w.dims[4]; w4++ {
															x4 := s2 + (w4 * c.dilation[2])
															if x4 >= 0 && x4 < x.dims[4] {
																for w5 := 0; w5 < w.dims[5]; w5++ {
																	x5 := s3 + (w5 * c.dilation[3])
																	if x5 >= 0 && x5 < x.dims[5] {
																		for w6 := 0; w6 < w.dims[6]; w6++ {
																			x6 := s4 + (w6 * c.dilation[4])
																			if x6 >= 0 && x6 < x.dims[6] {
																				adder += x.f32data[(x.stride[0]*x0)+
																					(x.stride[1]*x1)+
																					(x.stride[2]*x2)+
																					(x.stride[3]*x3)+
																					(x.stride[4]*x4)+
																					(x.stride[5]*x5)+
																					(x.stride[6]*x6)] *
																					w.f32data[(w.stride[0]*w0)+
																						(w.stride[1]*w1)+
																						(w.stride[2]*w2)+
																						(w.stride[3]*w3)+
																						(w.stride[4]*w4)+
																						(w.stride[5]*w5)+
																						(w.stride[6]*w6)]

																			}
																		}
																	}
																}

															}
														}

													}

												}
											}
										}
									}
									adder += wb.f32data[w0]
									y.Set(adder, []int{y0, y1, y2, y3, y4, y5, y6})

								}
							}
						}
					}
				}
				mux.RUnlock()
				wg.Done()
			}(y0, y1, s0)

		}

	}
	wg.Wait()
}

func (c *Convolution) forwardNCHW8d(x, w, wb, y *Tensor) {

	var wg sync.WaitGroup
	var mux sync.RWMutex

	for y0 := 0; y0 < x.dims[0]; y0++ { //x and y utput batch
		for y1 := 0; y1 < y.dims[1]; y1++ { // output feature maps for w and y

			s0 := -c.padding[0]
			wg.Add(1)
			go func(y0, y1, s0 int) {
				var w0, x0, x1 int
				x0 = y0
				w0 = y1
				mux.RLock()
				for y2 := 0; y2 < y.dims[2]; y2, s0 = y2+1, s0+c.stride[0] { //output dims1
					s1 := -c.padding[1]
					for y3 := 0; y3 < y.dims[3]; y3, s1 = y3+1, s1+c.stride[1] { //output dim2
						s2 := -c.padding[2]
						for y4 := 0; y2 < y.dims[2]; y2, s0 = y2+1, s0+c.stride[0] { //output dims1
							s3 := -c.padding[3]
							for y5 := 0; y3 < y.dims[3]; y3, s1 = y3+1, s1+c.stride[1] { //output dim2
								s4 := -c.padding[4]
								for y6 := 0; y2 < y.dims[2]; y2, s0 = y2+1, s0+c.stride[0] { //output dims1
									s5 := -c.padding[5]
									for y7 := 0; y3 < y.dims[3]; y3, s1 = y3+1, s1+c.stride[1] { //output dim2
										var adder float32
										for w1 := 0; w1 < w.dims[1]; w1++ { //w input feature maps for x and w
											x1 = w1
											for w2 := 0; w2 < w.dims[2]; w2++ { //w dim
												x2 := s0 + (w2 * c.dilation[0])
												if x2 >= 0 && x2 < x.dims[2] {
													for w3 := 0; w3 < w.dims[3]; w3++ { // w dim1
														x3 := s1 + (w3 * c.dilation[1])
														if x3 >= 0 && x3 < x.dims[3] {
															for w4 := 0; w4 < w.dims[4]; w4++ { //w dim
																x4 := s2 + (w4 * c.dilation[2])
																if x4 >= 0 && x4 < x.dims[4] {
																	for w5 := 0; w5 < w.dims[5]; w5++ { // w dim1
																		x5 := s3 + (w5 * c.dilation[3])
																		if x5 >= 0 && x5 < x.dims[5] {
																			for w6 := 0; w6 < w.dims[6]; w6++ {
																				x6 := s4 + (w6 * c.dilation[4])
																				if x6 >= 0 && x6 < x.dims[6] {
																					for w7 := 0; w7 < w.dims[7]; w7++ {
																						x7 := s5 + (w7 * c.dilation[5])
																						if x7 >= 0 && x7 < x.dims[7] {

																							adder += x.f32data[(x.stride[0]*x0)+
																								(x.stride[1]*x1)+
																								(x.stride[2]*x2)+
																								(x.stride[3]*x3)+
																								(x.stride[4]*x4)+
																								(x.stride[5]*x5)+
																								(x.stride[6]*x6)+
																								(x.stride[7]*x7)] *
																								w.f32data[(w.stride[0]*w0)+
																									(w.stride[1]*w1)+
																									(w.stride[2]*w2)+
																									(w.stride[3]*w3)+
																									(w.stride[4]*w4)+
																									(w.stride[5]*w5)+
																									(w.stride[6]*w6)+
																									(w.stride[7]*w7)]

																						}
																					}
																				}
																			}
																		}
																	}
																}
															}
														}
													}
												}
											}
										}
										adder += wb.f32data[w0]
										y.Set(adder, []int{y0, y1, y2, y3, y4, y5, y6, y7})

									}
								}
							}
						}
					}
				}
				mux.RUnlock()
				wg.Done()
			}(y0, y1, s0)

		}

	}
	wg.Wait()
}
