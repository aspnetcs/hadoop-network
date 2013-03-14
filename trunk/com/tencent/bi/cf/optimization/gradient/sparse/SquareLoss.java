package com.tencent.bi.cf.optimization.gradient.sparse;

import com.tencent.bi.cf.optimization.Loss;

import cern.colt.matrix.DoubleFactory1D;
import cern.colt.matrix.DoubleMatrix1D;

/**
 * Square Loss
 * @author tigerzhong
 *
 */
public class SquareLoss implements Loss {

	private static final long serialVersionUID = 1L;

	@Override
	public double getValue(DoubleMatrix1D u, DoubleMatrix1D v, double r) {
	    // (uv-r)^2
	    // @weixue As far as I know, most implementations use (r-uv)^2. Logically, there is no difference
	    // between these 2 approaches. Is there any implementation specific consideration behind this?
		double p = u.zDotProduct(v);
		return (p-r)*(p-r);
	}

	@Override
	public DoubleMatrix1D getGradient(DoubleMatrix1D u, DoubleMatrix1D v, double r, double lambda) {
		/*
		 * g = (uv-r)*v + lambda*u
		 * @weixue coefficient 2 has been rolled into learning rate, so is missing here.
		 */
		int d = u.size();
		DoubleMatrix1D g = DoubleFactory1D.dense.make(d);
		double e = u.zDotProduct(v) - r;
		for(int i=0;i<d;i++){
			double tg = e*v.get(i) + lambda*u.get(i);
			g.set(i, tg);
		}
		return g;
	}

}
