package com.tencent.bi.cf.optimization.gradient.sparse;

import com.tencent.bi.cf.optimization.Loss;

import cern.colt.matrix.DoubleFactory1D;
import cern.colt.matrix.DoubleMatrix1D;

/**
 * Square Loss
 * @author tigerzhong
 *
 */
public class LogSquareLoss implements Loss {

	private static final long serialVersionUID = 1L;

	@Override
	public double getValue(DoubleMatrix1D u, DoubleMatrix1D v, double r) {
		double p = u.zDotProduct(v);
		return Math.log((p-r)*(p-r));
	}

	@Override
	public DoubleMatrix1D getGradient(DoubleMatrix1D u, DoubleMatrix1D v, double r, double lambda) {
		/*
		 * g = v/(uv-r) + lambda*u
		 */
		int d = u.size();
		DoubleMatrix1D g = DoubleFactory1D.dense.make(d);
		double e = u.zDotProduct(v) - r;
		for(int i=0;i<d;i++){
			double tg = v.get(i)/e + lambda*u.get(i);
			g.set(i, tg);
		}
		return g;
	}

}
