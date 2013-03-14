package com.tencent.bi.cf.optimization.gradient.bias;

import com.tencent.bi.cf.optimization.Loss;

import cern.colt.matrix.DoubleFactory1D;
import cern.colt.matrix.DoubleMatrix1D;

/**
 * Square Loss with Bias
 * @author tigerzhong
 *
 */
public class SquareBiasLoss implements Loss {

	private static final long serialVersionUID = 1L;

	public double getValue(DoubleMatrix1D u, DoubleMatrix1D v, double uBias,
			double vBias, double r) {
		double p = u.zDotProduct(v) + uBias + vBias;
		return (p-r)*(p-r);
	}

	public DoubleMatrix1D getGradient(DoubleMatrix1D u, DoubleMatrix1D v,
			double uBias, double vBias, double r, double lambda) {
		/*
		 * g = (uv-r)*v + lambda*u
		 */
		int d = u.size();
		DoubleMatrix1D g = DoubleFactory1D.dense.make(d);
		double e = u.zDotProduct(v) + uBias + vBias - r;
		for(int i=0;i<d;i++){
			double tg = e*v.get(i) + lambda*u.get(i);
			g.set(i, tg);
		}
		return g;
	}

	public double getGradientBias(DoubleMatrix1D u, DoubleMatrix1D v,
			double uBias, double vBias, double r, double lambda) {
		double e = u.zDotProduct(v) + uBias + vBias - r;
		return e + lambda*uBias;
	}

//////////////////////////////////////////////////////////////////////////////////////////////////////////	
	@Deprecated
	public double getValue(DoubleMatrix1D u, DoubleMatrix1D v, double r) {
		//Ignore!!
		return 0;
	}

	@Deprecated
	public DoubleMatrix1D getGradient(DoubleMatrix1D u, DoubleMatrix1D v,
			double r, double lambda) {
		//Ignore!!
		return null;
	}

}
