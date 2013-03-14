package com.tencent.bi.cf.optimization.gradient.sparse;

import com.tencent.bi.cf.optimization.Loss;

import cern.colt.matrix.DoubleFactory1D;
import cern.colt.matrix.DoubleMatrix1D;

/**
 * Loss using Poisson Distribution as Posterior Probability  
 * @author tigerzhong
 *
 */
public class PoissonLoss implements Loss {

	private static final long serialVersionUID = 1L;

	@Override
	public double getValue(DoubleMatrix1D u, DoubleMatrix1D v, double r) {
		double p = u.zDotProduct(v);
		return p-r*Math.log(p);
	}

	@Override
	public DoubleMatrix1D getGradient(DoubleMatrix1D u, DoubleMatrix1D v,
			double r, double lambda) {
		/*
		 * g = v - r/u + lambda*u
		 */
		int d = u.size();
		DoubleMatrix1D g = DoubleFactory1D.dense.make(d);
		for(int i=0;i<d;i++){
			double tg = v.get(i) - r/u.get(i) + lambda*u.get(i);
			g.set(i, tg);
		}
		return g;
	}

}
