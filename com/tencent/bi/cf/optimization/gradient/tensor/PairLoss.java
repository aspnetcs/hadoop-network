package com.tencent.bi.cf.optimization.gradient.tensor;

import cern.colt.matrix.DoubleFactory1D;
import cern.colt.matrix.DoubleMatrix1D;

/**
 * Pair based Loss
 * @author tigerzhong
 *
 */
public class PairLoss implements TensorLoss {

	private static final long serialVersionUID = 1L;

	@Override
	public double getValue(DoubleMatrix1D u, DoubleMatrix1D v, DoubleMatrix1D s, double r) {
		double a = u.zDotProduct(v);
		double b = v.zDotProduct(s);
		double c = u.zDotProduct(v);	
		return (r-a-b-c)*(r-a-b-c);
	}

	@Override
	public DoubleMatrix1D getGradient(DoubleMatrix1D u, DoubleMatrix1D v,
			DoubleMatrix1D s, double r, double lambda) {
		double a = u.zDotProduct(v);
		double b = v.zDotProduct(s);
		double c = u.zDotProduct(v);
		double e = a+b+c - r;
		int d = u.size();
		DoubleMatrix1D g = DoubleFactory1D.dense.make(d);
		for(int i=0;i<d;i++){
			double tg = e*(v.get(i)+s.get(i))+lambda*u.get(i);
			g.set(i, tg);
		}
		return g;
	}

/////////////////////////////////////////////////////////////////////////////////////////////////////	
	@Deprecated
	public double getValue(DoubleMatrix1D u, DoubleMatrix1D v, double r) {
		return 0;
	}

	@Deprecated
	public DoubleMatrix1D getGradient(DoubleMatrix1D u, DoubleMatrix1D v,
			double r, double lambda) {
		return null;
	}

}
