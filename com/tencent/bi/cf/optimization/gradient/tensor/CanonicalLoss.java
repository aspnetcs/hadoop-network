package com.tencent.bi.cf.optimization.gradient.tensor;

import cern.colt.matrix.DoubleFactory1D;
import cern.colt.matrix.DoubleMatrix1D;

/**
 * Canonical Loss
 * @author tigerzhong
 *
 */
public class CanonicalLoss implements TensorLoss {

	private static final long serialVersionUID = 1L;

	@Override
	public double getValue(DoubleMatrix1D u, DoubleMatrix1D v,
			DoubleMatrix1D s, double r) {
		int d = u.size();
		double p = 0.0;
		for(int i=0;i<d;i++)
			p += u.get(i)*v.get(i)*s.get(i);
		return (p-r)*(p-r);
	}

	@Override
	public DoubleMatrix1D getGradient(DoubleMatrix1D u, DoubleMatrix1D v,
			DoubleMatrix1D s, double r, double lambda) {
		int d = u.size();
		double p = 0.0;
		for(int i=0;i<d;i++){
			p += u.get(i)*v.get(i)*s.get(i);
		}
		double e = p - r;
		DoubleMatrix1D g = DoubleFactory1D.dense.make(d);
		for(int i=0;i<d;i++){
			double tg = e*(v.get(i)*s.get(i))+lambda*u.get(i);
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
