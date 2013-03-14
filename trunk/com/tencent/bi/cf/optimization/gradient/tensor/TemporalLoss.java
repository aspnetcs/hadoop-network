package com.tencent.bi.cf.optimization.gradient.tensor;

import cern.colt.matrix.DoubleFactory2D;
import cern.colt.matrix.DoubleMatrix1D;
import cern.colt.matrix.DoubleMatrix2D;

public class TemporalLoss extends CanonicalLoss {

	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;

	public DoubleMatrix2D getGradient(DoubleMatrix1D u, DoubleMatrix1D v, DoubleMatrix1D s1,
			DoubleMatrix1D s, double r, double lambda) {
		int d = u.size();
		double p = 0.0;
		for(int i=0;i<d;i++)
			p += u.get(i)*v.get(i)*s.get(i);
		double e = p - r;
		DoubleMatrix2D g = DoubleFactory2D.dense.make(d, 3);
		for(int i=0;i<d;i++){
			double gu = e*(v.get(i)*s.get(i))+lambda*u.get(i);
			double gv = e*(u.get(i)*s.get(i))+lambda*v.get(i);
			double gs = e*(v.get(i)*u.get(i))+lambda*(s.get(i)-s1.get(i));
			g.set(0, i, gu);
			g.set(1, i, gv);
			g.set(2, i, gs);
		}
		return g;
	}

}
