package com.tencent.bi.cf.model.tensor;

import org.apache.hadoop.conf.Configuration;

import cern.colt.matrix.DoubleMatrix1D;

import com.tencent.bi.cf.optimization.gradient.tensor.CanonicalLoss;
import com.tencent.bi.utils.hadoop.FileOperators;

/**
 * Canonical Tensor Factorization
 * T_ijk = U_i .* V_j .* S_k
 * @author tigerzhong
 *
 */
public class CanonicalTF extends AbstractTensor {
	
	@Override
	public void buildModel() throws Exception {
		Configuration conf = FileOperators.getConfiguration();
		CanonicalLoss lossFunc = new CanonicalLoss();
		for(int i=0;i<super.numIt;i++){
			solver.initialize(lossFunc, super.lambda, super.learningRate, super.U.columns());
			solver.update(super.U, super.V, S, super.inputPath, conf.get("hadoop.output.path")+"CanonicalTF/"+i+"/");
		}
	}

	@Override
	public double predict(int p, int q, int o) {
		/*p=u*v*s*/
		int d = U.columns();
		double pre = 0.0;
		for(int i=0;i<d;i++)
			pre += U.viewRow(p).get(i)*V.viewRow(q).get(i)*S.viewRow(o).get(i);
		return pre;
	}

	@Deprecated
	public double predict(DoubleMatrix1D u, DoubleMatrix1D v, DoubleMatrix1D s) {
		// TODO Auto-generated method stub
		return 0;
	}

	@Deprecated
	public void predictPair(String inputPath, int numD)
			throws Exception {
		// TODO Auto-generated method stub
		
	}

	@Deprecated
	public void predictAll() throws Exception {
		// TODO Auto-generated method stub
		
	}

}
