package com.tencent.bi.cf.model.tensor;

import org.apache.hadoop.conf.Configuration;

import cern.colt.matrix.DoubleMatrix1D;

import com.tencent.bi.cf.optimization.gradient.tensor.PairLoss;
import com.tencent.bi.utils.hadoop.FileOperators;

/**
 * Pair Tensor Factorization
 * T_ijk = U_iV_j + V_jS_k + U_iS_j
 * @author tigerzhong
 *
 */
public class PairTF extends AbstractTensor{
	
	@Override
	public void buildModel() throws Exception {
		PairLoss lossFunc = new PairLoss();
		Configuration conf = FileOperators.getConfiguration();
		for(int i=0;i<super.numIt;i++){
			solver.initialize(lossFunc, super.lambda, super.learningRate, super.U.columns());
			solver.update(super.U, super.V, S, super.inputPath, conf.get("hadoop.output.path")+"PairTF/"+i+"/");
		}
	}

	@Override
	public double predict(int p, int q, int o) {
		/*uv+us+vs*/
		int d = U.columns();
		double pre = 0.0;
		for(int i=0;i<d;i++){
			pre += U.viewRow(p).get(i)*V.viewRow(q).get(i)+S.viewRow(o).get(i)*V.viewRow(q).get(i)+U.viewRow(p).get(i)*S.viewRow(o).get(i);
		}
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
