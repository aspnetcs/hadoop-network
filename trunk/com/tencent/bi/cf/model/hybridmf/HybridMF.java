package com.tencent.bi.cf.model.hybridmf;

import org.apache.hadoop.conf.Configuration;

import com.tencent.bi.cf.optimization.gradient.hybrid.SGDHybrid;
import com.tencent.bi.cf.optimization.gradient.hybrid.SquareHybridLoss;
import com.tencent.bi.utils.hadoop.FileOperators;

import cern.colt.matrix.DoubleFactory2D;
import cern.colt.matrix.DoubleMatrix1D;
import cern.colt.matrix.impl.DenseDoubleMatrix2D;

/**
 * Regression plus MF
 * R = UV' + Fu*B*Fv'
 * @author tigerzhong
 *
 */
public class HybridMF extends AbstractHybridMF {

	/**
	 * Regression matrix
	 */
	protected DenseDoubleMatrix2D B = null;	
	/**
	 * User feature matrix
	 */
	protected DenseDoubleMatrix2D Fu = null;	
	/**
	 * Item feature matrix
	 */
	protected DenseDoubleMatrix2D Fv = null;	
	/**
	 * Optimization method
	 */
	protected SGDHybrid solver;
	/**
	 * Loss function
	 */
	protected SquareHybridLoss lossFunc;
	
	@Override
	public void initModel(int m, int n, int fm, int fn, int d, String solverName, double lambda, double learningRate, int numIt, String inPath) throws Exception{
		super.initModel(m, n, d, "", lambda, learningRate, numIt, inPath);
		B = (DenseDoubleMatrix2D) DoubleFactory2D.dense.random(fm,fn);
		solver = new SGDHybrid();
	}
	
	@Override
	public void buildModel() throws Exception {
		SquareHybridLoss lossFunc = new SquareHybridLoss();
		Configuration conf = FileOperators.getConfiguration();
		for(int i=0;i<super.numIt;i++){
			solver.initialize(lossFunc, super.lambda, super.learningRate, super.U.columns());
			solver.update(super.U, super.V, B, super.inputPath, conf.get("hadoop.output.path")+"HybridMF/"+i+"/");
		}
	}

	@Override
	public double predict(int p, int q, int o) {
		return lossFunc.getPrediction(super.U.viewRow(p), super.V.viewRow(q), this.B, Fu.viewRow(p), Fv.viewRow(q));
	}

	@Deprecated
	public double predict(DoubleMatrix1D u, DoubleMatrix1D v, DoubleMatrix1D s) {
		// TODO Auto-generated method stub
		return 0;
	}

	@Deprecated
	public void predictPair(String inputPath, int numD)
			throws Exception {		
	}

	@Deprecated
	public void predictAll() throws Exception {
	}

}
