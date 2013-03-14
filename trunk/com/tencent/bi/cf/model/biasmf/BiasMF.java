package com.tencent.bi.cf.model.biasmf;

import java.util.List;

import org.apache.hadoop.conf.Configuration;

import com.tencent.bi.cf.model.AbstractMF;
import com.tencent.bi.cf.model.common.BiasGenerator;
import com.tencent.bi.cf.optimization.gradient.bias.SGDBias;
import com.tencent.bi.cf.optimization.gradient.bias.SquareBiasLoss;
import com.tencent.bi.utils.hadoop.DataOperators;
import com.tencent.bi.utils.hadoop.FileOperators;

import cern.colt.matrix.DoubleFactory1D;
import cern.colt.matrix.DoubleMatrix1D;
import cern.colt.matrix.impl.DenseDoubleMatrix1D;

/**
 * Matrix Factorization with User and Item Bias
 * R = UV' + uBias + vBias
 * @author tigerzhong
 *
 */
public class BiasMF extends AbstractMF {
	/**
	 * User Bias
	 */
	protected DenseDoubleMatrix1D uBias = null;
	/**
	 * Item Bias
	 */
	protected DenseDoubleMatrix1D vBias = null;
	/**
	 * Optimization method
	 */
	protected SGDBias solver = null;
	
	@Override
	public void initModel(int m, int n, int d, String solverName, double lambda, double learningRate, int numIt, String inputPath) throws Exception{
		super.initModel(m, n, d, "", lambda, learningRate, numIt, inputPath);
		BiasGenerator.getBias(super.inputPath, m, n);
		Configuration conf = FileOperators.getConfiguration();
		//Get item bias
		List<String> line = DataOperators.readTextFromHDFS(new Configuration(), conf.get("hadoop.cache.path")+"vBias.dat");
		vBias = (DenseDoubleMatrix1D) DoubleFactory1D.dense.make(n);
		for(int i=0;i<line.size();i++){
			vBias.set(i, Double.parseDouble(line.get(i)));
		}
		//Get user bias	
		List<String> resList = DataOperators.readTextFromHDFS(new Configuration(), conf.get("hadoop.cache.path")+"uBias/");
		uBias = new DenseDoubleMatrix1D(m);
		for(String resLine : resList){
			String[] items = resLine.split(",");
			uBias.set(Integer.parseInt(items[0]), Double.parseDouble(items[1]));
		}
		solver = (SGDBias) Class.forName(solverName).newInstance();
	}

	@Override
	public void buildModel() throws Exception {
		SquareBiasLoss lossFunc = new SquareBiasLoss();
		Configuration conf = FileOperators.getConfiguration();
		for(int i=0;i<super.numIt;i++){
			solver.initialize(lossFunc, super.lambda, super.learningRate, super.U.columns());
			solver.update(super.U, super.V, this.uBias, this.vBias, super.inputPath, conf.get("hadoop.output.path")+"/BiasMF/"+i+"/");
		}			
	}

	@Override
	public double predict(int p, int q, int o) {
		/* uv + uBias + vBias*/
		return U.viewRow(p).zDotProduct(V.viewRow(q)) + uBias.get(p) + vBias.get(q);
	}

	public void setuBias(DenseDoubleMatrix1D uBias) {
		this.uBias = uBias;
	}

	public void setvBias(DenseDoubleMatrix1D vBias) {
		this.vBias = vBias;
	}

	@Deprecated
	public double predict(DoubleMatrix1D u, DoubleMatrix1D v, DoubleMatrix1D s) { return 0; }

	@Deprecated
	public void predictPair(String inputPath, int numD) throws Exception {	}

	@Deprecated
	public void predictAll() throws Exception {	}
}
