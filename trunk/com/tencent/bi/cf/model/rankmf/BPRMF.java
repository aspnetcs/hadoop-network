package com.tencent.bi.cf.model.rankmf;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;

import cern.colt.matrix.DoubleMatrix1D;

import com.tencent.bi.cf.model.AbstractMF;
import com.tencent.bi.cf.optimization.gradient.rank.AUCLoss;
import com.tencent.bi.utils.hadoop.FileOperators;

/**
 * 	Bayesian Personalized Ranking with MF
 * 
 *	BibTeX:
 * <pre>
 * 	@inproceedings{Rendle:2009:BBP:1795114.1795167,
 * 		author = {Rendle, Steffen and Freudenthaler, Christoph and Gantner, Zeno and Schmidt-Thie Lars},
 * 		title = {BPR: Bayesian personalized ranking from implicit feedback},
 * 		booktitle = {Proceedings of the Twenty-Fifth Conference on Uncertainty in Artificial Intelligence},
 * 		series = {UAI '09},
 * 		year = {2009},
 * 		location = {Montreal, Quebec, Canada},
 * 		pages = {452--461},
 * 		publisher = {AUAI Press},
 * 		address = {Arlington, Virginia, United States},
 * 	} 
 * </pre>
 * 
 * @author tigerzhong
 *
 */
public class BPRMF extends AbstractMF{

	@Override
	public void buildModel() throws Exception {
		//Preprocess
		Configuration conf = FileOperators.getConfiguration();
		PairwiseGenerator.getPariwise(super.inputPath, conf.get("hadoop.tmp.path")+"/Pairwise/");
		AUCLoss lossFunc = new AUCLoss();
		for(int i=0;i<super.numIt;i++){
			solver.initialize(lossFunc, super.lambda, super.learningRate, super.numD);
			solver.update(super.U, super.V, conf.get("hadoop.tmp.path")+"/Pairwise/", conf.get("hadoop.output.path")+"/BPRMF/"+i+"/");
		}
		//Postprocess
		FileSystem fs = FileSystem.get(new Configuration());
		fs.delete(new Path(conf.get("hadoop.output.path")), true);
	}

	@Override
	public double predict(int p, int q, int o) {
		return U.viewRow(p).zDotProduct(V.viewRow(q));
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
