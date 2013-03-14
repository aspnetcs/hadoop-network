package com.tencent.bi.cf.optimization.gradient.collective;

import java.io.IOException;
import java.net.URI;
import java.util.List;


import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.filecache.DistributedCache;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.FileSplit;
import org.apache.hadoop.mapreduce.lib.input.TextInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.TextOutputFormat;

import com.tencent.bi.cf.optimization.Loss;
import com.tencent.bi.cf.optimization.MFSolver;
import com.tencent.bi.cf.optimization.gradient.common.CombinerReducer;
import com.tencent.bi.utils.StringUtils;
import com.tencent.bi.utils.hadoop.DataOperators;
import com.tencent.bi.utils.hadoop.FileOperators;
import com.tencent.bi.utils.hadoop.ModelOperator;
import com.tencent.bi.utils.io.MatrixIO;

import cern.colt.matrix.DoubleMatrix1D;
import cern.colt.matrix.impl.DenseDoubleMatrix2D;
import cern.jet.math.Mult;

/**
 * SGD for CMF 
 * @author tigerzhong
 *
 */
public class SGDCollective implements MFSolver {
	/**
	 * Loss function
	 */
	protected static Loss lossFunc = null;
	/**
	 * Latent matrix for user
	 */
	protected static DenseDoubleMatrix2D U = null;
	/**
	 * Latent matrix for item
	 */
	protected static DenseDoubleMatrix2D V = null;
	/**
	 * Latent matrix for context
	 */
	protected static DenseDoubleMatrix2D S = null;
	/**
	 * Regularization parameter
	 */
	protected static double lambdaV = 0.005;
	/**
	 * Regularization parameter
	 */
	protected static double lambdaS = 0.005;
	/**
	 * Regularization parameter
	 */
	protected static double lambda = 0.005;
	/**
	 * Learning Rate
	 */
	protected static double learningRate = 0.01;
	/**
	 * Number of Latent Dimension
	 */
	protected static int numD = 10;
	
	public void initialize(Loss lossFunc, double lambdaV, double lambdaS, double lambda, double learningRate, int numD)
			throws Exception {
		SGDCollective.lossFunc = lossFunc;
		SGDCollective.lambda = lambda;
		SGDCollective.learningRate = learningRate;
		SGDCollective.numD = numD;
		SGDCollective.lambdaS = lambdaS;
		SGDCollective.lambdaV = lambdaV;
	}
	
	/**
	 * Update for collective MF
	 * @param U
	 * @param V
	 * @param S
	 * @param inputPath
	 * @param outputPath
	 * @throws Exception
	 */
	public void update(DenseDoubleMatrix2D U, DenseDoubleMatrix2D V, DenseDoubleMatrix2D S, String inputPath, String outputPath)
			throws Exception {
		Configuration conf = FileOperators.getConfiguration();
		MatrixIO.saveDenseMatrix2D2HDFS(conf, conf.get("hadoop.cache.path")+"U.dat", U);
		MatrixIO.saveDenseMatrix2D2HDFS(conf, conf.get("hadoop.cache.path")+"V.dat", V);
		MatrixIO.saveDenseMatrix2D2HDFS(conf, conf.get("hadoop.cache.path")+"S.dat", S);
		conf.set("model.loss", ModelOperator.loss2String(conf, SGDCollective.lossFunc));
		conf.setInt("model.numD", numD);
		conf.setFloat("model.learningRate", (float) learningRate);
		conf.setFloat("model.lambda", (float) lambda);
		conf.setFloat("model.lambdaV", (float) lambdaV);
		conf.setFloat("model.lambdaS", (float) lambdaS);
		//MR, get gradient
		Job job = new Job(conf);
		job.setJobName("CMF-SGD-Update");		
		job.setOutputKeyClass(Text.class);
		job.setOutputValueClass(Text.class);
		job.setMapperClass(SGDMapper.class);
		job.setReducerClass(CombinerReducer.class);
		job.setInputFormatClass(TextInputFormat.class);
		job.setOutputFormatClass(TextOutputFormat.class);
		FileInputFormat.setInputPaths(job, new Path(inputPath));
		FileOutputFormat.setOutputPath(job, new Path(outputPath));
		DistributedCache.addCacheFile(new URI(conf.get("hadoop.cache.path")+"U.dat"), job.getConfiguration());
		DistributedCache.addCacheFile(new URI(conf.get("hadoop.cache.path")+"V.dat"), job.getConfiguration());
		DistributedCache.addCacheFile(new URI(conf.get("hadoop.cache.path")+"S.dat"), job.getConfiguration());
		job.waitForCompletion(true);
		//Postprocess, updating U, V, S
		extract(U, V, S, job.getConfiguration(), outputPath);
	}
	/**
	 * Extract U,V,S
	 * @param U
	 * @param V
	 * @param S
	 * @param conf
	 * @param pathName
	 * @throws IOException
	 */
	private void extract(DenseDoubleMatrix2D U, DenseDoubleMatrix2D V, DenseDoubleMatrix2D S, Configuration conf, String pathName) throws IOException{
		List<String> resList = DataOperators.readTextFromHDFS(conf, pathName);
		for(String line : resList){
			String[] items = line.split("\t");
			String c = items[0].split(",")[0];
			int t = Integer.parseInt(items[0].split(",")[1]);
			String[] vals = items[1].split(",");
			if(c.equalsIgnoreCase("U")){
				for(int i=0;i<vals.length;i++)
					U.set(t, i, U.get(t, i) - learningRate*Double.parseDouble(vals[i]));
			}
			else if(c.equalsIgnoreCase("V")){
				for(int i=0;i<vals.length;i++)
					V.set(t, i, V.get(t, i) - learningRate*Double.parseDouble(vals[i]));				
			}
			else if(c.equalsIgnoreCase("S")){
				for(int i=0;i<vals.length;i++)
					S.set(t, i, S.get(t, i) - learningRate*Double.parseDouble(vals[i]));				
			}
		}
	}
	
	/**
	 * Mapper for SGD
	 * @author tigerzhong
	 *
	 */
	public class SGDMapper extends Mapper<LongWritable, Text, Text, Text>{
		/**
		 * Output key
		 */
		private Text outKey = new Text();
		/**
		 * Output value
		 */
		private Text outValue = new Text();
		@Override
		public void map(LongWritable key, Text value, Context context)
				throws IOException, InterruptedException {
			String path = ((FileSplit) context.getInputSplit()).getPath().toString();
			String items[] = value.toString().split(",",-1);
			int p = Integer.parseInt(items[0]);
			int q = Integer.parseInt(items[1]);
			double r = Double.parseDouble(items[2]);
			DoubleMatrix1D a = null, b = null;
			if(path.indexOf("/MU")>-1){
				a = lossFunc.getGradient(U.viewRow(p), V.viewRow(q), r, lambda);
				b = lossFunc.getGradient(V.viewRow(q), U.viewRow(p), r, lambda);
				outKey.set("U,"+p);
				outValue.set(StringUtils.array2String(a.toArray()));
				context.write(outKey, outValue);
				outKey.set("V,"+q);
				outValue.set(StringUtils.array2String(b.toArray()));
				context.write(outKey, outValue);
			}
			else if(path.indexOf("/MS")>-1){
				a = lossFunc.getGradient(U.viewRow(p), S.viewRow(q), r, lambda);
				b = lossFunc.getGradient(S.viewRow(q), U.viewRow(p), r, lambda);
				a.assign(Mult.mult(lambdaV));
				b.assign(Mult.mult(lambdaV));
				outKey.set("U,"+p);
				outValue.set(StringUtils.array2String(a.toArray()));
				context.write(outKey, outValue);
				outKey.set("S,"+q);
				outValue.set(StringUtils.array2String(b.toArray()));
				context.write(outKey, outValue);
			}
			else if(path.indexOf("/MV")>-1){
				a = lossFunc.getGradient(V.viewRow(p), S.viewRow(q), r, lambda);
				b = lossFunc.getGradient(S.viewRow(q), V.viewRow(p), r, lambda);
				a.assign(Mult.mult(lambdaS));
				b.assign(Mult.mult(lambdaS));
				outKey.set("V,"+p);
				outValue.set(StringUtils.array2String(a.toArray()));
				context.write(outKey, outValue);
				outKey.set("S,"+q);
				outValue.set(StringUtils.array2String(b.toArray()));
				context.write(outKey, outValue);
			}
		}
		
		@Override
		protected void setup(Context context) throws IOException, InterruptedException{
			super.setup(context);
			Configuration conf = context.getConfiguration();
			//Read U, V, S
			Path[] paths = DistributedCache.getLocalCacheFiles(conf);
			U = MatrixIO.readDenseMatrixFromText(paths[0].toString());
			V = MatrixIO.readDenseMatrixFromText(paths[1].toString());
			S = MatrixIO.readDenseMatrixFromText(paths[2].toString());
			SGDCollective.numD = conf.getInt("model.numD", 10);
			SGDCollective.lambda = conf.getFloat("model.lambda", 0.005f);
			SGDCollective.lambdaS = conf.getFloat("model.lambdaS", 0.005f);
			SGDCollective.lambdaV = conf.getFloat("model.lambdaV", 0.005f);
			SGDCollective.lossFunc = ModelOperator.string2Loss(conf,conf.get("model.loss"));
		}
	}
	
//////////////////////////////////////////////////////////////////////////////////////////////////////////////
	@Deprecated
	public void update(DenseDoubleMatrix2D U, DenseDoubleMatrix2D V, String inputPath, String outputPath)
			throws Exception {
		//Ignore
		return;
	}

	@Deprecated
	public void initialize(Loss lossFunc, double lambda, double learningRate,
			int numD) throws Exception {
		// TODO Auto-generated method stub
	}
}
