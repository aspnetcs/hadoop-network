package com.tencent.bi.cf.optimization.gradient.rank;

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
import org.apache.hadoop.mapreduce.lib.input.TextInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.TextOutputFormat;

import cern.colt.matrix.DoubleMatrix1D;
import cern.colt.matrix.impl.DenseDoubleMatrix2D;

import com.tencent.bi.cf.optimization.Loss;
import com.tencent.bi.cf.optimization.MFSolver;
import com.tencent.bi.cf.optimization.gradient.common.CombinerReducer;
import com.tencent.bi.utils.StringUtils;
import com.tencent.bi.utils.hadoop.DataOperators;
import com.tencent.bi.utils.hadoop.FileOperators;
import com.tencent.bi.utils.hadoop.ModelOperator;
import com.tencent.bi.utils.io.MatrixIO;

/**
 * SGD for BPR-MF
 * @author tigerzhong
 *
 */
public class SGDBPR implements MFSolver {
	/**
	 * Loss function
	 */
	protected static AUCLoss lossFunc = null;
	/**
	 * Regularization parameter
	 */
	protected static double lambda = 0.005;
	/**
	 * Learning rate
	 */
	protected static double learningRate = 0.01;
	/**
	 * Number of latent dimensions
	 */
	protected static int numD = 10;
	
	@Override
	public void initialize(Loss lossFunc, double lambda, double learningRate,
			int numD) throws Exception {
		SGDBPR.lossFunc = (AUCLoss) lossFunc;
		SGDBPR.lambda = lambda;
		SGDBPR.learningRate = learningRate;
		SGDBPR.numD = numD;
	}

	@Override
	public void update(DenseDoubleMatrix2D U, DenseDoubleMatrix2D V,
			String inputPath, String outputPath) throws Exception {
		//Preprocess, write U, V to HDFS as side-data
		Configuration conf = FileOperators.getConfiguration();
		String catchPath = conf.get("hadoop.cache.path");
		MatrixIO.saveDenseMatrix2D2HDFS(conf, catchPath+"U.dat", U);
		MatrixIO.saveDenseMatrix2D2HDFS(conf, catchPath+"V.dat", V);
		conf.set("model.loss", ModelOperator.loss2String(conf, SGDBPR.lossFunc));
		conf.setInt("model.numD", numD);
		conf.setFloat("model.learningRate", (float) learningRate);
		conf.setFloat("model.lambda", (float) lambda);
		//MR, obtaining gradient
		Job job = new Job(conf);
		job.setJarByClass(SGDBPR.class);
		job.setJobName("MF-SGDBPR-Update");		
		job.setOutputKeyClass(Text.class);
		job.setOutputValueClass(Text.class);
		job.setMapperClass(SGDMapper.class);
		job.setReducerClass(CombinerReducer.class);
		job.setInputFormatClass(TextInputFormat.class);
		job.setOutputFormatClass(TextOutputFormat.class);
		FileInputFormat.setInputPaths(job, new Path(inputPath));
		FileOutputFormat.setOutputPath(job, new Path(outputPath));
		DistributedCache.addCacheFile(new URI(catchPath+"U.dat"), job.getConfiguration());
		DistributedCache.addCacheFile(new URI(catchPath+"V.dat"), job.getConfiguration());
		job.waitForCompletion(true);
		//Postprocess, updating U, V
		extract(U, V, job.getConfiguration(), outputPath);
	}

	/**
	 * Extract U and V from file
	 * @param U, latent matrix for users
	 * @param V, latent matrix for items
	 * @param conf, configuration
	 * @param pathName, data path name
	 * @throws IOException
	 */
	private void extract(DenseDoubleMatrix2D U, DenseDoubleMatrix2D V, Configuration conf, String pathName) throws IOException{
		List<String> resList = DataOperators.readTextFromHDFS(conf, pathName);
		for(String line : resList){
			String[] items = line.split("\t");
			String c = items[0].split(",")[0];
			int t = Integer.parseInt(items[0].split(",")[1]);
			String[] vals = items[1].split(",");
			if(c.equalsIgnoreCase("U"))
				for(int i=0;i<vals.length;i++)
					U.set(t, i, U.get(t, i) - learningRate*Double.parseDouble(vals[i]));
			else
				for(int i=0;i<vals.length;i++)
					V.set(t, i, V.get(t, i) - learningRate*Double.parseDouble(vals[i]));
		}
	}
	
	/**
	 * Mapper for SGD, computing the gradients
	 * @author tigerzhong
	 *
	 */
	public static class SGDMapper extends Mapper<LongWritable, Text, Text, Text>{
		/**
		 * Latent Matrix for User
		 */
		protected DenseDoubleMatrix2D U = null;
		/**
		 * Latent Matrix for Item
		 */
		protected DenseDoubleMatrix2D V = null;
		/**
		 * Output Key
		 */
		private Text outKey = new Text();
		/**
		 * Output Text
		 */
		private Text outText = new Text();
		
		@Override
		public void map(LongWritable key, Text value, Context context)
				throws IOException, InterruptedException {
			String line = value.toString().trim();
			String[] items = line.split(",",2); //triplet
			if(items.length<2) return;
			String[] qs = items[1].split(",");
			if(qs.length<2) return;
			int p = Integer.parseInt(items[0]);
			int q1 = Integer.parseInt(qs[0]);
			int q2 = Integer.parseInt(qs[1]);
			//Get gradient
			DoubleMatrix1D u = lossFunc.getGradient(U.viewRow(p), V.viewRow(q1), V.viewRow(q2), 0, lambda);	
			DoubleMatrix1D v1 = lossFunc.getGradient(U.viewRow(p), V.viewRow(q1), V.viewRow(q2), 1, lambda);
			DoubleMatrix1D v2 = lossFunc.getGradient(U.viewRow(p), V.viewRow(q1), V.viewRow(q2), 2, lambda);	
			//Get loss
			long cnt = (long)(lossFunc.getValue(U.viewRow(p), V.viewRow(q1), V.viewRow(q2))*1000);
			context.getCounter("Eval", "Loss").increment(cnt);
			context.getCounter("Eval", "Cnt").increment(1);
			//Output
			outKey.set("U,"+p);
			outText.set(StringUtils.array2String(u.toArray()));
			context.write(outKey, outText);
			outKey.set("V,"+q1);
			outText.set(StringUtils.array2String(v1.toArray()));
			context.write(outKey, outText);
			outKey.set("V,"+q2);
			outText.set(StringUtils.array2String(v2.toArray()));
			context.write(outKey, outText);
		}
		
		@Override
		protected void setup(Context context) throws IOException, InterruptedException{
			super.setup(context);
			Configuration conf = context.getConfiguration();
			Path[] paths = DistributedCache.getLocalCacheFiles(conf); //Read U, V from side data
			U = MatrixIO.readDenseMatrixFromText(paths[0].toString());
			V = MatrixIO.readDenseMatrixFromText(paths[1].toString());
			SGDBPR.numD = conf.getInt("model.numD", 10);
			SGDBPR.lambda = conf.getFloat("model.lambda", 0.005f);
			SGDBPR.lossFunc = (AUCLoss) ModelOperator.string2Loss(conf,conf.get("model.loss"));
		}
	}
}
