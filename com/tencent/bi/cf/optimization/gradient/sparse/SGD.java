package com.tencent.bi.cf.optimization.gradient.sparse;

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


/**
 * SGD-based implementation of a Basic MF solver.
 * @author tigerzhong
 */
public class SGD implements MFSolver{
	/**
	 * Loss function
	 * @weixue All parameter members are static ones, so are shared among all instances?!
	 */
	protected static Loss lossFunc = null;
	/**
	 * Regularization Parameter
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
	public void update(DenseDoubleMatrix2D U, DenseDoubleMatrix2D V, String inputPath, String outputPath) throws Exception {
		//Preprocess, write U, V to HDFS as side-data
	    //Load HadoopMF specific settings along with Hadoop configuration.
		Configuration conf = FileOperators.getConfiguration();
		//Where on HDFS we cache algorithm specific files.
		String catchPath = conf.get("hadoop.cache.path");
		//Save the model (U,V) to HDFS.
		MatrixIO.saveDenseMatrix2D2HDFS(conf, catchPath+"U.dat", U);
		MatrixIO.saveDenseMatrix2D2HDFS(conf, catchPath+"V.dat", V);
		//Save the loss function we used as a conf parameter.
		conf.set("model.loss", ModelOperator.loss2String(conf, SGD.lossFunc));
		//Save other parameters into conf.
		conf.setInt("model.numD", numD);
		conf.setFloat("model.learningRate", (float) learningRate);
		conf.setFloat("model.lambda", (float) lambda);
		//MR, obtaining gradient
		Job job = new Job(conf);
		job.setJarByClass(SGD.class);
		//A MR job for an update.
		job.setJobName("MF-SGD-Update");		
		job.setOutputKeyClass(Text.class);
		job.setOutputValueClass(Text.class);
		//Compute and output gradients for U and V vectors on each training record.
		job.setMapperClass(SGDMapper.class);
		//Merge the gradients.
		job.setReducerClass(CombinerReducer.class);
		job.setInputFormatClass(TextInputFormat.class);
		job.setOutputFormatClass(TextOutputFormat.class);
		FileInputFormat.setInputPaths(job, new Path(inputPath));
		FileOutputFormat.setOutputPath(job, new Path(outputPath));
		//Save paths to U,V as conf parameters.
		DistributedCache.addCacheFile(new URI(catchPath+"U.dat"), job.getConfiguration());
		DistributedCache.addCacheFile(new URI(catchPath+"V.dat"), job.getConfiguration());
		//Submit the job and wait for completion.
		job.waitForCompletion(true);
		//Postprocess, updating U, V
		extract(U, V, job.getConfiguration(), outputPath);
	}
	
	/**
	 * Extract gradients for U and V from file and update U and V accordingly.
	 * @param U
	 * @param V
	 * @param conf
	 * @param pathName The path to combined gradients for U and V.
	 * @throws IOException
	 */
	private void extract(DenseDoubleMatrix2D U, DenseDoubleMatrix2D V, Configuration conf, String pathName) throws IOException{
	    // Load the combined gradients as list of strings.
		List<String> resList = DataOperators.readTextFromHDFS(conf, pathName);
		for(String line : resList){
			String[] items = line.split("\t");
			String c = items[0].split(",")[0];
			// Row index.
			int t = Integer.parseInt(items[0].split(",")[1]);
			// Gradient vector.
			String[] vals = items[1].split(",");
			if(c.equalsIgnoreCase("U")) // U vector
				for(int i=0;i<vals.length;i++)
					U.set(t, i, U.get(t, i) - learningRate*Double.parseDouble(vals[i]));
			else // V vector
				for(int i=0;i<vals.length;i++)
					V.set(t, i, V.get(t, i) - learningRate*Double.parseDouble(vals[i]));
		}
	}
	
	@Override
	public void initialize(Loss lossFunc, double lambda, double learningRate, int numD) throws Exception {
	    //@weixue set class-wise parameters.
		SGD.lossFunc = lossFunc;
		SGD.lambda = lambda;
		SGD.numD = numD;
		SGD.learningRate = learningRate;
	}
	
	/**
	 * Mapper for SGD, compute and output the gradients for U and V vectors.
	 * @author tigerzhong
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
		 * Reused Output Key
		 */
		private Text outKey = new Text();
		/**
		 * Reused Output Text
		 */
		private Text outText = new Text();
		
		@Override
		public void map(LongWritable key, Text value, Context context)
				throws IOException, InterruptedException {
			String items[] = value.toString().split(",",-1); //triplet
			//This training record is cell(p,q)=r.
			int p = Integer.parseInt(items[0]);
			int q = Integer.parseInt(items[1]);
			double r = Double.parseDouble(items[2]);
			//Get gradient
			//@weixue U = mxk, V= nxk, both use "viewRow()".
			DoubleMatrix1D u = lossFunc.getGradient(U.viewRow(p), V.viewRow(q), r, lambda);	
			DoubleMatrix1D v = lossFunc.getGradient(V.viewRow(q), U.viewRow(p), r, lambda);
			//Get loss. Since Hadoop only supports integer(long) counters, here is a trick.
			long cnt = (long)(lossFunc.getValue(U.viewRow(p), V.viewRow(q), r)*1000);
			//Update some useful counters.
			//Loss
			context.getCounter("Eval", "Loss").increment(cnt);
			//Count of training record.
			context.getCounter("Eval", "Cnt").increment(1);
			//Output gradients for U vector and V vector respectively.
			outKey.set("U,"+p);
			outText.set(StringUtils.array2String(u.toArray()));
			context.write(outKey, outText);
			outKey.set("V,"+q);
			outText.set(StringUtils.array2String(v.toArray()));
			context.write(outKey, outText);
		}
		
		@Override
		protected void setup(Context context) throws IOException, InterruptedException{
			super.setup(context);
			Configuration conf = context.getConfiguration();
			// The path to cache files downloaded to local file system.
			Path[] paths = DistributedCache.getLocalCacheFiles(conf); //Read U, V from side data
			// Load U and V into memory.
			U = MatrixIO.readDenseMatrixFromText(paths[0].toString());
			V = MatrixIO.readDenseMatrixFromText(paths[1].toString());
			// Parameters. 
			SGD.numD = conf.getInt("model.numD", 10);
			SGD.lambda = conf.getFloat("model.lambda", 0.005f);
			SGD.lossFunc = ModelOperator.string2Loss(conf,conf.get("model.loss"));
		}
	}
	
}
