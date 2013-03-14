package com.tencent.bi.cf.optimization.gradient.hybrid;

import java.io.IOException;
import java.net.URI;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;


import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.filecache.DistributedCache;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.TextInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.TextOutputFormat;
import org.apache.hadoop.mapreduce.lib.input.FileSplit;

import com.tencent.bi.cf.optimization.Loss;
import com.tencent.bi.cf.optimization.MFSolver;
import com.tencent.bi.cf.optimization.gradient.common.CombinerReducer;
import com.tencent.bi.utils.StringUtils;
import com.tencent.bi.utils.hadoop.DataOperators;
import com.tencent.bi.utils.hadoop.FileOperators;
import com.tencent.bi.utils.hadoop.ModelOperator;
import com.tencent.bi.utils.io.MatrixIO;

import cern.colt.matrix.DoubleMatrix2D;
import cern.colt.matrix.impl.DenseDoubleMatrix1D;
import cern.colt.matrix.impl.DenseDoubleMatrix2D;
import cern.jet.math.PlusMult;

/**
 * Distributed SGD for RLMF
 * @author tigerzhong
 *
 */
public class DistributedSGDRegression implements MFSolver{
	/**
	 * Loss function
	 */
	protected static SquareRegressionLoss lossFunc = null;
	/**
	 * Regularization parameter
	 */
	protected static double lambda = 0.005;
	/**
	 * Learning rate
	 */
	protected static double learningRate = 0.01;

	@Override
	public void initialize(Loss lossFunc, double lambda, double learningRate, int numD) throws Exception {
		DistributedSGDRegression.lossFunc = (SquareRegressionLoss) lossFunc;
		DistributedSGDRegression.lambda = lambda;
		DistributedSGDRegression.learningRate = learningRate;
	}
	
	/**
	 * Update the latent matrix
	 * @param B, regression matrix
	 * @param Fu, feature matrix for users
	 * @param Fv, feature matrix for items
	 * @param inputPath, data input path 
	 * @param outputPath, temporary output path 
	 * @throws Exception
	 */
	public void update(DenseDoubleMatrix2D B, String inputPath, String outputPath) throws Exception {
		//Path of all latent matrices
		Configuration conf = FileOperators.getConfiguration();
		String catchPath = conf.get("hadoop.cache.path");
		String tmpPath = conf.get("hadoop.tmp.path")+"DistributedSGDRegression/";
		//Initializing the configuration
		conf.set("model.loss", ModelOperator.loss2String(conf, DistributedSGDRegression.lossFunc));	//loss function
		conf.setFloat("model.learningRate", (float) learningRate);	//learning rate
		conf.setFloat("model.lambda", (float) lambda);				//regularization parameter
		//Saving data for distributed cache	
		MatrixIO.saveDenseMatrix2D2HDFS(conf, catchPath+"B.dat", B);
		//First MR, getting gradients of B
		Job job = new Job(conf);
		job.setJarByClass(DistributedSGDRegression.class);
		job.setJobName("RegressionMF-DistributedSGD-Update-Phase1");
		job.setOutputKeyClass(Text.class);
		job.setOutputValueClass(Text.class);
		job.setMapperClass(SGDMapper.class);
		job.setReducerClass(SGDReducer.class);
		job.setInputFormatClass(TextInputFormat.class);
		job.setOutputFormatClass(TextOutputFormat.class);
		FileInputFormat.addInputPath(job, new Path(inputPath));			//multiple inputs
		FileInputFormat.addInputPath(job, new Path(conf.get("hadoop.data.path")+"Fu"));
		FileOutputFormat.setOutputPath(job, new Path(tmpPath));	//distributed cache
		DistributedCache.addCacheFile(new URI(conf.get("hadoop.data.path")+"Fv.dat"), job.getConfiguration());
		DistributedCache.addCacheFile(new URI(catchPath+"B.dat"), job.getConfiguration());
		job.waitForCompletion(true);
		//Second MR, updating B and V
		job = new Job(conf);
		job.setJarByClass(DistributedSGDRegression.class);
		job.setJobName("RegressionMF-DistributedSGD-Update-Phase2");
		job.setOutputKeyClass(Text.class);
		job.setOutputValueClass(Text.class);
		job.setMapperClass(CombinerMapper.class);
		//job.setCombinerClass(CombinerReducer.class);
		job.setReducerClass(CombinerReducer.class);
		job.setInputFormatClass(TextInputFormat.class);
		job.setOutputFormatClass(TextOutputFormat.class);
		FileInputFormat.setInputPaths(job, new Path(tmpPath));
		FileOutputFormat.setOutputPath(job, new Path(outputPath));
		job.waitForCompletion(true);
		extract(B, conf, outputPath); 	//Extract B from temporary outputs
		FileSystem fs = FileSystem.get(conf); 
		fs.delete(new Path(tmpPath), true);	//Postprocess, removing all temporary outputs
	}
	
	/**
	 * Extracting the gradients of  B from temporary outputs, and then updating them
	 * @param B, regression matrix
	 * @param conf, job configuration
	 * @param pathName, name of input path
	 * @throws IOException
	 */
	private void extract(DenseDoubleMatrix2D B, Configuration conf, String pathName) throws IOException{
		//Reading all from outputs
		List<String> resList = DataOperators.readTextFromHDFS(conf, pathName);
		//Processing line by line
		for(String line : resList){
			String[] items = line.split("\t");
			String[] vals = items[1].split(",");
			int pt = 0;
			for(int i=0;i<B.rows();i++)
				for(int j=0;j<B.columns();j++)
					B.setQuick(i, j, B.getQuick(i, j)- learningRate*Double.parseDouble(vals[pt++])/conf.getInt("mapred.reduce.tasks",100));
		}
	}
	
	/**
	 * First mapper, for distributing data
	 * @author tigerzhong
	 *
	 */
	public static class SGDMapper extends Mapper<LongWritable, Text, Text, Text> {
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
			String path = ((FileSplit) context.getInputSplit()).getPath().toString();	//get the path of input data
			String[] items = value.toString().split(",",2);	//get key
			outKey.set(items[0]);
			String outLine = items[1];
			if (path.indexOf("/Fu")>-1) //features for user
				outLine += ",fu";
			outText.set(outLine);
			context.write(outKey, outText);
		}
	}
	
	/**
	 * First reducer, getting the gradients of all factors and updating U
	 * @author tigerzhong
	 *
	 */
	public static class SGDReducer extends Reducer<Text, Text, Text, Text> {
		/**
		 * Feature vector for the current user
		 */
		protected static DenseDoubleMatrix1D Fu = null;
		/**
		 * Regression matrix
		 */
		protected static DenseDoubleMatrix2D B = null;
		/**
		 * Gradients of B
		 */
		protected static DenseDoubleMatrix2D gB = null;
		/**
		 * Feature matrix for items
		 */
		protected static DenseDoubleMatrix2D Fv = null;
		/**
		 * Output Key
		 */
		private Text outKey = new Text();
		/**
		 * Output Text
		 */
		private Text outText = new Text();
		
		@Override
		public void reduce(Text key, Iterable<Text> values, Context context) throws IOException, InterruptedException {
			Map<Integer, Double> valueList = new HashMap<Integer, Double>();
			Iterator<Text> it = values.iterator();
			gB = new DenseDoubleMatrix2D(B.rows(), B.columns());
			while(it.hasNext()){
				String line = it.next().toString();
				String[] items = line.split(",");
				if (items[items.length-1].equals("fu")) { //features for user
					Fu = new DenseDoubleMatrix1D(items.length-1);
					for(int i=0;i<items.length-1;i++)
						Fu.setQuick(i, Double.parseDouble(items[i]));
				} else {	//dating data
					valueList.put(Integer.parseInt(items[0]), Double.parseDouble(items[1]));
				}
			}
			int numR = valueList.size();		//number of data for the current user
			for(Integer q: valueList.keySet()){	//process each click data
				double r = valueList.get(q);	//click
				//Compute square error, for monitor
				long cnt = (long)(lossFunc.getValue(B, Fu, Fv.viewRow(q), r)*1000);
				context.getCounter("Eval", "Loss").increment(cnt);
				context.getCounter("Eval", "Cnt").increment(1);
				//Get gradient
				DoubleMatrix2D b = lossFunc.getGradientB(B, Fu, Fv.viewRow(q), r, lambda);
				//Combine B's gradients
				gB.assign(b, PlusMult.plusDiv(numR));
			}
			outKey.set(""+(Integer.parseInt(key.toString().trim())%context.getNumReduceTasks()));
			outText.set(StringUtils.matrix2String(gB));
			context.write(outKey, outText);
		}
		
		@Override
		protected void setup(Context context) throws IOException, InterruptedException{
			super.setup(context);
			Configuration conf = context.getConfiguration();
			//Read V, FV and B
			Path[] paths = DistributedCache.getLocalCacheFiles(conf);
			Fv = MatrixIO.readDenseMatrixFromText(paths[0].toString());	//feature matrix for items
			B = MatrixIO.readDenseMatrixFromText(paths[1].toString());	//regression matrix
			DistributedSGDRegression.lambda = conf.getFloat("model.lambda", 0.005f);	//learning rate
			DistributedSGDRegression.lossFunc = (SquareRegressionLoss) ModelOperator.string2Loss(conf,conf.get("model.loss")); //loss function
		}
	}
	
	/**
	 * Second mapper, for distributing data
	 * @author tigerzhong
	 *
	 */
	public static class CombinerMapper extends Mapper<LongWritable, Text, Text, Text> {
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
			/*Distribute the gradient of B*/
			String[] items = value.toString().split("\t");
			outKey.set(items[0]);
			outText.set(items[1]);
			context.write(outKey, outText);
		}
	}
	
/////////////////////////////////////////////////////////////////////////////////////////////////////	
	@Deprecated
	public void update(DenseDoubleMatrix2D U, DenseDoubleMatrix2D V,
			String inputPath, String outputPath) throws Exception {
		//Ignore
		return;
	}
}
