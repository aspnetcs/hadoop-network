package com.tencent.bi.cf.optimization.gradient.collective;

import java.io.IOException;
import java.net.URI;
import java.util.HashMap;
import java.util.List;
import java.util.Map;


import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.filecache.DistributedCache;
import org.apache.hadoop.fs.FSDataOutputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
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
import cern.colt.matrix.impl.DenseDoubleMatrix1D;
import cern.colt.matrix.impl.DenseDoubleMatrix2D;
import cern.colt.matrix.linalg.SeqBlas;
import cern.jet.math.Mult;

/**
 * Distributed SGD for CMF 
 * @author tigerzhong
 *
 */
public class DistributedSGDCollective implements MFSolver {
	/**
	 * Loss function
	 */
	protected static Loss lossFunc = null;
	/**
	 * Latent matrix for item
	 */
	protected static DenseDoubleMatrix2D V = null;
	/**
	 * Latent matrix for context
	 */
	protected static DenseDoubleMatrix2D S = null;
	/**
	 * Regularization parameter for R_v
	 */
	protected static double lambdaV = 0.005;
	/**
	 * Regularization parameter for R_s
	 */
	protected static double lambdaS = 0.005;
	/**
	 * Regularization parameter
	 */
	protected static double lambda = 0.005;
	/**
	 * Learning rate
	 */
	protected static double learningRate = 0.01;
	/**
	 * Number of Latent Dimension
	 */
	protected static int numD = 10;
	
	public void initialize(Loss lossFunc, double lambdaV, double lambdaS, double lambda, double learningRate, int numD)
			throws Exception {
		DistributedSGDCollective.lossFunc = lossFunc;
		DistributedSGDCollective.lambda = lambda;
		DistributedSGDCollective.learningRate = learningRate;
		DistributedSGDCollective.numD = numD;
		DistributedSGDCollective.lambdaS = lambdaS;
		DistributedSGDCollective.lambdaV = lambdaV;
	}
	
	/**
	 * Update for collective MF
	 * @param U, latent matrix for user
	 * @param V, latent matrix for item
	 * @param S, latent matrix for context
	 * @param inputPath
	 * @param outputPath
	 * @throws Exception
	 */
	public void update(DenseDoubleMatrix2D U, DenseDoubleMatrix2D V, DenseDoubleMatrix2D S, String inputPath, String outputPath)
			throws Exception {
		Configuration conf = FileOperators.getConfiguration();
		String tmpPath = conf.get("hadoop.tmp.path")+"DistributedSGDCollective/";
		MatrixIO.saveDenseMatrix2D2HDFS(conf, conf.get("hadoop.cache.path")+"V.dat", V);
		MatrixIO.saveDenseMatrix2D2HDFS(conf, conf.get("hadoop.cache.path")+"S.dat", S);
		conf.set("model.loss", ModelOperator.loss2String(conf, DistributedSGDCollective.lossFunc));
		conf.setInt("model.numD", numD);
		conf.setFloat("model.learningRate", (float) learningRate);
		conf.setFloat("model.lambda", (float) lambda);
		conf.setFloat("model.lambdaV", (float) lambdaV);
		conf.setFloat("model.lambdaS", (float) lambdaS);
		//First MR, get gradient
		Job job = new Job(conf);
		job.setJarByClass(DistributedSGDCollective.class);
		job.setJobName("CMF-DistibutedSGD-Update-Phase1");		
		job.setOutputKeyClass(Text.class);
		job.setOutputValueClass(Text.class);
		job.setMapperClass(SGDMapper.class);
		job.setReducerClass(SGDReducer.class);
		job.setInputFormatClass(TextInputFormat.class);
		job.setOutputFormatClass(TextOutputFormat.class);
		FileInputFormat.addInputPath(job, new Path(inputPath+"MU/"));
		FileInputFormat.addInputPath(job, new Path(inputPath+"MV/"));
		FileInputFormat.addInputPath(job, new Path(inputPath+"MS/"));
		FileInputFormat.addInputPath(job, new Path(conf.get("hadoop.cache.path")+"U/"));
		FileOutputFormat.setOutputPath(job, new Path(tmpPath));
		DistributedCache.addCacheFile(new URI(conf.get("hadoop.cache.path")+"V.dat"), job.getConfiguration());
		DistributedCache.addCacheFile(new URI(conf.get("hadoop.cache.path")+"S.dat"), job.getConfiguration());
		job.waitForCompletion(true);
		FileSystem fs = FileSystem.get(conf); 
		fs.delete(new Path(conf.get("hadoop.cache.path")+"U/"), true);
		//Second MR, combiner
		job = new Job(conf);
		job.setJarByClass(DistributedSGDCollective.class);
		job.setJobName("CMF-DistibutedSGD-Update-Phase2");		
		job.setOutputKeyClass(Text.class);
		job.setOutputValueClass(Text.class);
		job.setMapperClass(CombinerMapper.class);
		job.setReducerClass(CombinerReducer.class);
		job.setInputFormatClass(TextInputFormat.class);
		job.setOutputFormatClass(TextOutputFormat.class);
		FileInputFormat.setInputPaths(job, new Path(tmpPath));
		FileOutputFormat.setOutputPath(job, new Path(outputPath));
		DistributedCache.addCacheFile(new URI(conf.get("hadoop.cache.path")+"V.dat"), job.getConfiguration());
		DistributedCache.addCacheFile(new URI(conf.get("hadoop.cache.path")+"S.dat"), job.getConfiguration());
		job.waitForCompletion(true);
		//Postprocess, updating V, S
		extract(V, S, job.getConfiguration(), outputPath);
		fs.delete(new Path(tmpPath), true);
	}
	/**
	 * Extract V,S
	 * @param V
	 * @param S
	 * @param conf
	 * @param pathName
	 * @throws IOException
	 */
	private void extract(DenseDoubleMatrix2D V, DenseDoubleMatrix2D S, Configuration conf, String pathName) throws IOException{
		List<String> resList = DataOperators.readTextFromHDFS(conf, pathName);
		for(String line : resList){
			String[] items = line.split("\t");
			String c = items[0].split(",")[0];
			int t = Integer.parseInt(items[0].split(",")[1]);
			String[] vals = items[1].split(",");
			if(c.equalsIgnoreCase("V")){
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
	 * First mapper, for distributing data
	 * @author tigerzhong
	 *
	 */
	public static class SGDMapper extends Mapper<LongWritable, Text, Text, Text> {
		/**
		 * Output key
		 */
		private Text outKey = new Text();
		/**
		 * Output value
		 */
		private Text outText = new Text();

		@Override
		public void map(LongWritable key, Text value, Context context)
				throws IOException, InterruptedException {
			String path = ((FileSplit) context.getInputSplit()).getPath().toString();
			String[] items = value.toString().split(",",2);
			String outK = items[0].trim();	//key
			String outLine = items[1];		//value
			//Decide the kind of input
			if(path.indexOf("/U")>-1){ 	//latent factors for user
				outLine += ",U";
				outKey.set(outK+",MU");
				outText.set(outLine);
				context.write(outKey, outText);
				outKey.set(outK+",MS");
				outText.set(outLine);
				context.write(outKey, outText);
			} else {
				if(path.indexOf("/MU")>-1){
					outK += ",MU";
				} else if(path.indexOf("/MV")>-1){
					outK += ",MV";
				} else if(path.indexOf("/MS")>-1){
					outK += ",MS";
				}
				outKey.set(outK);
				outText.set(outLine);
				context.write(outKey, outText);
			}
		}
		
		@Override
		protected void setup(Context context) throws IOException, InterruptedException{
			super.setup(context);
		}
	}
	
	
	/**
	 * Mapper for SGD
	 * @author tigerzhong
	 *
	 */
	public static class SGDReducer extends Reducer<Text, Text, Text, Text> {
		/**
		 * Latent vector for the current user
		 */
		protected DenseDoubleMatrix1D U = null;
		/**
		 * Output key
		 */
		private Text outKey = new Text();
		/**
		 * Output value
		 */
		private Text outValue = new Text();
		@Override
		public void reduce(Text key, Iterable<Text> values,
				Context context) throws IOException, InterruptedException {
			String[] head = key.toString().split(",",-1);
			int p = Integer.parseInt(head[0]);	//first entry
			String c = head[1];					//category
			Map<Integer, Double> valueList = new HashMap<Integer, Double>();
			for(Text value : values){
				String items[] = value.toString().split(",",-1);
				if(items[items.length-1].equals("U")){
					for(int i=0;i<items.length-1;i++)
						U.set(i, Double.parseDouble(items[i]));	
				} else{ 										//Data
					valueList.put(Integer.parseInt(items[0]), Double.parseDouble(items[1]));
				}
			}
			int numR = valueList.size();
			//Compute gradient, and update U at the same time
			for(Integer q : valueList.keySet()){
				double r = valueList.get(q);	//Rating
				DoubleMatrix1D a = null, b = null;
				if(c.equalsIgnoreCase("MU")){
					a = lossFunc.getGradient(U, V.viewRow(q), r, lambda);
					b = lossFunc.getGradient(V.viewRow(q), U, r, lambda);
					SeqBlas.seqBlas.daxpy(-learningRate/numR, a, U);	//update U
					outKey.set("V,"+q);
					outValue.set(StringUtils.array2String(b.toArray()));
					context.write(outKey, outValue);
				} else if(c.equalsIgnoreCase("MS")){
					a = lossFunc.getGradient(U, S.viewRow(q), r, lambda);
					b = lossFunc.getGradient(S.viewRow(q), U, r, lambda);
					a.assign(Mult.mult(lambdaS));
					b.assign(Mult.mult(lambdaS));
					SeqBlas.seqBlas.daxpy(-learningRate/numR, a, U);	//update U
					outKey.set("S,"+q);
					outValue.set(StringUtils.array2String(b.toArray()));
					context.write(outKey, outValue);
				} else if(c.equalsIgnoreCase("MV")){
					a = lossFunc.getGradient(V.viewRow(p), S.viewRow(q), r, lambda);
					b = lossFunc.getGradient(S.viewRow(q), V.viewRow(p), r, lambda);
					a.assign(Mult.mult(lambdaV));
					b.assign(Mult.mult(lambdaV));
					outKey.set("V,"+p);
					outValue.set(StringUtils.array2String(a.toArray()));
					context.write(outKey, outValue);
					outKey.set("S,"+q);
					outValue.set(StringUtils.array2String(b.toArray()));
					context.write(outKey, outValue);
				}
			}
			outKey.set("U,"+p);
			outValue.set(StringUtils.array2String(U.toArray()));
			context.write(outKey, outValue);
		}
		
		@Override
		protected void setup(Context context) throws IOException, InterruptedException{
			super.setup(context);
			Configuration conf = context.getConfiguration();
			//Read U, V, S
			Path[] paths = DistributedCache.getLocalCacheFiles(conf);
			V = MatrixIO.readDenseMatrixFromText(paths[0].toString());
			S = MatrixIO.readDenseMatrixFromText(paths[1].toString());
			DistributedSGDCollective.numD = conf.getInt("model.numD", 10);
			DistributedSGDCollective.lambda = conf.getFloat("model.lambda", 0.005f);
			DistributedSGDCollective.lambdaS = conf.getFloat("model.lambdaS", 0.005f);
			DistributedSGDCollective.lambdaV = conf.getFloat("model.lambdaV", 0.005f);
			DistributedSGDCollective.lossFunc = ModelOperator.string2Loss(conf,conf.get("model.loss"));
			U = new DenseDoubleMatrix1D(DistributedSGDCollective.numD);
		}
	}

	/**
	 * Second mapper, distributing V while outputting U
	 * @author tigerzhong
	 *
	 */
	public static class CombinerMapper extends Mapper<LongWritable, Text, Text, Text> {
		/**
		 * Output stream for U
		 */
		protected FSDataOutputStream uFW = null;
		/**
		 * Output key
		 */
		private Text outKey = new Text();
		/**
		 * Output value
		 */
		private Text outText = new Text();
		@Override
		public void map(LongWritable key, Text value, Context context)
				throws IOException, InterruptedException {
			String[] items = value.toString().split("\t");
			if(items[0].startsWith("U")){
				uFW.write((items[0].split(",")[1]+","+items[1]+"\n").getBytes());
			} else {
				outKey.set(items[0]);
				outText.set(items[1]);
				context.write(outKey, outText);
			}
		}
		
		@Override
		protected void setup(Context context) throws IOException, InterruptedException{
			super.setup(context);
			Configuration conf = context.getConfiguration();
			FileSystem fs = FileSystem.get(context.getConfiguration()); 
			DistributedSGDCollective.numD = context.getConfiguration().getInt("model.numD", 10);	//number of latent dimension
			DistributedSGDCollective.learningRate = context.getConfiguration().getFloat("model.learningRate", (float) 0.1);
			uFW = fs.create(new Path(conf.get("hadoop.cache.path")+"U/"+((long)(Math.random()*1000000000))+".dat"));
		}
		
		@Override
		protected void cleanup(Context context) throws IOException, InterruptedException {
			super.cleanup(context);
			uFW.close();
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
