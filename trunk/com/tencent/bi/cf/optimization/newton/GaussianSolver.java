package com.tencent.bi.cf.optimization.newton;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.net.URI;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Map;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.filecache.DistributedCache;
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
import com.tencent.bi.utils.StringUtils;
import com.tencent.bi.utils.hadoop.FileOperators;

import cern.colt.matrix.DoubleMatrix1D;
import cern.colt.matrix.DoubleMatrix2D;
import cern.colt.matrix.impl.DenseDoubleMatrix1D;
import cern.colt.matrix.impl.DenseDoubleMatrix2D;
import cern.jet.math.Mult;

/**
 * Newton Solver, no finished yet!!!!
 * @author tigerzhong
 */
@Deprecated
public class GaussianSolver implements MFSolver{
	/**
	 * Number of latent factors
	 */
	protected static int numD = 10;
	/**
	 * Regularization Term
	 */
	protected static double lambda = 0.0;
	/**
	 * Output Key
	 */
	private static Text outKey = new Text();
	/**
	 * Output Text
	 */
	private static Text outText = new Text();
	
	@Override
	public void initialize(Loss lossFunc, double lambda, double learningRate,
			int numD) throws Exception {
		GaussianSolver.numD = numD;
		GaussianSolver.lambda = lambda;
	}

	@Override
	public void update(DenseDoubleMatrix2D U, DenseDoubleMatrix2D V,
			String inputPath, String outputPath) throws Exception {
		Configuration conf = FileOperators.getConfiguration();
		String catchPath = conf.get("hadoop.cache.path");
		conf.setInt("com.tencent.mf.model.numD", numD);
		conf.setFloat("com.tencent.mf.model.lambda", (float) lambda);
		//First MR, u_i*R_ij
		Job job = new Job(conf);
		job.setJarByClass(GaussianSolver.class);
		job.setJobName("MF-Newton-Step1");		
		job.setOutputKeyClass(Text.class);
		job.setOutputValueClass(Text.class);
		job.setMapperClass(RowMapper.class);
		job.setReducerClass(RowReducer.class);
		job.setInputFormatClass(TextInputFormat.class);
		job.setOutputFormatClass(TextOutputFormat.class);
		job.setNumReduceTasks(numD*50);
		FileInputFormat.addInputPath(job, new Path(inputPath));
		if(U!=null)
			FileInputFormat.addInputPath(job, new Path(catchPath+"V.dat"));
		else FileInputFormat.addInputPath(job, new Path(catchPath+"U.dat"));
		FileOutputFormat.setOutputPath(job, new Path(outputPath+"S1"));
		job.waitForCompletion(true);
		//Second MR, sum_i(u_i*R_ij)
		job = new Job(conf);
		job.setJarByClass(GaussianSolver.class);
		job.setJobName("MF-Newton-Step2");		
		job.setOutputKeyClass(Text.class);
		job.setOutputValueClass(Text.class);
		job.setMapperClass(ColMapper.class);
		job.setReducerClass(ColReducer.class);
		job.setInputFormatClass(TextInputFormat.class);
		job.setOutputFormatClass(TextOutputFormat.class);
		job.setNumReduceTasks(numD*50);
		FileInputFormat.addInputPath(job, new Path(outputPath+"S1"));
		FileOutputFormat.setOutputPath(job, new Path(outputPath+"S2"));
		job.waitForCompletion(true);
		//Third MR, UU^T
		job = new Job(conf);
		job.setJarByClass(GaussianSolver.class);
		job.setJobName("MF-Newton-Step3");		
		job.setOutputKeyClass(Text.class);
		job.setOutputValueClass(Text.class);
		job.setMapperClass(SMMapper.class);
		job.setReducerClass(SMReducer.class);
		job.setInputFormatClass(TextInputFormat.class);
		job.setOutputFormatClass(TextOutputFormat.class);
		job.setNumReduceTasks(numD*50);
		if(U!=null)
			FileInputFormat.addInputPath(job, new Path(catchPath+"V.dat"));
		else FileInputFormat.addInputPath(job, new Path(catchPath+"U.dat"));
		FileOutputFormat.setOutputPath(job, new Path(outputPath+"S3"));
		job.waitForCompletion(true);
		//Fourth MR, UU^TV
		job = new Job(conf);
		job.setJarByClass(GaussianSolver.class);
		job.setJobName("MF-Newton-Step3");		
		job.setOutputKeyClass(Text.class);
		job.setOutputValueClass(Text.class);
		job.setMapperClass(SVMapper.class);
		job.setInputFormatClass(TextInputFormat.class);
		job.setOutputFormatClass(TextOutputFormat.class);
		job.setNumReduceTasks(0);
		if(U!=null)
			FileInputFormat.addInputPath(job, new Path(catchPath+"U.dat"));
		else FileInputFormat.addInputPath(job, new Path(catchPath+"V.dat"));
		FileOutputFormat.setOutputPath(job, new Path(outputPath+"S4"));
		DistributedCache.addCacheFile(new URI(outputPath+"S3/part*"), job.getConfiguration());
		job.waitForCompletion(true);
		//Fifth MR, Update
		job = new Job(conf);
		job.setJarByClass(GaussianSolver.class);
		job.setJobName("MF-Newton-Step3");		
		job.setOutputKeyClass(Text.class);
		job.setOutputValueClass(Text.class);
		job.setMapperClass(EmitMapper.class);
		job.setReducerClass(EmitReducer.class);
		job.setInputFormatClass(TextInputFormat.class);
		job.setOutputFormatClass(TextOutputFormat.class);
		job.setNumReduceTasks(numD*50);
		FileInputFormat.addInputPath(job, new Path(outputPath+"S4"));
		FileInputFormat.addInputPath(job, new Path(outputPath+"S2"));
		if(U!=null)
			FileInputFormat.addInputPath(job, new Path(catchPath+"U.dat"));
		else FileInputFormat.addInputPath(job, new Path(catchPath+"V.dat"));
		FileOutputFormat.setOutputPath(job, new Path(outputPath+"S5"));
		job.waitForCompletion(true);
	}
	
///////////////////////////////////////////////////////////////////////////////////////////////
	
	public static class RowMapper extends Mapper<LongWritable, Text, Text, Text>{
		@Override
		public void map(LongWritable key, Text value, Context context)
				throws IOException, InterruptedException {
			String[] items = value.toString().split(",");
			outKey.set(((int)Double.parseDouble(items[0]))+"");
			StringBuilder res = new StringBuilder("");
			for(int i=1;i<items.length;i++){
				if(i!=1) res.append(",");
				res.append(items[i]);
			}
			outText.set(res.toString());
			context.write(outKey, outText);
		}
	}
	
	public static class RowReducer extends Reducer<Text, Text, Text, Text> {
		
		@Override
		public void reduce(Text key, Iterable<Text> values, Context context) throws IOException, InterruptedException {
			//Current vector
			DoubleMatrix1D U = new DenseDoubleMatrix1D(numD);
			//Value
			Map<Integer, Double> vList = new HashMap<Integer, Double>();
			Iterator<Text> it = values.iterator();
			while(it.hasNext()){
				String[] items = it.next().toString().split(",");
				if(items.length==2){ //Data
					vList.put(Integer.parseInt(items[0]), Double.parseDouble(items[1]));
				}
				else {//U
					for(int i=0;i<items.length;i++){
						U.set(i, Double.parseDouble(items[i]));
					}
				}
			}
			//Output
			for(Integer k : vList.keySet()){
				DoubleMatrix1D u = U.assign(Mult.mult(vList.get(k)));
				outKey.set(k+"");
				outText.set(StringUtils.array2String(u.toArray()));
				context.write(outKey, outText);
			}
		}
		
		@Override
		protected void setup(Context context) throws IOException, InterruptedException{
			Configuration conf = context.getConfiguration();
			GaussianSolver.numD = conf.getInt("com.tencent.mf.model.numD", 10);
		}
	}
	
///////////////////////////////////////////////////////////////////////////////////////////////
	
	public static class ColMapper extends Mapper<LongWritable, Text, Text, Text>{
		@Override
		public void map(LongWritable key, Text value, Context context)
				throws IOException, InterruptedException {
			String[] items = value.toString().split("\t");
			outKey.set(items[0]);
			outText.set(items[1]);
			context.write(outKey, outText);
		}
	}
	
	public static class ColReducer extends Reducer<Text, Text, Text, Text> {
		
		@Override
		public void reduce(Text key, Iterable<Text> values, Context context) throws IOException, InterruptedException {
			//Current vector
			DoubleMatrix1D U = new DenseDoubleMatrix1D(numD);
			Iterator<Text> it = values.iterator();
			//Combine
			while(it.hasNext()){
				String[] items = it.next().toString().split(",");
				for(int i=0;i<items.length;i++){
					U.set(i, Double.parseDouble(items[i])+U.get(i));
				}
			}
			//Output
			outKey.set(key);
			outText.set(StringUtils.array2String(U.toArray()));
			context.write(outKey, outText);
		}
		
		@Override
		protected void setup(Context context) throws IOException, InterruptedException{
			Configuration conf = context.getConfiguration();
			GaussianSolver.numD = conf.getInt("com.tencent.mf.model.numD", 10);
		}
	}

///////////////////////////////////////////////////////////////////////////////////////////////
	
	public static class SMMapper extends Mapper<LongWritable, Text, Text, Text>{
		@Override
		public void map(LongWritable key, Text value, Context context)
				throws IOException, InterruptedException {
			String[] items = value.toString().split(",");
			outKey.set("0");
			DoubleMatrix1D U = new DenseDoubleMatrix1D(numD);
			DenseDoubleMatrix2D S = new DenseDoubleMatrix2D(numD, numD);
			for(int i=0;i<numD;i++){
				U.set(i, Double.parseDouble(items[i+1]));
			}
			for(int i=0;i<numD;i++)
				for(int j=0;j<numD;j++)
					S.set(i, j, U.get(i)*U.get(j));
			outText.set(StringUtils.array2String(S.like1D(numD*numD).toArray()));
			context.write(outKey, outText);
		}
		@Override
		protected void setup(Context context) throws IOException, InterruptedException{
			Configuration conf = context.getConfiguration();
			GaussianSolver.numD = conf.getInt("com.tencent.mf.model.numD", 10);
		}
	}
	
	public static class SMReducer extends Reducer<Text, Text, Text, Text> {
		
		@Override
		public void reduce(Text key, Iterable<Text> values, Context context) throws IOException, InterruptedException {
			//Current vector
			DoubleMatrix1D S = new DenseDoubleMatrix1D(numD*numD);
			Iterator<Text> it = values.iterator();
			//Combine
			while(it.hasNext()){
				String[] items = it.next().toString().split(",");
				for(int i=0;i<items.length;i++){
					S.set(i, Double.parseDouble(items[i])+S.get(i));
				}
			}
			for(int i=0;i<S.size();i++){
				if(i%numD==0)
					S.set(i, S.get(i)+GaussianSolver.lambda);
			}
			//Output
			outKey.set(key);
			outText.set(StringUtils.array2String(S.toArray()));
			context.write(outKey, outText);
		}
		
		@Override
		protected void setup(Context context) throws IOException, InterruptedException{
			Configuration conf = context.getConfiguration();
			GaussianSolver.numD = conf.getInt("com.tencent.mf.model.numD", 10);
			GaussianSolver.lambda = conf.getInt("com.tencent.mf.model.lambda", 10);
		}
	}

///////////////////////////////////////////////////////////////////////////////////////////////
	public static class SVMapper extends Mapper<LongWritable, Text, Text, Text>{
		
		protected DoubleMatrix2D S = null;
		
		@Override
		public void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
			String[] items = value.toString().split(",");
			outKey.set(((int)Double.parseDouble(items[0]))+"");
			DoubleMatrix1D V = new DenseDoubleMatrix1D(numD);
			for(int i=1;i<items.length;i++){
				V.set(i-1, Double.parseDouble(items[i]));
			}
			DoubleMatrix1D rV = new DenseDoubleMatrix1D(numD);
			rV = S.zMult(V, rV);
			outText.set(StringUtils.array2String(rV.toArray()));
			context.write(outKey, outText);
		}
		
		@Override
		protected void setup(Context context) throws IOException, InterruptedException{
			Configuration conf = context.getConfiguration();
			GaussianSolver.numD = conf.getInt("com.tencent.mf.model.numD", 10);
			GaussianSolver.lambda = conf.getInt("com.tencent.mf.model.lambda", 10);
			S = new DenseDoubleMatrix2D(numD,numD);
			Path[] paths = DistributedCache.getLocalCacheFiles(conf);
			BufferedReader fis = new BufferedReader(new FileReader(paths[0].toString()));
			String line = fis.readLine();
			String[] items = line.split("\t")[1].split(",");
			int pt = 0;
			for(int i=0;i<numD;i++)
				for(int j=0;j<numD;j++)
					S.set(i, j, Double.parseDouble(items[pt++]));
		}
	}

///////////////////////////////////////////////////////////////////////////////////////////////
	public static class EmitMapper extends Mapper<LongWritable, Text, Text, Text>{
		@Override
		public void map(LongWritable key, Text value, Context context)
				throws IOException, InterruptedException {
			String[] items = value.toString().split("\t");
			outKey.set(items[0]);	
			String path = ((FileSplit) context.getInputSplit()).getPath().getName().toString();
			if(path.indexOf("S2")>-1){
				outText.set("X"+","+items[1]);
			} else if(path.indexOf("S3")>-1){
				outText.set("Y"+","+items[1]);
			} else {
				outText.set("D"+","+items[1]);
			}
			context.write(outKey, outText);
		}
	}
	
	public static class EmitReducer extends Reducer<Text, Text, Text, Text> {
		
		@Override
		public void reduce(Text key, Iterable<Text> values, Context context) throws IOException, InterruptedException {
			//Current vector
			DoubleMatrix1D D = new DenseDoubleMatrix1D(numD);
			DoubleMatrix1D X = new DenseDoubleMatrix1D(numD);
			DoubleMatrix1D Y = new DenseDoubleMatrix1D(numD);
			Iterator<Text> it = values.iterator();
			//Combine
			while(it.hasNext()){
				String[] items = it.next().toString().split(",");
				if(items[0].equalsIgnoreCase("X")){
					for(int i=1;i<items.length;i++){
						X.set(i-1, Double.parseDouble(items[i]));
					}
				}
				if(items[0].equalsIgnoreCase("Y")){
					for(int i=1;i<items.length;i++){
						Y.set(i-1, Double.parseDouble(items[i]));
					}
				}
				if(items[0].equalsIgnoreCase("D")){
					for(int i=1;i<items.length;i++){
						D.set(i-1, Double.parseDouble(items[i]));
					}
				}
			}
			for(int i=0;i<numD;i++){
				D.set(i, D.get(i)*X.get(i)/Y.get(i));
			}
			//Output
			outKey.set(key);
			outText.set(StringUtils.array2String(D.toArray()));
			context.write(outKey, outText);
		}
		
		@Override
		protected void setup(Context context) throws IOException, InterruptedException{
			Configuration conf = context.getConfiguration();
			GaussianSolver.numD = conf.getInt("com.tencent.mf.model.numD", 10);
		}
	}
}
