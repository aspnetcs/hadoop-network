package com.tencent.bi.cf.optimization.gradient.sparse;

import java.io.IOException;
import java.net.URI;
import java.util.HashMap;
import java.util.Map;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.filecache.DistributedCache;
import org.apache.hadoop.fs.FSDataOutputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.lib.input.FileSplit;
import org.apache.hadoop.mapreduce.lib.input.SequenceFileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.TextInputFormat;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.SequenceFileOutputFormat;

import com.tencent.bi.cf.optimization.Loss;
import com.tencent.bi.cf.optimization.MFSolver;
import com.tencent.bi.cf.optimization.gradient.common.CombineCombiner;
import com.tencent.bi.cf.optimization.gradient.common.CombineReducer;
import com.tencent.bi.cf.optimization.gradient.common.Extraction;
import com.tencent.bi.cf.optimization.gradient.common.LongPartition.ModLongPairPartitioner;
import com.tencent.bi.cf.optimization.gradient.common.LongPartition.ModLongPartitioner;
import com.tencent.bi.utils.StringUtils;
import com.tencent.bi.utils.hadoop.FileOperators;
import com.tencent.bi.utils.hadoop.ModelOperator;
import com.tencent.bi.utils.io.MatrixIO;
import com.tencent.bi.utils.serialization.LongPairWritable;
import com.tencent.bi.utils.serialization.MatrixRowWritable;

import cern.colt.matrix.DoubleMatrix1D;
import cern.colt.matrix.impl.DenseDoubleMatrix1D;
import cern.colt.matrix.impl.DenseDoubleMatrix2D;
import cern.colt.matrix.linalg.SeqBlas;

/**
 * Distributed SGD
 * @author tigerzhong
 *
 */
public class DistributedSGD implements MFSolver{
	/**
	 * Loss function
	 */
	protected static Loss lossFunc = null;
	/**
	 * Regularization parameter
	 */
	protected static double lambda = 0.005;
	/**
	 * Learning rate
	 */
	protected static double learningRate = 0.01;
	/**
	 * Number of latent dimension
	 */
	protected static int numD = 10;
	
	@Override
	public void initialize(Loss lossFunc, double lambda, double learningRate,
			int numD) throws Exception {
		DistributedSGD.lossFunc = lossFunc;
		DistributedSGD.lambda = lambda;
		DistributedSGD.learningRate = learningRate;
		DistributedSGD.numD = numD;
	}

	@Override
	public void update(DenseDoubleMatrix2D U, DenseDoubleMatrix2D V, String inputPath, String outputPath) throws Exception {
		//Initialization
		Configuration conf = FileOperators.getConfiguration();
		String catchPath = conf.get("hadoop.cache.path");
		String tmpPath = conf.get("hadoop.tmp.path")+"DistributedSGD/";
		conf.set("model.loss", ModelOperator.loss2String(conf, DistributedSGD.lossFunc));
		conf.setInt("model.numD", numD);
		conf.setFloat("model.learningRate", (float) learningRate);
		conf.setFloat("model.lambda", (float) lambda);
		MatrixIO.saveDenseMatrix2D2HDFS(conf, catchPath+"V.dat", V);
		
		//First MR, get gradients, and update U
		Job job = new Job(conf);
		job.setJarByClass(DistributedSGD.class);
		job.setJobName("MF-DistributedSGD-Update-Phase1");
		job.setMapOutputKeyClass(LongWritable.class);
		job.setMapOutputValueClass(MatrixRowWritable.class);
		job.setOutputKeyClass(LongPairWritable.class);
		job.setOutputValueClass(MatrixRowWritable.class);
		job.setMapperClass(SGDMapper.class);
		job.setPartitionerClass(ModLongPartitioner.class);
		job.setReducerClass(SGDReducer.class);
		job.setInputFormatClass(TextInputFormat.class);
		job.setOutputFormatClass(SequenceFileOutputFormat.class);
		FileInputFormat.addInputPath(job, new Path(inputPath));
		FileInputFormat.addInputPath(job, new Path(catchPath+"U/"));
		FileOutputFormat.setOutputPath(job, new Path(tmpPath));
		DistributedCache.addCacheFile(new URI(catchPath+"V.dat"), job.getConfiguration());
		job.waitForCompletion(true);
		
		//Clear
		FileSystem fs = FileSystem.get(conf); 
		fs.delete(new Path(catchPath+"U/"), true);
		
		//Second MR, update V
		job = new Job(conf);
		job.setJarByClass(DistributedSGD.class);
		job.setJobName("MF-DistributedSGD-Update-Phase2");
		job.setMapOutputKeyClass(LongPairWritable.class);
		job.setMapOutputValueClass(MatrixRowWritable.class);
		job.setOutputKeyClass(LongPairWritable.class);
		job.setOutputValueClass(MatrixRowWritable.class);
		job.setMapperClass(CombineMapper.class);
		job.setCombinerClass(CombineCombiner.class);
		job.setPartitionerClass(ModLongPairPartitioner.class);
		job.setReducerClass(CombineReducer.class);
		job.setInputFormatClass(SequenceFileInputFormat.class);
		job.setOutputFormatClass(SequenceFileOutputFormat.class);
		FileInputFormat.setInputPaths(job, new Path(tmpPath));
		FileOutputFormat.setOutputPath(job, new Path(outputPath));
		job.waitForCompletion(true);
		
		//Postprocess
		Extraction.extract(V, new Configuration(), outputPath, learningRate);
		fs.delete(new Path(tmpPath), true);		
	}	
	
	/**
	 * Mapper for SGD, distributing data
	 * @author tigerzhong
	 *
	 */
	public static class SGDMapper extends Mapper<LongWritable, Text, LongWritable, MatrixRowWritable>{
		/**
		 * Output key
		 */
		private LongWritable outKey = new LongWritable();
		/**
		 * Output value
		 */
		private MatrixRowWritable outValue = new MatrixRowWritable();
		
		@Override
		public void map(LongWritable key, Text value, Context context)
				throws IOException, InterruptedException {
			String path = ((FileSplit) context.getInputSplit()).getPath().toString();
			String[] items = value.toString().split(",");
			outKey.set(Long.parseLong(items[0]));
			double[] vec = new double[items.length-1];
			if(path.indexOf("/U")>-1){	//latent factors for user
				for(int i=0;i<items.length-1;i++)
					vec[i] = Double.parseDouble(items[i+1]);
			} else {					//data
				outValue.set(Long.parseLong(items[1]), Double.parseDouble(items[2]));						
				context.write(outKey, outValue);
				return;
			}
			outValue.set(vec);
			context.write(outKey, outValue);
		}
	}
	
	/**
	 * First reducer, for computing the gradient of V and updating U
	 * @author tigerzhong
	 *
	 */
	public static class SGDReducer extends Reducer<LongWritable, MatrixRowWritable, LongPairWritable, MatrixRowWritable> {

		/**
		 * Latent Matrix for the Current User
		 */
		protected DenseDoubleMatrix1D U = null;
		/**
		 * Latent matrix for items
		 */
		protected DenseDoubleMatrix2D V = null;
		/**
		 * Output key
		 */
		private LongPairWritable outKey = new LongPairWritable();
		/**
		 * Output value
		 */
		private MatrixRowWritable outValue = new MatrixRowWritable();
		
		private Map<Integer, Double> valueList = new HashMap<Integer, Double>(1024);
		
		@Override
		public void reduce(LongWritable key, Iterable<MatrixRowWritable> values,
				Context context) throws IOException, InterruptedException {
			valueList.clear();
			//Get Data
			for(MatrixRowWritable value : values){
				if (!value.isSparse()){	//U
					U = (DenseDoubleMatrix1D) value.getDenseVector().copy();	
				} else{ 										//Data
					valueList.put((int) value.getFirstID(), value.getFirstVal());
				}
			}
			int numR = valueList.size();
			//Compute Gradient, and update U at the same time
			for(Integer q : valueList.keySet()){
				double r = valueList.get(q);	//Rating
				DoubleMatrix1D u = lossFunc.getGradient(U, V.viewRow(q), r, lambda);	//U
				SeqBlas.seqBlas.daxpy(-learningRate/numR, u, U);
				DoubleMatrix1D v = lossFunc.getGradient(V.viewRow(q), U, r, lambda);	//V
				/*Compute Square Error, for monitor*/
				long cnt = (long)(lossFunc.getValue(U, V.viewRow(q), r)*1000);
				context.getCounter("Eval", "Loss").increment(cnt);
				context.getCounter("Eval", "Cnt").increment(1);
				//Output the gradient of V
				outKey.set(q,0);
				outValue.set(v.toArray());
				context.write(outKey, outValue);
			}
			//Output the new U
			outKey.set(key.get(), 1);
			outValue.set(U.toArray());
			context.write(outKey, outValue);
		}
		
		@Override
		protected void setup(Context context) throws IOException, InterruptedException{
			super.setup(context);
			Configuration conf = context.getConfiguration();
			//Read U and V
			Path[] paths = DistributedCache.getLocalCacheFiles(conf);
			DistributedSGD.numD = conf.getInt("model.numD", 10);
			DistributedSGD.lambda = conf.getFloat("model.lambda", 0.005f);
			DistributedSGD.lossFunc = ModelOperator.string2Loss(conf,conf.get("model.loss"));
			U = new DenseDoubleMatrix1D(DistributedSGD.numD);
			V = MatrixIO.readDenseMatrixFromText(paths[0].toString());
		}
	}
	
	/**
	 * Second mapper, distributing V while outputting U
	 * @author tigerzhong
	 *
	 */
	public static class CombineMapper extends Mapper<LongPairWritable, MatrixRowWritable, LongPairWritable, MatrixRowWritable> {
		/**
		 * Output stream for U
		 */
		protected FSDataOutputStream uFW = null;
		/**
		 * Output value
		 */
		private MatrixRowWritable outValue = new MatrixRowWritable();
		@Override
		public void map(LongPairWritable key, MatrixRowWritable value, Context context)
				throws IOException, InterruptedException {
			if(key.getSecond()==1){
				uFW.write((key.getFirst()+","+StringUtils.array2String(value.getVector())+"\n").getBytes());
			} else {
				//Others
				double[] vec = value.viewVector();
				double[] outVec = new double[vec.length+1];
				outVec[0] = 1.0;	//count
				for(int i=0;i<vec.length;i++) outVec[i+1] = vec[i];
				outValue.set(outVec);
				context.write(key, outValue);
			}
		}
		
		@Override
		protected void setup(Context context) throws IOException, InterruptedException{
			super.setup(context);
			Configuration conf = context.getConfiguration();
			FileSystem fs = FileSystem.get(conf); 
			uFW = fs.create(new Path(conf.get("hadoop.cache.path")+"U/"+((long)(Math.random()*1000000000))+".dat"));
		}
		
		@Override
		protected void cleanup(Context context) throws IOException, InterruptedException {
			super.cleanup(context);
			uFW.close();
		}
	}
	
}
